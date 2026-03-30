"""
图RAG检索模块（ARXIV版本）
图查询意图理解 -> 多跳/子图/路径检索 -> 图结果转文档。
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from app.graph import Neo4jGraphStore

from .types import Document


logger = logging.getLogger(__name__)


class QueryType(Enum):
    """图查询类型。"""

    ENTITY_RELATION = "entity_relation"
    MULTI_HOP = "multi_hop"
    SUBGRAPH = "subgraph"
    PATH_FINDING = "path_finding"
    CLUSTERING = "clustering"


@dataclass
class GraphQuery:
    """图查询结构。"""

    query_type: QueryType
    source_entities: List[str]
    target_entities: List[str] = field(default_factory=list)
    relation_types: List[str] = field(default_factory=list)
    max_depth: int = 2
    max_nodes: int = 50
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphPath:
    """路径查询结果。"""

    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    path_length: int
    relevance_score: float
    path_type: str
    paper_id: str = ""


@dataclass
class KnowledgeSubgraph:
    """子图查询结果。"""

    central_nodes: List[Dict[str, Any]]
    connected_nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    graph_metrics: Dict[str, float]
    reasoning_chains: List[str]
    paper_ids: List[str] = field(default_factory=list)


class GraphRAGRetrieval:
    """ARXIV 图检索实现。"""

    def __init__(self, config, llm_client=None):
        self.config = config
        self.llm_client = llm_client
        self.graph_store: Optional[Neo4jGraphStore] = None

        self.entity_cache: Dict[str, Dict[str, Any]] = {}
        self.relation_cache: Dict[str, int] = {}
        self.subgraph_cache: Dict[str, KnowledgeSubgraph] = {}

    def initialize(self):
        self.graph_store = Neo4jGraphStore(
            uri=self.config.neo4j_uri,
            user=self.config.neo4j_user,
            password=self.config.neo4j_password,
            database=self.config.neo4j_database,
        )
        self._build_graph_index()

    def _build_graph_index(self):
        """预热常见实体与关系类型，先构图索引。"""
        if not self.graph_store:
            return

        self.entity_cache = {}
        self.relation_cache = {}

        try:
            with self.graph_store.driver.session(database=self.graph_store.database) as session:
                entity_query = """
                MATCH (n)
                WHERE
                  (
                    n:Author OR n:Keyword OR n:Category
                    OR (n:Paper AND n.source_tag = $source_tag)
                  )
                WITH n, size([(n)--() | 1]) AS degree
                RETURN labels(n) AS node_labels,
                       coalesce(n.paper_id, n.name, n.title, n.tag) AS node_id,
                       coalesce(n.name, n.title, n.paper_id, n.tag) AS name,
                       degree
                ORDER BY degree DESC
                LIMIT 1000
                """
                for record in session.run(entity_query, {"source_tag": self.config.source_tag}):
                    node_id = str(record.get("node_id", "") or "").strip()
                    if not node_id:
                        continue
                    self.entity_cache[node_id] = {
                        "labels": list(record.get("node_labels", []) or []),
                        "name": str(record.get("name", "") or ""),
                        "degree": int(record.get("degree", 0) or 0),
                    }

                relation_query = """
                MATCH ()-[r]-()
                RETURN type(r) AS rel_type, count(r) AS frequency
                ORDER BY frequency DESC
                """
                for record in session.run(relation_query):
                    rel_type = str(record.get("rel_type", "") or "").strip()
                    if rel_type:
                        self.relation_cache[rel_type] = int(record.get("frequency", 0) or 0)

        except Exception as exc:
            logger.warning("Build graph index failed: %s", exc)

    def graph_rag_search(self, query: str, top_k: int = 5, analysis: Any = None) -> List[Document]:
        if not self.graph_store:
            return []

        graph_query = self.understand_graph_query(query, analysis=analysis)
        logger.info(
            "Graph query understood: type=%s sources=%s targets=%s relations=%s",
            graph_query.query_type.value,
            graph_query.source_entities,
            graph_query.target_entities,
            graph_query.relation_types,
        )

        try:
            if graph_query.query_type in {QueryType.MULTI_HOP, QueryType.PATH_FINDING, QueryType.ENTITY_RELATION}:
                paths = self.multi_hop_traversal(graph_query)
                docs = self._paths_to_documents(paths, query)
            else:
                subgraph = self.extract_knowledge_subgraph(graph_query)
                reasoning_chains = self.graph_structure_reasoning(subgraph, query)
                docs = self._subgraph_to_documents(subgraph, reasoning_chains, query)

            ranked = self._rank_by_graph_relevance(docs, query)
            return ranked[:top_k]
        except Exception as exc:
            logger.error("Graph retrieval failed: %s", exc)
            return []

    def understand_graph_query(self, query: str, analysis: Any = None) -> GraphQuery:
        """
        先将自然语言问题映射为图查询计划。
        """
        structured_hints = self._analysis_hints_to_json(analysis)
        if self.llm_client:
            prompt = f"""
你是 ARXIV GraphRAG 的图查询规划器。你的任务不是回答问题，而是把自然语言问题映射到已有图结构。

图中节点和关系如下：
- Paper：包含 title、paper_id、summary、published、url、primary_category、keywords_text、source_tag
- Author：作者姓名
- Category：arXiv 分类，例如 cs.CL、cs.AI
- Keyword：主题词/方法词/任务词
- Source：来源标签

关系：
- (Author)-[:AUTHORED]->(Paper)
- (Paper)-[:IN_CATEGORY]->(Category)
- (Paper)-[:HAS_KEYWORD]->(Keyword)
- (Paper)-[:FROM_SOURCE]->(Source)
- (Paper)-[:RELATED_TO]->(Paper)

请根据该图结构分析下面的问题，并只返回一个 JSON 对象，不要输出任何额外说明。

问题：{query}

路由层已给出的实体/关系提示，可作为参考但不能机械照抄：
{structured_hints}

你需要输出这些字段：
1. query_type：
   - entity_relation：问直接关系，例如“这篇论文作者是谁”
   - multi_hop：需要跨两个及以上关系推理，例如“某作者和某主题之间通过哪些论文连接”
   - subgraph：围绕一个主题/实体展开局部知识网络，例如“AI-writing 相关论文网络”
   - path_finding：找两个实体之间最短或最关键路径
   - clustering：找与某论文/主题相近的一组论文

2. source_entities：
   - 只放可能能映射到 Author / Paper / Category / Keyword / Source 节点的具体实体
   - 例如作者名、论文标题、paper_id、分类、关键词

3. target_entities：
   - 只有在问题明确要求路径终点或目标实体时才填写
   - 不确定时返回 []

4. relation_types：
   - 只允许从 ["AUTHORED", "IN_CATEGORY", "HAS_KEYWORD", "RELATED_TO", "FROM_SOURCE"] 中选择

5. max_depth：
   - 1 到 4 的整数

6. constraints：
   - 可选过滤条件，例如：
     {{
       "published_after": "2024-01-01",
       "published_before": "2025-12-31",
       "latest_only": true
     }}

示例1：
问题："Beyond Detection: Rethinking Education in the Age of AI-writing 的作者是谁"
输出：
{{
  "query_type": "entity_relation",
  "source_entities": ["Beyond Detection: Rethinking Education in the Age of AI-writing"],
  "target_entities": ["Author"],
  "relation_types": ["AUTHORED"],
  "max_depth": 1,
  "constraints": {{}}
}}

示例2：
问题："Maria Marina 和 AI-writing 之间有哪些论文连接"
输出：
{{
  "query_type": "multi_hop",
  "source_entities": ["Maria Marina"],
  "target_entities": ["AI-writing"],
  "relation_types": ["AUTHORED", "HAS_KEYWORD"],
  "max_depth": 3,
  "constraints": {{}}
}}

示例3：
问题："cs.CL 中和检索增强生成相关的论文网络"
输出：
{{
  "query_type": "subgraph",
  "source_entities": ["cs.CL", "retrieval augmented generation"],
  "target_entities": [],
  "relation_types": ["IN_CATEGORY", "HAS_KEYWORD", "RELATED_TO"],
  "max_depth": 2,
  "constraints": {{}}
}}
"""
            try:
                response = self._llm_request(prompt, max_tokens=900)
                payload = self._parse_llm_json((response.choices[0].message.content or "").strip())
                return self._build_graph_query_from_payload(payload, query, analysis)
            except Exception as exc:
                logger.warning("Understand graph query via LLM failed, fallback to rules: %s", exc)

        return self._rule_based_graph_query(query, analysis)

    def multi_hop_traversal(self, graph_query: GraphQuery) -> List[GraphPath]:
        if not self.graph_store:
            return []

        try:
            with self.graph_store.driver.session(database=self.graph_store.database) as session:
                if graph_query.query_type == QueryType.ENTITY_RELATION:
                    paths = self._find_entity_relations(graph_query, session)
                    if paths:
                        return paths
                if graph_query.query_type == QueryType.PATH_FINDING:
                    paths = self._find_shortest_paths(graph_query, session)
                    if paths:
                        return paths

                target_filter_clause = ""
                if graph_query.target_entities:
                    target_filter_clause = """
                  AND ANY(term IN $target_terms WHERE
                    (
                      target:Author AND toLower(term) IN ['author', 'authors', '作者']
                    ) OR (
                      target:Paper AND toLower(term) IN ['paper', 'papers', '论文']
                    ) OR (
                      target:Category AND toLower(term) IN ['category', 'categories', '类别', '分类']
                    ) OR (
                      target:Keyword AND toLower(term) IN ['keyword', 'keywords', '关键词', '主题']
                    ) OR (
                      target:Source AND toLower(term) IN ['source', '来源']
                    ) OR (
                      target:Author AND toLower(coalesce(target.name, '')) CONTAINS toLower(term)
                    ) OR (
                      target:Keyword AND toLower(coalesce(target.name, '')) CONTAINS toLower(term)
                    ) OR (
                      target:Category AND toLower(coalesce(target.name, '')) CONTAINS toLower(term)
                    ) OR (
                      target:Paper AND (
                        toLower(coalesce(target.title, '')) CONTAINS toLower(term)
                        OR toLower(coalesce(target.paper_id, '')) CONTAINS toLower(term)
                      )
                    ) OR (
                      target:Source AND toLower(coalesce(target.tag, '')) CONTAINS toLower(term)
                    )
                  )
"""

                query = f"""
                UNWIND $source_terms AS source_term
                MATCH (source)
                WHERE {self._entity_match_clause('source', 'source_term')}
                MATCH path = (source)-[*1..{max(1, min(4, graph_query.max_depth))}]-(target)
                WHERE source <> target
                  AND ANY(n IN nodes(path) WHERE n:Paper AND n.source_tag = $source_tag){target_filter_clause}
                WITH source, target, path,
                     length(path) AS path_len,
                     relationships(path) AS rels,
                     nodes(path) AS path_nodes,
                     CASE
                       WHEN target:Paper THEN target
                       ELSE head([n IN reverse(nodes(path)) WHERE n:Paper AND n.source_tag = $source_tag])
                     END AS anchor_paper
                WHERE anchor_paper IS NOT NULL
                WITH source, target, path, path_len, rels, path_nodes, anchor_paper,
                     size([r IN rels WHERE type(r) IN $relation_types]) AS relation_hits,
                     reduce(s = 0.0, n IN path_nodes | s + toFloat(size([(n)--() | 1]))) /
                     CASE WHEN size(path_nodes) = 0 THEN 1.0 ELSE toFloat(size(path_nodes)) END AS avg_degree
                RETURN anchor_paper.paper_id AS paper_id,
                       path_len,
                       rels,
                       path_nodes,
                       (
                         (1.0 / toFloat(path_len)) +
                         (CASE WHEN size($relation_types) = 0 THEN 0.0 ELSE 0.35 * toFloat(relation_hits) / toFloat(path_len) END) +
                         (0.15 * avg_degree / 10.0)
                       ) AS relevance
                ORDER BY relevance DESC, path_len ASC
                LIMIT $limit
                """

                params = {
                    "source_terms": graph_query.source_entities[:12],
                    "target_terms": graph_query.target_entities[:12],
                    "relation_types": graph_query.relation_types,
                    "source_tag": self.config.source_tag,
                    "limit": max(10, min(graph_query.max_nodes, 60)),
                }

                results: List[GraphPath] = []
                for record in session.run(query, params):
                    path = self._parse_neo4j_path(record, path_type=graph_query.query_type.value)
                    if path:
                        results.append(path)
                return results
        except Exception as exc:
            logger.error("Multi-hop traversal failed: %s", exc)
            return []

    def extract_knowledge_subgraph(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        if not self.graph_store:
            return self._fallback_subgraph_extraction(graph_query)

        cache_key = json.dumps(
            {
                "sources": graph_query.source_entities,
                "targets": graph_query.target_entities,
                "relations": graph_query.relation_types,
                "depth": graph_query.max_depth,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if cache_key in self.subgraph_cache:
            return self.subgraph_cache[cache_key]

        try:
            with self.graph_store.driver.session(database=self.graph_store.database) as session:
                query = f"""
                UNWIND $source_terms AS source_term
                MATCH (source)
                WHERE {self._entity_match_clause('source', 'source_term')}
                MATCH path = (source)-[*1..{max(1, min(4, graph_query.max_depth))}]-(neighbor)
                WHERE ANY(n IN nodes(path) WHERE n:Paper AND n.source_tag = $source_tag)
                WITH collect(DISTINCT source)[0..12] AS sources,
                     collect(DISTINCT neighbor)[0..$max_nodes] AS neighbors,
                     collect(DISTINCT relationships(path))[0..$max_nodes] AS rel_paths,
                     collect(DISTINCT [n IN nodes(path) WHERE n:Paper AND n.source_tag = $source_tag | n.paper_id])[0..$max_nodes] AS paper_path_ids
                RETURN sources, neighbors, rel_paths, paper_path_ids
                """
                record = session.run(
                    query,
                    {
                        "source_terms": graph_query.source_entities[:12],
                        "source_tag": self.config.source_tag,
                        "max_nodes": max(10, min(graph_query.max_nodes, 80)),
                    },
                ).single()
                if not record:
                    subgraph = self._fallback_subgraph_extraction(graph_query)
                else:
                    subgraph = self._build_knowledge_subgraph(record)
                self.subgraph_cache[cache_key] = subgraph
                return subgraph
        except Exception as exc:
            logger.error("Subgraph extraction failed: %s", exc)
            return self._fallback_subgraph_extraction(graph_query)

    def graph_structure_reasoning(self, subgraph: KnowledgeSubgraph, query: str) -> List[str]:
        reasoning_chains: List[str] = []
        try:
            patterns = self._identify_reasoning_patterns(subgraph)
            for pattern in patterns:
                chain = self._build_reasoning_chain(pattern, subgraph)
                if chain:
                    reasoning_chains.append(chain)
            return self._validate_reasoning_chains(reasoning_chains, query)
        except Exception as exc:
            logger.warning("Graph structure reasoning failed: %s", exc)
            return []

    def _paths_to_documents(self, paths: List[GraphPath], query: str) -> List[Document]:
        if not paths or not self.graph_store:
            return []

        seen_ids: Set[str] = set()
        ordered_ids: List[str] = []
        best_scores: Dict[str, float] = {}
        best_paths: Dict[str, GraphPath] = {}

        for path in paths:
            pid = str(path.paper_id or "").strip()
            if not pid:
                continue
            score = float(path.relevance_score or 0.0)
            if pid not in best_scores or score > best_scores[pid]:
                best_scores[pid] = score
                best_paths[pid] = path
            if pid not in seen_ids:
                seen_ids.add(pid)
                ordered_ids.append(pid)

        details = self.graph_store.get_papers_by_ids(ordered_ids, source_tag=self.config.source_tag)
        detail_map = {str(item.get("paper_id", "")).strip(): item for item in details}

        documents: List[Document] = []
        for pid in ordered_ids:
            detail = detail_map.get(pid)
            path = best_paths.get(pid)
            if not detail or not path:
                continue
            path_desc = self._build_path_description(path)
            content = self._build_paper_content(detail, graph_context=path_desc)
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "search_type": "graph_path",
                        "graph_query_type": path.path_type,
                        "graph_score": float(best_scores.get(pid, 0.0)),
                        "relevance_score": float(best_scores.get(pid, 0.0)),
                        "path_length": path.path_length,
                        "node_count": len(path.nodes),
                        "relationship_count": len(path.relationships),
                        "paper_id": detail.get("paper_id", ""),
                        "title": detail.get("title", ""),
                        "url": detail.get("url", ""),
                        "authors": detail.get("authors", []),
                        "published": detail.get("published", ""),
                    },
                )
            )
        return documents

    def _subgraph_to_documents(
        self,
        subgraph: KnowledgeSubgraph,
        reasoning_chains: List[str],
        query: str,
    ) -> List[Document]:
        if not self.graph_store or not subgraph.paper_ids:
            return []

        details = self.graph_store.get_papers_by_ids(subgraph.paper_ids[:24], source_tag=self.config.source_tag)
        detail_map = {str(item.get("paper_id", "")).strip(): item for item in details}

        documents: List[Document] = []
        base_desc = self._build_subgraph_description(subgraph)
        for idx, paper_id in enumerate(subgraph.paper_ids):
            detail = detail_map.get(str(paper_id).strip())
            if not detail:
                continue

            paper_neighbors = self._paper_neighbors_from_subgraph(subgraph, paper_id)
            neighbor_preview = " | ".join(paper_neighbors[:10])
            graph_context = base_desc
            if reasoning_chains:
                graph_context += "\nreasoning_chains: " + " ; ".join(reasoning_chains[:3])
            if neighbor_preview:
                graph_context += "\nsubgraph_neighbors: " + neighbor_preview

            graph_score = self._subgraph_doc_score(subgraph, paper_id, rank_index=idx)
            documents.append(
                Document(
                    page_content=self._build_paper_content(detail, graph_context=graph_context),
                    metadata={
                        "search_type": "knowledge_subgraph",
                        "graph_query_type": QueryType.SUBGRAPH.value,
                        "graph_score": graph_score,
                        "relevance_score": graph_score,
                        "graph_density": subgraph.graph_metrics.get("density", 0.0),
                        "node_count": len(subgraph.connected_nodes),
                        "relationship_count": len(subgraph.relationships),
                        "reasoning_chains": reasoning_chains[:3],
                        "paper_id": detail.get("paper_id", ""),
                        "title": detail.get("title", ""),
                        "url": detail.get("url", ""),
                        "authors": detail.get("authors", []),
                        "published": detail.get("published", ""),
                    },
                )
            )
        return documents

    def _build_graph_query_from_payload(self, payload: Dict[str, Any], query: str, analysis: Any = None) -> GraphQuery:
        raw_type = str(payload.get("query_type", "") or "").strip().lower()
        try:
            query_type = QueryType(raw_type)
        except Exception:
            query_type = QueryType.SUBGRAPH

        source_entities = self._normalize_entity_terms(payload.get("source_entities", []))
        target_entities = self._normalize_entity_terms(payload.get("target_entities", []))
        relation_types = self._normalize_relation_types(payload.get("relation_types", []))
        max_depth = self._normalize_int(payload.get("max_depth"), default=2, minimum=1, maximum=4)
        constraints = payload.get("constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}

        if not source_entities:
            source_entities = self._analysis_entities(analysis) or [query.strip()]

        return GraphQuery(
            query_type=query_type,
            source_entities=source_entities[:12],
            target_entities=target_entities[:12],
            relation_types=relation_types[:8],
            max_depth=max_depth,
            max_nodes=50,
            constraints=constraints,
        )

    def _rule_based_graph_query(self, query: str, analysis: Any = None) -> GraphQuery:
        source_entities = self._analysis_entities(analysis)
        relation_types = self._analysis_relations(analysis)
        target_entities: List[str] = []

        text = str(query or "").strip()
        q = text.lower()
        query_type = QueryType.SUBGRAPH

        if any(kw in q for kw in ["子图", "subgraph", "网络", "network"]):
            query_type = QueryType.SUBGRAPH
        elif any(kw in q for kw in ["路径", "path", "connect", "connection", "between"]):
            query_type = QueryType.PATH_FINDING
        elif any(kw in q for kw in ["作者是谁", "who wrote", "who is the author", "作者有谁"]):
            query_type = QueryType.ENTITY_RELATION
            relation_types = relation_types or ["AUTHORED"]
            target_entities = ["Author"]
        elif any(kw in q for kw in ["关系", "相关", "共同", "一起", "连接", "之间", "collaboration", "coauthor", "relation"]):
            query_type = QueryType.MULTI_HOP
        elif any(kw in q for kw in ["相似", "类似", "similar", "related papers", "推荐"]):
            query_type = QueryType.CLUSTERING

        if not source_entities and query_type == QueryType.ENTITY_RELATION:
            for sep in ["的作者是谁", "的作者有谁", "作者是谁", "作者有谁"]:
                if sep in text:
                    candidate = text.split(sep, 1)[0].strip(" ：:，,。！？?\"“”")
                    if candidate:
                        source_entities = [candidate]
                        break
            if not source_entities:
                english_patterns = [
                    r"who\s+wrote\s+(.+)",
                    r"who\s+is\s+the\s+author\s+of\s+(.+)",
                ]
                for pattern in english_patterns:
                    match = re.search(pattern, q, flags=re.IGNORECASE)
                    if match:
                        candidate = match.group(1).strip(" :,.!?\"'")
                        if candidate:
                            source_entities = [candidate]
                            break
        if not source_entities and not target_entities and any(token in text for token in ["和", " between ", "之间"]):
            zh_match = re.match(r"(.+?)和(.+?)(?:之间|相关|有什么|有哪些|连接|路径)", text)
            if zh_match:
                left = zh_match.group(1).strip(" ：:，,。！？?\"“”")
                right = zh_match.group(2).strip(" ：:，,。！？?\"“”")
                left = re.sub(r"(中|里)$", "", left).strip()
                right = re.sub(r"^(与|和)", "", right).strip()
                if left and right:
                    if query_type == QueryType.SUBGRAPH:
                        source_entities = [left, right]
                    else:
                        source_entities = [left]
                        target_entities = [right]
            else:
                en_match = re.search(r"(.+?)\s+and\s+(.+?)\s+(?:between|related|connection|path)", text, flags=re.IGNORECASE)
                if en_match:
                    left = en_match.group(1).strip(" :,.!?\"'")
                    right = en_match.group(2).strip(" :,.!?\"'")
                    if left and right:
                        if query_type == QueryType.SUBGRAPH:
                            source_entities = [left, right]
                        else:
                            source_entities = [left]
                            target_entities = [right]
        if not source_entities:
            source_entities = self._extract_terms_from_query(text)
        if not source_entities:
            source_entities = [text]

        if not relation_types:
            if query_type == QueryType.ENTITY_RELATION:
                relation_types = ["AUTHORED"]
            elif query_type == QueryType.MULTI_HOP:
                relation_types = ["AUTHORED", "HAS_KEYWORD", "IN_CATEGORY", "RELATED_TO"]
            else:
                relation_types = ["HAS_KEYWORD", "IN_CATEGORY", "RELATED_TO"]

        max_depth = 3 if query_type in {QueryType.MULTI_HOP, QueryType.PATH_FINDING} else 2
        return GraphQuery(
            query_type=query_type,
            source_entities=source_entities[:12],
            target_entities=target_entities[:12],
            relation_types=relation_types[:8],
            max_depth=max_depth,
            max_nodes=50,
            constraints={},
        )

    def _parse_neo4j_path(self, record, path_type: str) -> Optional[GraphPath]:
        try:
            path_nodes: List[Dict[str, Any]] = []
            for node in record["path_nodes"]:
                if node is None:
                    continue
                path_nodes.append(
                    {
                        "id": str(node.get("paper_id", node.get("name", node.get("title", node.get("tag", "")))) or ""),
                        "name": str(node.get("name", node.get("title", node.get("paper_id", node.get("tag", "")))) or ""),
                        "labels": list(node.labels),
                        "properties": dict(node),
                    }
                )

            relationships: List[Dict[str, Any]] = []
            for rel in record["rels"]:
                relationships.append(
                    {
                        "type": str(type(rel).__name__),
                        "properties": dict(rel),
                    }
                )

            return GraphPath(
                nodes=path_nodes,
                relationships=relationships,
                path_length=int(record.get("path_len", 0) or 0),
                relevance_score=float(record.get("relevance", 0.0) or 0.0),
                path_type=path_type,
                paper_id=str(record.get("paper_id", "") or ""),
            )
        except Exception as exc:
            logger.warning("Parse neo4j path failed: %s", exc)
            return None

    def _build_knowledge_subgraph(self, record) -> KnowledgeSubgraph:
        central_nodes = []
        connected_nodes = []
        relationships = []
        paper_ids: List[str] = []
        seen_paper_ids: Set[str] = set()

        for node in record.get("sources", []) or []:
            if node is None:
                continue
            central_nodes.append(
                {
                    "id": str(node.get("paper_id", node.get("name", node.get("title", node.get("tag", "")))) or ""),
                    "name": str(node.get("name", node.get("title", node.get("paper_id", node.get("tag", "")))) or ""),
                    "labels": list(node.labels),
                    "properties": dict(node),
                }
            )

        for node in record.get("neighbors", []) or []:
            if node is None:
                continue
            item = {
                "id": str(node.get("paper_id", node.get("name", node.get("title", node.get("tag", "")))) or ""),
                "name": str(node.get("name", node.get("title", node.get("paper_id", node.get("tag", "")))) or ""),
                "labels": list(node.labels),
                "properties": dict(node),
            }
            connected_nodes.append(item)
            node_paper_id = str(node.get("paper_id", "") or "").strip()
            if node_paper_id and node_paper_id not in seen_paper_ids:
                seen_paper_ids.add(node_paper_id)
                paper_ids.append(node_paper_id)

        rel_count = 0
        for rel_path in record.get("rel_paths", []) or []:
            for rel in rel_path or []:
                relationships.append(
                    {
                        "type": str(type(rel).__name__),
                        "properties": dict(rel),
                    }
                )
                rel_count += 1

        for id_group in record.get("paper_path_ids", []) or []:
            for paper_id in id_group or []:
                pid = str(paper_id or "").strip()
                if pid and pid not in seen_paper_ids:
                    seen_paper_ids.add(pid)
                    paper_ids.append(pid)

        node_count = len(connected_nodes)
        density = 0.0
        if node_count > 1:
            density = min(1.0, float(rel_count) / float(node_count * (node_count - 1)))

        return KnowledgeSubgraph(
            central_nodes=central_nodes,
            connected_nodes=connected_nodes,
            relationships=relationships,
            graph_metrics={
                "node_count": float(node_count),
                "relationship_count": float(rel_count),
                "density": density,
            },
            reasoning_chains=[],
            paper_ids=paper_ids[:24],
        )

    def _build_path_description(self, path: GraphPath) -> str:
        if not path.nodes:
            return "空图路径"

        parts: List[str] = []
        for idx, node in enumerate(path.nodes):
            name = str(node.get("name", f"node_{idx}")).strip() or f"node_{idx}"
            parts.append(name)
            if idx < len(path.relationships):
                rel_type = str(path.relationships[idx].get("type", "RELATED")).strip() or "RELATED"
                parts.append(f" -[{rel_type}]-> ")
        return "".join(parts)

    def _build_subgraph_description(self, subgraph: KnowledgeSubgraph) -> str:
        center_names = [str(node.get("name", "")).strip() for node in subgraph.central_nodes if node.get("name")]
        center_text = ", ".join(center_names[:6]) if center_names else "查询实体"
        return (
            f"围绕 {center_text} 的知识子图："
            f"包含 {len(subgraph.connected_nodes)} 个相关节点、"
            f"{len(subgraph.relationships)} 条关系，"
            f"图密度 {subgraph.graph_metrics.get('density', 0.0):.3f}"
        )

    def _rank_by_graph_relevance(self, documents: List[Document], query: str) -> List[Document]:
        return sorted(
            documents,
            key=lambda doc: float(
                doc.metadata.get(
                    "graph_score",
                    doc.metadata.get("relevance_score", 0.0),
                )
                or 0.0
            ),
            reverse=True,
        )

    def _identify_reasoning_patterns(self, subgraph: KnowledgeSubgraph) -> List[str]:
        patterns: List[str] = []
        rel_types = {str(rel.get("type", "")).strip() for rel in subgraph.relationships if rel.get("type")}
        if "AUTHORED" in rel_types:
            patterns.append("author_paper_link")
        if "HAS_KEYWORD" in rel_types:
            patterns.append("topic_paper_link")
        if "IN_CATEGORY" in rel_types:
            patterns.append("category_grouping")
        if "RELATED_TO" in rel_types:
            patterns.append("paper_similarity")
        return patterns[:4]

    def _build_reasoning_chain(self, pattern: str, subgraph: KnowledgeSubgraph) -> Optional[str]:
        center_names = [str(node.get("name", "")).strip() for node in subgraph.central_nodes if node.get("name")]
        center_text = ", ".join(center_names[:3]) if center_names else "核心实体"
        if pattern == "author_paper_link":
            return f"{center_text} 可通过 Author -> Paper 关系定位作者相关论文。"
        if pattern == "topic_paper_link":
            return f"{center_text} 可通过 Keyword -> Paper 关系串联同主题论文。"
        if pattern == "category_grouping":
            return f"{center_text} 可通过 Category -> Paper 关系限定学科范围。"
        if pattern == "paper_similarity":
            return f"{center_text} 可通过 RELATED_TO 关系扩展到相似论文群。"
        return None

    def _validate_reasoning_chains(self, chains: List[str], query: str) -> List[str]:
        deduped: List[str] = []
        seen = set()
        for chain in chains:
            text = str(chain or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            deduped.append(text)
        return deduped[:3]

    def _find_entity_relations(self, graph_query: GraphQuery, session, path_type: str = QueryType.ENTITY_RELATION.value) -> List[GraphPath]:
        source_terms = graph_query.source_entities[:4]
        target_terms = graph_query.target_entities[:4]
        if not source_terms:
            return []

        if not target_terms and len(source_terms) >= 2:
            target_terms = [source_terms[-1]]
            source_terms = source_terms[:-1]

        if not target_terms:
            return []

        query = f"""
        UNWIND $source_terms AS source_term
        UNWIND $target_terms AS target_term
        MATCH (source)
        WHERE {self._entity_match_clause('source', 'source_term')}
        MATCH (target)
        WHERE {self._entity_match_clause('target', 'target_term')}
          AND source <> target
        MATCH path = shortestPath((source)-[*1..{max(1, min(4, graph_query.max_depth))}]-(target))
        WHERE path IS NOT NULL
        WITH path,
             length(path) AS path_len,
             relationships(path) AS rels,
             nodes(path) AS path_nodes,
             CASE
               WHEN target:Paper THEN target
               WHEN source:Paper THEN source
               ELSE head([n IN reverse(nodes(path)) WHERE n:Paper AND n.source_tag = $source_tag])
             END AS anchor_paper
        WHERE anchor_paper IS NOT NULL
        RETURN anchor_paper.paper_id AS paper_id,
               path_len,
               rels,
               path_nodes,
               (
                 (1.0 / toFloat(path_len)) +
                 0.4 * toFloat(size([r IN rels WHERE type(r) IN $relation_types])) /
                 CASE WHEN path_len = 0 THEN 1.0 ELSE toFloat(path_len) END
               ) AS relevance
        ORDER BY relevance DESC, path_len ASC
        LIMIT 20
        """

        paths: List[GraphPath] = []
        for record in session.run(
            query,
            {
                "source_terms": source_terms,
                "target_terms": target_terms,
                "relation_types": graph_query.relation_types or ["AUTHORED", "HAS_KEYWORD", "IN_CATEGORY", "RELATED_TO"],
                "source_tag": self.config.source_tag,
            },
        ):
            path = self._parse_neo4j_path(record, path_type=path_type)
            if path:
                paths.append(path)
        return paths

    def _find_shortest_paths(self, graph_query: GraphQuery, session) -> List[GraphPath]:
        target_terms = graph_query.target_entities[:4]
        source_terms = graph_query.source_entities[:4]
        if not source_terms or not target_terms:
            return []
        return self._find_entity_relations(
            GraphQuery(
                query_type=QueryType.PATH_FINDING,
                source_entities=source_terms,
                target_entities=target_terms,
                relation_types=graph_query.relation_types,
                max_depth=graph_query.max_depth,
                max_nodes=graph_query.max_nodes,
                constraints=graph_query.constraints,
            ),
            session,
            path_type=QueryType.PATH_FINDING.value,
        )

    def _fallback_subgraph_extraction(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        return KnowledgeSubgraph(
            central_nodes=[],
            connected_nodes=[],
            relationships=[],
            graph_metrics={},
            reasoning_chains=[],
            paper_ids=[],
        )

    def _build_paper_content(self, detail: Dict[str, Any], graph_context: str = "") -> str:
        authors = detail.get("authors", [])
        keywords = detail.get("keywords", [])
        categories = detail.get("categories", [])

        lines = [
            f"title: {detail.get('title', '')}",
            f"paper_id: {detail.get('paper_id', '')}",
            f"published: {detail.get('published', '')}",
            f"url: {detail.get('url', '')}",
            f"authors: {', '.join(authors) if isinstance(authors, list) else authors}",
            f"categories: {', '.join(categories) if isinstance(categories, list) else categories}",
            f"keywords: {', '.join(keywords) if isinstance(keywords, list) else keywords}",
            f"summary: {detail.get('summary', '')}",
        ]
        if graph_context:
            lines.append(f"graph_context: {graph_context}")
        return "\n".join(lines)

    def _paper_neighbors_from_subgraph(self, subgraph: KnowledgeSubgraph, paper_id: str) -> List[str]:
        previews: List[str] = []
        target = str(paper_id or "").strip()
        if not target:
            return previews
        for node in subgraph.connected_nodes:
            props = node.get("properties", {}) if isinstance(node, dict) else {}
            if str(props.get("paper_id", "") or "").strip() != target:
                continue
            node_name = str(node.get("name", target)).strip() or target
            node_labels = ",".join(node.get("labels", []) or [])
            previews.append(f"{node_name}({node_labels})")
        for rel in subgraph.relationships[:8]:
            rel_type = str(rel.get("type", "")).strip()
            if rel_type:
                previews.append(rel_type)
        deduped: List[str] = []
        seen = set()
        for item in previews:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped[:10]

    def _subgraph_doc_score(self, subgraph: KnowledgeSubgraph, paper_id: str, rank_index: int) -> float:
        density = float(subgraph.graph_metrics.get("density", 0.0) or 0.0)
        node_count = max(1.0, float(subgraph.graph_metrics.get("node_count", 0.0) or 1.0))
        coverage = min(1.0, len(subgraph.paper_ids) / node_count)
        rank_bonus = 1.0 / float(rank_index + 1)
        return 0.45 * density + 0.35 * coverage + 0.20 * rank_bonus

    def _analysis_hints_to_json(self, analysis: Any) -> str:
        if analysis is None:
            return '{"entities": [], "relations": []}'
        payload = {
            "entities": [
                {
                    "text": str(getattr(entity, "text", "")).strip(),
                    "entity_type": str(getattr(entity, "entity_type", "")).strip(),
                }
                for entity in getattr(analysis, "entities", []) or []
            ],
            "relations": [
                {
                    "text": str(getattr(rel, "text", "")).strip(),
                    "relation_type": str(getattr(rel, "relation_type", "")).strip(),
                }
                for rel in getattr(analysis, "relations", []) or []
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    def _analysis_entities(self, analysis: Any) -> List[str]:
        items: List[str] = []
        if analysis is None:
            return items
        for entity in getattr(analysis, "entities", []) or []:
            text = str(getattr(entity, "text", "")).strip()
            if text and text.lower() not in {"author_query", "paper_query", "category_query", "keyword_query"}:
                items.append(text)
        return self._normalize_entity_terms(items)

    def _analysis_relations(self, analysis: Any) -> List[str]:
        relation_types: List[str] = []
        if analysis is None:
            return relation_types
        for relation in getattr(analysis, "relations", []) or []:
            relation_type = str(getattr(relation, "relation_type", "")).strip().upper()
            if relation_type in {"AUTHORED", "IN_CATEGORY", "HAS_KEYWORD", "RELATED_TO", "FROM_SOURCE"}:
                relation_types.append(relation_type)
        return list(dict.fromkeys(relation_types))

    def _extract_terms_from_query(self, query: str) -> List[str]:
        if not query:
            return []

        quoted = re.findall(r"[\"“](.*?)[\"”]", query)
        paper_like = re.findall(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", query)
        arxiv_cat = re.findall(r"\b[a-z]{2}\.[A-Z]{2}\b", query, flags=re.IGNORECASE)

        tokens = quoted + paper_like + arxiv_cat
        if tokens:
            return self._normalize_entity_terms(tokens)

        cleaned = re.split(r"[，,。！？?;；\n]", query)
        candidates = [item.strip() for item in cleaned if len(item.strip()) >= 2]
        return self._normalize_entity_terms(candidates[:4])

    def _normalize_entity_terms(self, values: Any) -> List[str]:
        if not isinstance(values, list):
            return []
        items: List[str] = []
        seen = set()
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            items.append(text)
        return items

    def _normalize_relation_types(self, values: Any) -> List[str]:
        allowed = {"AUTHORED", "IN_CATEGORY", "HAS_KEYWORD", "RELATED_TO", "FROM_SOURCE"}
        if not isinstance(values, list):
            return []
        items: List[str] = []
        seen = set()
        for value in values:
            relation = str(value or "").strip().upper()
            if relation not in allowed or relation in seen:
                continue
            seen.add(relation)
            items.append(relation)
        return items

    def _normalize_int(self, value: Any, default: int, minimum: int, maximum: int) -> int:
        try:
            number = int(value)
        except Exception:
            number = default
        return max(minimum, min(maximum, number))

    def _entity_match_clause(self, variable_name: str, term_name: str) -> str:
        return f"""
(
  {variable_name}:Author AND toLower({term_name}) IN ['author', 'authors', '作者']
) OR (
  {variable_name}:Paper AND toLower({term_name}) IN ['paper', 'papers', '论文']
) OR (
  {variable_name}:Category AND toLower({term_name}) IN ['category', 'categories', '类别', '分类']
) OR (
  {variable_name}:Keyword AND toLower({term_name}) IN ['keyword', 'keywords', '关键词', '主题']
) OR (
  {variable_name}:Source AND toLower({term_name}) IN ['source', '来源']
) OR (
  {variable_name}:Author AND toLower(coalesce({variable_name}.name, '')) CONTAINS toLower({term_name})
) OR (
  {variable_name}:Keyword AND toLower(coalesce({variable_name}.name, '')) CONTAINS toLower({term_name})
) OR (
  {variable_name}:Category AND toLower(coalesce({variable_name}.name, '')) CONTAINS toLower({term_name})
) OR (
  {variable_name}:Paper AND (
    toLower(coalesce({variable_name}.title, '')) CONTAINS toLower({term_name})
    OR toLower(coalesce({variable_name}.paper_id, '')) CONTAINS toLower({term_name})
  )
) OR (
  {variable_name}:Source AND toLower(coalesce({variable_name}.tag, '')) CONTAINS toLower({term_name})
)
"""

    def _llm_request(self, prompt: str, max_tokens: int):
        kwargs = {
            "model": self.config.router_llm_model or self.config.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        try:
            return self.llm_client.chat.completions.create(
                **kwargs,
                response_format={"type": "json_object"},
            )
        except Exception:
            return self.llm_client.chat.completions.create(**kwargs)

    def _parse_llm_json(self, text: str) -> Dict[str, Any]:
        if not text:
            raise ValueError("empty graph planner response")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                raise
            return json.loads(match.group(0))

    def close(self):
        if self.graph_store:
            self.graph_store.close()
