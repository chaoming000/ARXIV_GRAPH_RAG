"""
智能查询路由器（ARXIV版本）
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

from .types import Document


logger = logging.getLogger(__name__)


ALLOWED_ENTITY_TYPES = [
    "Author",
    "Paper",
    "Category",
    "Keyword",
]

ALLOWED_RELATION_TYPES = [
    "AUTHORED",
    "IN_CATEGORY",
    "HAS_KEYWORD",
    "RELATED_TO",
]

GRAPH_ENTITY_TYPES = {"Author", "Paper", "Category", "Keyword"}
GRAPH_RELATION_TYPES = {"AUTHORED", "IN_CATEGORY", "HAS_KEYWORD", "RELATED_TO"}

ENTITY_TYPE_GUIDE = {
    "Author": "作者姓名实体，用于作者-论文关系查询。示例：Maria Marina, Geoffrey Hinton。",
    "Paper": "论文实体，通常是完整标题或paper_id。示例：Beyond Detection: Rethinking Education in the Age of AI-writing。",
    "Category": "arXiv类别实体。示例：cs.CL, cs.AI, stat.ML。",
    "Keyword": "主题词/方法词/任务词等自由文本主题。示例：AI-writing, alignment, retrieval augmented generation。",
}

RELATION_TYPE_GUIDE = {
    "AUTHORED": "作者与论文关系，问题包含“作者是谁/谁写的/一起发表”等。",
    "IN_CATEGORY": "论文属于某arXiv类别，问题包含“属于什么类别/哪个学科”。",
    "HAS_KEYWORD": "论文包含某关键词主题，问题包含“关于XX的论文/关键词为XX”。",
    "RELATED_TO": "论文间关联关系，用于“相关论文/相似论文/关系网络/子图邻接”。",
}


class SearchStrategy(Enum):
    HYBRID_TRADITIONAL = "hybrid_traditional"
    GRAPH_RAG = "graph_rag"


@dataclass
class QueryEntity:
    text: str
    entity_type: str


@dataclass
class QueryRelation:
    text: str
    relation_type: str


@dataclass
class QueryAnalysis:
    query_complexity: float
    relationship_intensity: float
    reasoning_required: bool
    entity_count: int
    recommended_strategy: SearchStrategy
    confidence: float
    reasoning: str
    entities: List[QueryEntity] = field(default_factory=list)
    relations: List[QueryRelation] = field(default_factory=list)
    source: str = "rule"


class IntelligentQueryRouter:
    def __init__(self, traditional_retrieval, graph_rag_retrieval, llm_client, config):
        self.traditional_retrieval = traditional_retrieval
        self.graph_rag_retrieval = graph_rag_retrieval
        self.llm_client = llm_client
        self.config = config
        self.route_stats = {
            "traditional_count": 0,
            "graph_rag_count": 0,
            "total_queries": 0,
        }

    def analyze_query(self, query: str) -> QueryAnalysis:
        if self.llm_client:
            try:
                llm_data = self._llm_extract_entities_relations(query)
                entities = self._parse_entities(llm_data.get("entities", []))
                relations = self._parse_relations(llm_data.get("relations", []))
                return self._build_analysis_from_structured(
                    query=query,
                    entities=entities,
                    relations=relations,
                    source="llm",
                    llm_reason=str(llm_data.get("reason", "")).strip(),
                    llm_confidence=float(llm_data.get("confidence", 0.0) or 0.0),
                    llm_route_decision=str(llm_data.get("route_decision", "")).strip(),
                )
            except Exception as exc:
                logger.warning("LLM route analysis failed, fallback to rules: %s", exc)
        return self._rule_based_analysis(query)

    def _llm_extract_entities_relations(self, query: str) -> Dict[str, Any]:
        entity_guide = "\n".join([f"- {k}: {v}" for k, v in ENTITY_TYPE_GUIDE.items()])
        relation_guide = "\n".join([f"- {k}: {v}" for k, v in RELATION_TYPE_GUIDE.items()])
        prompt = f"""
你是 ARXIV GraphRAG 的“路由分析器”，只做结构化分析，不做问答。

你必须只输出一个 JSON 对象，禁止输出 markdown、解释文本、代码块、前后缀。

【目标】
从用户问题中抽取：
1) entities: 实体列表
2) relations: 关系列表
3) route_decision: 路由决策

【允许实体类型】(entity_type 只能取以下值)
{ALLOWED_ENTITY_TYPES}
实体类型说明：
{entity_guide}

【允许关系类型】(relation_type 只能取以下值)
{ALLOWED_RELATION_TYPES}
关系类型说明：
{relation_guide}

【路由决策规则】
route_decision 只能是：
- "graph_rag"
- "hybrid_traditional"

优先选择 graph_rag 的情形：
- 问题显式问“作者-论文”“论文-关键词”“论文-类别”“论文之间关系”
- 问题包含两个及以上实体并要求关系/路径/共同点（如“谁和谁一起发表过什么”）
- 问题是结构化关系查询（如“某作者参与哪些论文”）

优先选择 hybrid_traditional 的情形：
- 主题推荐、语义相似、摘要解读、开放式检索（关系约束弱）

【抽取规则】
- entities 中保留原文片段，不要改写。
- 人名、论文名、类别、关键词尽量都提取。
- 若问题是“X和Y一起发表过哪些论文”，应提取两个 Author 实体，并包含 AUTHORED 关系。
- 若实体或关系不确定，可返回空数组 []，不要编造。

【输出格式】严格遵循：
{{
  "entities": [
    {{"text": "xxx", "entity_type": "Author"}}
  ],
  "relations": [
    {{"text": "xxx", "relation_type": "AUTHORED"}}
  ],
  "route_decision": "graph_rag",
  "confidence": 0.0,
  "reason": "一句话原因"
}}

【Few-shot 示例】
示例1
问题: Beyond Detection: Rethinking Education in the Age of AI-writing的作者有谁
输出:
{{
  "entities": [
    {{"text": "Beyond Detection: Rethinking Education in the Age of AI-writing", "entity_type": "Paper"}}
  ],
  "relations": [
    {{"text": "作者有谁", "relation_type": "AUTHORED"}}
  ],
  "route_decision": "graph_rag",
  "confidence": 0.95,
  "reason": "明确的论文到作者关系查询"
}}

示例2
问题: 推荐几篇AI-writing相关论文
输出:
{{
  "entities": [
    {{"text": "AI-writing", "entity_type": "Keyword"}}
  ],
  "relations": [],
  "route_decision": "hybrid_traditional",
  "confidence": 0.86,
  "reason": "主题推荐，关系约束弱"
}}

示例3
问题: Maria和Marina一起发表过哪些论文
输出:
{{
  "entities": [
    {{"text": "Maria", "entity_type": "Author"}},
    {{"text": "Marina", "entity_type": "Author"}}
  ],
  "relations": [
    {{"text": "一起发表", "relation_type": "AUTHORED"}}
  ],
  "route_decision": "graph_rag",
  "confidence": 0.91,
  "reason": "多实体关系查询，需要图结构连接"
}}

示例4
问题: 总结最近的多模态模型趋势
输出:
{{
  "entities": [
    {{"text": "多模态模型", "entity_type": "Keyword"}}
  ],
  "relations": [],
  "route_decision": "hybrid_traditional",
  "confidence": 0.84,
  "reason": "开放式总结任务，优先语义检索"
}}

用户问题:
{query}
"""
        resp = self._llm_route_request(prompt)
        return self._parse_llm_json((resp.choices[0].message.content or "").strip())

    def _build_analysis_from_structured(
        self,
        query: str,
        entities: List[QueryEntity],
        relations: List[QueryRelation],
        source: str,
        llm_reason: str = "",
        llm_confidence: float = 0.0,
        llm_route_decision: str = "",
    ) -> QueryAnalysis:
        graph_entity_types = {e.entity_type for e in entities if e.entity_type in GRAPH_ENTITY_TYPES}
        has_graph_relation = any(r.relation_type in GRAPH_RELATION_TYPES for r in relations)
        has_multi_type_entities = len(graph_entity_types) >= 2

        llm_decision = (llm_route_decision or "").strip().lower()
        if llm_decision in {"graph_rag", "hybrid_traditional"}:
            strategy = SearchStrategy(llm_decision)
            relation_intensity = 0.85 if strategy == SearchStrategy.GRAPH_RAG else 0.25
            complexity = 0.65 if strategy == SearchStrategy.GRAPH_RAG else 0.45
            reason = f"采用LLM路由决策: {llm_decision}"
            confidence = max(0.75, min(1.0, llm_confidence or 0.0))
        elif has_graph_relation or has_multi_type_entities:
            strategy = SearchStrategy.GRAPH_RAG
            relation_intensity = 0.85 if has_graph_relation else 0.65
            complexity = 0.7 if has_multi_type_entities else 0.55
            reason = "检测到图关系或多类型实体，优先图检索"
            confidence = 0.85
        else:
            strategy = SearchStrategy.HYBRID_TRADITIONAL
            relation_intensity = 0.2
            complexity = 0.45
            reason = "未检测到图关系且实体类型单一，优先向量检索"
            confidence = 0.7

        if llm_reason:
            reason = f"{reason}; LLM: {llm_reason}"
        if llm_confidence > 0:
            confidence = max(confidence, min(1.0, llm_confidence))

        return QueryAnalysis(
            query_complexity=complexity,
            relationship_intensity=relation_intensity,
            reasoning_required=has_graph_relation or has_multi_type_entities,
            entity_count=max(1, len(entities)),
            recommended_strategy=strategy,
            confidence=confidence,
            reasoning=reason,
            entities=entities,
            relations=relations,
            source=source,
        )

    def _rule_based_analysis(self, query: str) -> QueryAnalysis:
        q = (query or "").lower()
        relation_keywords = [
            "关系",
            "涉及",
            "关联",
            "合作",
            "发表",
            "发文",
            "共同",
            "一起",
            "路径",
            "子图",
            "relation",
            "related",
            "involve",
            "collaboration",
            "publish",
            "published",
            "coauthor",
            "path",
            "subgraph",
            "author",
            "authors",
        ]
        relation_hits = [kw for kw in relation_keywords if kw in q]

        entities: List[QueryEntity] = []
        if any(k in q for k in ["author", "authors", "作者"]):
            entities.append(QueryEntity(text="author_query", entity_type="Author"))
        if any(k in q for k in ["paper", "papers", "论文"]):
            entities.append(QueryEntity(text="paper_query", entity_type="Paper"))
        if any(k in q for k in ["category", "类别", "cs.", "分类"]):
            entities.append(QueryEntity(text="category_query", entity_type="Category"))
        if any(k in q for k in ["keyword", "关键词"]):
            entities.append(QueryEntity(text="keyword_query", entity_type="Keyword"))
        if not entities:
            entities.append(QueryEntity(text=query.strip()[:48], entity_type="Keyword"))

        relations: List[QueryRelation] = []
        if relation_hits:
            relations.append(QueryRelation(text="keyword_relation_signal", relation_type="RELATED_TO"))
        if any(k in q for k in ["发表", "发文", "publish", "published", "coauthor", "共同", "一起"]):
            relations.append(QueryRelation(text="authored_signal", relation_type="AUTHORED"))

        # 简单抽取“X和Y一起发表...”中的两个作者名
        if any(k in q for k in ["发表", "发文", "publish", "coauthor", "一起", "共同"]) and "和" in query:
            left, right = query.split("和", 1)
            right = re.split(r"(一起|共同|发表|发文|论文|paper|papers|publish)", right, maxsplit=1)[0]
            left_name = left.strip(" ，,。:：?？\"'“”")
            right_name = right.strip(" ，,。:：?？\"'“”")
            if left_name:
                entities.append(QueryEntity(text=left_name, entity_type="Author"))
            if right_name:
                entities.append(QueryEntity(text=right_name, entity_type="Author"))

        # 英文双作者模式："A and B published ..."
        and_match = re.search(r"([A-Za-z][A-Za-z .-]{1,40})\s+and\s+([A-Za-z][A-Za-z .-]{1,40})", query)
        if and_match and any(k in q for k in ["publish", "published", "paper", "papers", "coauthor"]):
            entities.append(QueryEntity(text=and_match.group(1).strip(), entity_type="Author"))
            entities.append(QueryEntity(text=and_match.group(2).strip(), entity_type="Author"))

        return self._build_analysis_from_structured(
            query=query,
            entities=entities,
            relations=relations,
            source="rule",
            llm_reason="关键词回退策略",
            llm_confidence=0.6,
        )

    def route_query(self, query: str, top_k: int = 5) -> Tuple[List[Document], QueryAnalysis]:
        analysis = self.analyze_query(query)
        self._update_route_stats(analysis.recommended_strategy)
        try:
            if analysis.recommended_strategy == SearchStrategy.GRAPH_RAG:
                docs = self.graph_rag_retrieval.graph_rag_search(query, top_k, analysis=analysis)
                if not docs:
                    docs = self.traditional_retrieval.vector_search_enhanced(query, top_k)
                    for d in docs:
                        d.metadata["fallback_reason"] = "graph_failed_or_empty"
                        d.metadata["fallback_to"] = "vector"
            else:
                docs = self.traditional_retrieval.vector_search_enhanced(query, top_k)
            docs = self._post_process_results(docs, analysis)
            return docs, analysis
        except Exception as exc:
            logger.error("Route query failed: %s", exc)
            docs = self.traditional_retrieval.vector_search_enhanced(query, top_k)
            for d in docs:
                d.metadata["fallback_reason"] = "route_exception"
                d.metadata["fallback_to"] = "vector"
            return self._post_process_results(docs, analysis), analysis

    def _post_process_results(self, documents: List[Document], analysis: QueryAnalysis) -> List[Document]:
        for doc in documents:
            doc.metadata.update(
                {
                    "route_strategy": analysis.recommended_strategy.value,
                    "query_complexity": analysis.query_complexity,
                    "route_confidence": analysis.confidence,
                    "route_source": analysis.source,
                }
            )
        return documents

    def _update_route_stats(self, strategy: SearchStrategy):
        self.route_stats["total_queries"] += 1
        if strategy == SearchStrategy.HYBRID_TRADITIONAL:
            self.route_stats["traditional_count"] += 1
        else:
            self.route_stats["graph_rag_count"] += 1

    def get_route_statistics(self) -> Dict[str, Any]:
        total = self.route_stats["total_queries"]
        if total == 0:
            return self.route_stats.copy()
        return {
            **self.route_stats,
            "traditional_ratio": self.route_stats["traditional_count"] / total,
            "graph_rag_ratio": self.route_stats["graph_rag_count"] / total,
        }

    def explain_routing_decision(self, query: str) -> str:
        a = self.analyze_query(query)
        entity_str = ", ".join([f"{e.text}:{e.entity_type}" for e in a.entities]) or "none"
        relation_str = ", ".join([f"{r.text}:{r.relation_type}" for r in a.relations]) or "none"
        return (
            f"query={query}\n"
            f"entities={entity_str}\n"
            f"relations={relation_str}\n"
            f"complexity={a.query_complexity:.2f}, relation={a.relationship_intensity:.2f}, "
            f"reasoning_required={a.reasoning_required}\n"
            f"strategy={a.recommended_strategy.value}, confidence={a.confidence:.2f}, source={a.source}\n"
            f"reason={a.reasoning}"
        )

    def _llm_route_request(self, prompt: str):
        kwargs = {
            "model": self.config.router_llm_model or self.config.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 600,
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
            raise ValueError("empty llm route response")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                raise
            return json.loads(match.group(0))

    def _parse_entities(self, raw_entities: Any) -> List[QueryEntity]:
        entities: List[QueryEntity] = []
        if not isinstance(raw_entities, list):
            return entities
        for item in raw_entities:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            entity_type = str(item.get("entity_type", "")).strip()
            if not text or entity_type not in ALLOWED_ENTITY_TYPES:
                continue
            entities.append(QueryEntity(text=text, entity_type=entity_type))
        return entities

    def _parse_relations(self, raw_relations: Any) -> List[QueryRelation]:
        relations: List[QueryRelation] = []
        if not isinstance(raw_relations, list):
            return relations
        for item in raw_relations:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            relation_type = str(item.get("relation_type", "")).strip()
            if not relation_type or relation_type not in ALLOWED_RELATION_TYPES:
                continue
            relations.append(QueryRelation(text=text or relation_type, relation_type=relation_type))
        return relations
