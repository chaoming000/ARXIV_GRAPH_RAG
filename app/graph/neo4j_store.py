import logging
from dataclasses import asdict
from typing import Dict, List

from neo4j import GraphDatabase

from app.data import ArxivPaper


logger = logging.getLogger(__name__)


class Neo4jGraphStore:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        if self.driver:
            self.driver.close()

    def ensure_schema(self):
        statements = [
            "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
            "CREATE CONSTRAINT author_name_unique IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT category_name_unique IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT keyword_name_unique IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE",
            "CREATE CONSTRAINT source_tag_unique IF NOT EXISTS FOR (s:Source) REQUIRE s.tag IS UNIQUE",
        ]
        with self.driver.session(database=self.database) as session:
            for cypher in statements:
                session.run(cypher)

    def clear_source_tag(self, source_tag: str):
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (n {source_tag: $source_tag})
                DETACH DELETE n
                """,
                {"source_tag": source_tag},
            )

    def upsert_papers(self, papers: List[ArxivPaper], source_tag: str, batch_size: int = 50):
        if not papers:
            return

        payload = []
        for p in papers:
            item = asdict(p)
            item["keywords_text"] = ", ".join(item.get("keywords", []))
            payload.append(item)

        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MERGE (s:Source {tag: $source_tag})
                SET s.updated_at = datetime(),
                    s.paper_count = $paper_count
                """,
                {"source_tag": source_tag, "paper_count": len(payload)},
            )

            for start in range(0, len(payload), batch_size):
                batch = payload[start : start + batch_size]
                session.run(
                    """
                    UNWIND $papers AS p
                    MERGE (paper:Paper {paper_id: p.paper_id})
                    SET paper.title = p.title,
                        paper.summary = p.summary,
                        paper.published = p.published,
                        paper.updated = p.updated,
                        paper.url = p.url,
                        paper.primary_category = p.primary_category,
                        paper.keywords_text = p.keywords_text,
                        paper.source_tag = $source_tag
                    """,
                    {"papers": batch, "source_tag": source_tag},
                )
                session.run(
                    """
                    UNWIND $papers AS p
                    MATCH (paper:Paper {paper_id: p.paper_id})
                    UNWIND p.authors AS author_name
                    MERGE (a:Author {name: author_name})
                    SET a.source_tag = $source_tag
                    MERGE (a)-[:AUTHORED]->(paper)
                    """,
                    {"papers": batch, "source_tag": source_tag},
                )
                session.run(
                    """
                    UNWIND $papers AS p
                    MATCH (paper:Paper {paper_id: p.paper_id})
                    UNWIND p.categories AS category_name
                    MERGE (c:Category {name: category_name})
                    SET c.source_tag = $source_tag
                    MERGE (paper)-[:IN_CATEGORY]->(c)
                    """,
                    {"papers": batch, "source_tag": source_tag},
                )
                session.run(
                    """
                    UNWIND $papers AS p
                    MATCH (paper:Paper {paper_id: p.paper_id})
                    UNWIND p.keywords AS keyword
                    MERGE (k:Keyword {name: keyword})
                    SET k.source_tag = $source_tag
                    MERGE (paper)-[:HAS_KEYWORD]->(k)
                    """,
                    {"papers": batch, "source_tag": source_tag},
                )
                session.run(
                    """
                    UNWIND $papers AS p
                    MATCH (paper:Paper {paper_id: p.paper_id})
                    MATCH (s:Source {tag: $source_tag})
                    MERGE (paper)-[:FROM_SOURCE]->(s)
                    """,
                    {"papers": batch, "source_tag": source_tag},
                )

    def build_related_edges(self, source_tag: str, min_shared_keywords: int = 2):
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (p1:Paper {source_tag: $source_tag})-[r:RELATED_TO]-(p2:Paper {source_tag: $source_tag})
                DELETE r
                """,
                {"source_tag": source_tag},
            )

            session.run(
                """
                MATCH (p1:Paper {source_tag: $source_tag})-[:HAS_KEYWORD]->(k:Keyword)<-[:HAS_KEYWORD]-(p2:Paper {source_tag: $source_tag})
                WHERE p1.paper_id < p2.paper_id
                WITH p1, p2, count(DISTINCT k) AS shared_kw
                WHERE shared_kw >= $min_shared_keywords
                MERGE (p1)-[r:RELATED_TO]-(p2)
                SET r.shared_keywords = shared_kw,
                    r.source_tag = $source_tag
                """,
                {"source_tag": source_tag, "min_shared_keywords": min_shared_keywords},
            )

    def graph_retrieve(
        self,
        query_terms: List[str],
        top_k: int = 10,
        source_tag: str | None = None,
        retrieval_mode: str = "entity_relation",
    ) -> List[dict]:
        if not query_terms:
            return []

        norm_terms = [t.lower() for t in query_terms if t]
        mode = (retrieval_mode or "entity_relation").lower()
        score_map: Dict[str, float] = {}
        source_map: Dict[str, set] = {}

        with self.driver.session(database=self.database) as session:
            if mode == "entity_relation":
                self._merge_scores(
                    score_map,
                    source_map,
                    self._match_keyword_terms(session, norm_terms, source_tag, score=1.25),
                )
                self._merge_scores(
                    score_map,
                    source_map,
                    self._match_author_terms(session, norm_terms, source_tag, score=1.15),
                )
                self._merge_scores(
                    score_map,
                    source_map,
                    self._match_category_terms(session, norm_terms, source_tag, score=1.0),
                )
            else:
                self._merge_scores(
                    score_map,
                    source_map,
                    self._match_keyword_terms(session, norm_terms, source_tag, score=1.15),
                )
                self._merge_scores(
                    score_map,
                    source_map,
                    self._match_category_terms(session, norm_terms, source_tag, score=1.0),
                )
                self._merge_scores(
                    score_map,
                    source_map,
                    self._match_text_terms(session, norm_terms, source_tag, score=0.9),
                )

            ranked_ids = [
                pid
                for pid, _ in sorted(score_map.items(), key=lambda item: item[1], reverse=True)[: max(top_k * 3, 20)]
            ]
            if not ranked_ids:
                return []

            details = self._load_paper_details(session, ranked_ids, source_tag)
            for d in details:
                pid = d.get("paper_id", "")
                d["score"] = float(score_map.get(pid, 0.0))
                d["sources"] = sorted(source_map.get(pid, set()))
            details.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
            return details[:top_k]

    def retrieve_subgraph_for_entities(
        self,
        entity_terms: List[str],
        top_k: int = 20,
        source_tag: str | None = None,
        max_neighbors_per_entity: int = 30,
    ) -> List[dict]:
        """
        SUBGRAPH 检索：
        - 匹配问题中所有实体节点（Author/Paper/Category/Keyword）
        - 展开每个实体的一跳邻居
        - 汇总可达 Paper，并把“实体节点+邻居”上下文挂到结果上
        """
        if not entity_terms:
            return []

        terms: List[str] = []
        seen = set()
        for t in entity_terms:
            term = str(t or "").strip()
            if not term:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            terms.append(term)

        if not terms:
            return []

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                UNWIND $terms AS raw_term
                WITH trim(raw_term) AS raw_term, toLower(trim(raw_term)) AS term
                WHERE term <> ''
                MATCH (e)
                WHERE (
                    (e:Author AND toLower(e.name) CONTAINS term) OR
                    (e:Keyword AND toLower(e.name) CONTAINS term) OR
                    (e:Category AND toLower(e.name) CONTAINS term) OR
                    (e:Paper AND (toLower(coalesce(e.title, '')) CONTAINS term OR toLower(coalesce(e.paper_id, '')) CONTAINS term))
                )
                  AND ($source_tag IS NULL OR coalesce(e.source_tag, $source_tag) = $source_tag)
                OPTIONAL MATCH (e)-[r]-(n)
                WHERE n IS NOT NULL
                  AND ($source_tag IS NULL OR coalesce(n.source_tag, $source_tag) = $source_tag OR n:Source)
                WITH raw_term, e,
                     collect(DISTINCT {
                       relation: type(r),
                       node_type: CASE
                         WHEN n:Author THEN 'Author'
                         WHEN n:Paper THEN 'Paper'
                         WHEN n:Keyword THEN 'Keyword'
                         WHEN n:Category THEN 'Category'
                         WHEN n:Source THEN 'Source'
                         ELSE 'Entity'
                       END,
                       value: coalesce(n.name, n.title, n.paper_id),
                       paper_id: n.paper_id
                     })[0..$max_neighbors] AS neighbors
                RETURN raw_term AS query_term,
                       CASE
                         WHEN e:Author THEN 'Author'
                         WHEN e:Paper THEN 'Paper'
                         WHEN e:Keyword THEN 'Keyword'
                         WHEN e:Category THEN 'Category'
                         ELSE 'Entity'
                       END AS entity_type,
                       coalesce(e.name, e.title, e.paper_id) AS entity_value,
                       e.paper_id AS entity_paper_id,
                       neighbors
                LIMIT 800
                """,
                {
                    "terms": terms,
                    "source_tag": source_tag,
                    "max_neighbors": max(5, min(80, int(max_neighbors_per_entity))),
                },
            )
            rows = [record.data() for record in result]

            if not rows:
                return []

            paper_entities: Dict[str, set] = {}
            paper_terms: Dict[str, set] = {}
            paper_neighbors: Dict[str, List[dict]] = {}

            for row in rows:
                query_term = str(row.get("query_term", "")).strip().lower()
                entity_type = str(row.get("entity_type", "")).strip()
                entity_value = str(row.get("entity_value", "")).strip()
                entity_label = f"{entity_type}:{entity_value}" if entity_type and entity_value else entity_value

                raw_neighbors = row.get("neighbors", []) or []
                cleaned_neighbors = []
                for n in raw_neighbors:
                    if not isinstance(n, dict):
                        continue
                    value = str(n.get("value", "")).strip()
                    relation = str(n.get("relation", "")).strip()
                    node_type = str(n.get("node_type", "")).strip()
                    paper_id = str(n.get("paper_id", "")).strip()
                    if not value:
                        continue
                    cleaned_neighbors.append(
                        {
                            "relation": relation,
                            "node_type": node_type,
                            "value": value,
                            "paper_id": paper_id,
                        }
                    )

                linked_papers = set()
                own_pid = str(row.get("entity_paper_id", "")).strip()
                if own_pid:
                    linked_papers.add(own_pid)

                for n in cleaned_neighbors:
                    if n.get("node_type") == "Paper" and n.get("paper_id"):
                        linked_papers.add(str(n["paper_id"]).strip())

                if not linked_papers:
                    continue

                for pid in linked_papers:
                    paper_entities.setdefault(pid, set())
                    if entity_label:
                        paper_entities[pid].add(entity_label)

                    paper_terms.setdefault(pid, set())
                    if query_term:
                        paper_terms[pid].add(query_term)

                    paper_neighbors.setdefault(pid, [])
                    for n in cleaned_neighbors:
                        if n.get("paper_id") == pid:
                            continue
                        paper_neighbors[pid].append(
                            {
                                "entity": entity_value,
                                "entity_type": entity_type,
                                "relation": n.get("relation", ""),
                                "neighbor_type": n.get("node_type", ""),
                                "neighbor": n.get("value", ""),
                            }
                        )

            if not paper_entities:
                return []

            for pid, neighbors in list(paper_neighbors.items()):
                uniq = []
                seen_neighbor = set()
                for n in neighbors:
                    key = (
                        str(n.get("entity", "")),
                        str(n.get("relation", "")),
                        str(n.get("neighbor_type", "")),
                        str(n.get("neighbor", "")),
                    )
                    if key in seen_neighbor:
                        continue
                    seen_neighbor.add(key)
                    uniq.append(n)
                paper_neighbors[pid] = uniq

            max_term_cover = max(len(v) for v in paper_terms.values()) or 1
            max_entity_cover = max(len(v) for v in paper_entities.values()) or 1
            max_neighbor_cnt = max(len(v) for v in paper_neighbors.values()) if paper_neighbors else 1
            if max_neighbor_cnt <= 0:
                max_neighbor_cnt = 1

            scored = []
            for pid in paper_entities.keys():
                term_cover = len(paper_terms.get(pid, set()))
                entity_cover = len(paper_entities.get(pid, set()))
                neighbor_cnt = len(paper_neighbors.get(pid, []))

                term_score = min(1.0, float(term_cover) / float(max_term_cover))
                entity_score = min(1.0, float(entity_cover) / float(max_entity_cover))
                neighbor_score = min(1.0, float(neighbor_cnt) / float(max_neighbor_cnt))
                final_score = 0.55 * term_score + 0.30 * entity_score + 0.15 * neighbor_score

                scored.append((pid, final_score, term_score, entity_score, neighbor_score))

            scored.sort(key=lambda x: x[1], reverse=True)
            ranked_ids = [pid for pid, *_ in scored[: max(top_k * 3, 24)]]
            details = self._load_paper_details(session, ranked_ids, source_tag)
            if not details:
                return []

            scored_map = {pid: (s, ts, es, ns) for pid, s, ts, es, ns in scored}
            for d in details:
                pid = d.get("paper_id", "")
                s, ts, es, ns = scored_map.get(pid, (0.0, 0.0, 0.0, 0.0))
                d["score"] = float(s)
                d["sources"] = ["subgraph_all_entities"]
                d["subgraph_entities"] = sorted(paper_entities.get(pid, set()))
                d["subgraph_neighbors"] = (paper_neighbors.get(pid, []) or [])[:40]
                d["subgraph_coverage"] = {
                    "term_score": float(ts),
                    "entity_score": float(es),
                    "neighbor_score": float(ns),
                    "matched_term_count": len(paper_terms.get(pid, set())),
                    "matched_entity_count": len(paper_entities.get(pid, set())),
                }

            details.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
            return details[:top_k]

    def _merge_scores(
        self,
        score_map: Dict[str, float],
        source_map: Dict[str, set],
        rows: List[dict],
    ):
        for row in rows:
            pid = row.get("paper_id")
            if not pid:
                continue
            score = float(row.get("score", 0.0))
            source = str(row.get("source", "unknown"))
            prev = float(score_map.get(pid, 0.0))
            score_map[pid] = max(prev, score)
            source_map.setdefault(pid, set()).add(source)

    def _match_keyword_terms(self, session, terms: List[str], source_tag: str | None, score: float) -> List[dict]:
        if not terms:
            return []
        result = session.run(
            """
            UNWIND $terms AS term
            MATCH (p:Paper)-[:HAS_KEYWORD]->(k:Keyword)
            WHERE ($source_tag IS NULL OR p.source_tag = $source_tag)
              AND (
                toLower(k.name) CONTAINS toLower(term) 
                OR 
                toLower(term) CONTAINS toLower(k.name)
              )
            RETURN DISTINCT p.paper_id AS paper_id, $score AS score, 'keyword_match' AS source
            LIMIT 300
            """,
            {"terms": terms, "source_tag": source_tag, "score": score},
        )
        return [record.data() for record in result]

    def _match_category_terms(self, session, terms: List[str], source_tag: str | None, score: float) -> List[dict]:
        if not terms:
            return []
        result = session.run(
            """
            UNWIND $terms AS term
            MATCH (p:Paper)-[:IN_CATEGORY]->(c:Category)
            WHERE ($source_tag IS NULL OR p.source_tag = $source_tag)
              AND toLower(c.name) CONTAINS term
            RETURN DISTINCT p.paper_id AS paper_id, $score AS score, 'category_match' AS source
            LIMIT 300
            """,
            {"terms": terms, "source_tag": source_tag, "score": score},
        )
        return [record.data() for record in result]

    def _match_author_terms(self, session, terms: List[str], source_tag: str | None, score: float) -> List[dict]:
        if not terms:
            return []
        result = session.run(
            """
            UNWIND $terms AS term
            MATCH (a:Author)-[:AUTHORED]->(p:Paper)
            WHERE ($source_tag IS NULL OR p.source_tag = $source_tag)
              AND toLower(a.name) CONTAINS term
            RETURN DISTINCT p.paper_id AS paper_id, $score AS score, 'author_match' AS source
            LIMIT 300
            """,
            {"terms": terms, "source_tag": source_tag, "score": score},
        )
        return [record.data() for record in result]

    def _match_text_terms(self, session, terms: List[str], source_tag: str | None, score: float) -> List[dict]:
        if not terms:
            return []
        result = session.run(
            """
            UNWIND $terms AS term
            MATCH (p:Paper)
            WHERE ($source_tag IS NULL OR p.source_tag = $source_tag)
              AND (
                toLower(p.title) CONTAINS term OR
                toLower(p.summary) CONTAINS term OR
                toLower(coalesce(p.keywords_text, '')) CONTAINS term
              )
            RETURN DISTINCT p.paper_id AS paper_id, $score AS score, 'paper_text_match' AS source
            LIMIT 400
            """,
            {"terms": terms, "source_tag": source_tag, "score": score},
        )
        return [record.data() for record in result]

    def _load_paper_details(self, session, paper_ids: List[str], source_tag: str | None) -> List[dict]:
        if not paper_ids:
            return []
        result = session.run(
            """
            UNWIND $paper_ids AS pid
            MATCH (p:Paper {paper_id: pid})
            WHERE ($source_tag IS NULL OR p.source_tag = $source_tag)
            OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
            WITH p, collect(DISTINCT a.name)[0..8] AS authors
            OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)
            WITH p, authors, collect(DISTINCT k.name)[0..12] AS keywords
            OPTIONAL MATCH (p)-[:IN_CATEGORY]->(c:Category)
            RETURN p.paper_id AS paper_id,
                   p.title AS title,
                   p.summary AS summary,
                   p.url AS url,
                   p.published AS published,
                   authors,
                   keywords,
                   collect(DISTINCT c.name)[0..8] AS categories
            """,
            {"paper_ids": paper_ids, "source_tag": source_tag},
        )
        return [record.data() for record in result]

    def get_papers_by_ids(self, paper_ids: List[str], source_tag: str | None = None) -> List[dict]:
        if not paper_ids:
            return []
        with self.driver.session(database=self.database) as session:
            return self._load_paper_details(session, paper_ids, source_tag)

    def get_stats(self, source_tag: str) -> dict:
        with self.driver.session(database=self.database) as session:
            counts = session.run(
                """
                CALL {
                  MATCH (p:Paper {source_tag:$source_tag}) RETURN count(p) AS papers
                }
                CALL {
                  MATCH (a:Author {source_tag:$source_tag}) RETURN count(a) AS authors
                }
                CALL {
                  MATCH (c:Category {source_tag:$source_tag}) RETURN count(c) AS categories
                }
                CALL {
                  MATCH (k:Keyword {source_tag:$source_tag}) RETURN count(k) AS keywords
                }
                CALL {
                  MATCH (:Paper {source_tag:$source_tag})-[r:RELATED_TO]-(:Paper {source_tag:$source_tag}) RETURN count(r) AS related_edges
                }
                RETURN papers, authors, categories, keywords, related_edges
                """,
                {"source_tag": source_tag},
            ).single()
            return counts.data() if counts else {}
