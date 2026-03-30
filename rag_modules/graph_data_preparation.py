"""
图数据准备模块（ARXIV版本）
"""

import logging
from typing import Dict, List

from app.data import ArxivPaper, ArxivSourceClient
from app.graph import Neo4jGraphStore

from .types import Document


logger = logging.getLogger(__name__)


class GraphDataPreparationModule:
    """从 arXiv 准备图数据，并转换为文档/分块。"""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str,
        arxiv_endpoint: str,
        request_delay_sec: float,
    ):
        self.source_client = ArxivSourceClient(endpoint=arxiv_endpoint, delay_sec=request_delay_sec)
        self.graph_store = Neo4jGraphStore(uri=uri, user=user, password=password, database=database)

        self.papers: List[ArxivPaper] = []
        self.documents: List[Document] = []
        self.chunks: List[Document] = []

    def load_graph_data(
        self,
        cache_path: str,
        search_query: str,
        max_results: int,
        page_size: int,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> Dict[str, int]:
        if use_cache and not force_refresh:
            cached = self.source_client.load_cache(cache_path)
            if cached:
                self.papers = cached
                logger.info("Loaded %s papers from cache", len(cached))
                return {"papers": len(self.papers)}

        self.papers = self.source_client.fetch_papers(
            search_query=search_query,
            max_results=max_results,
            page_size=page_size,
        )
        self.source_client.save_cache(self.papers, cache_path)
        logger.info("Fetched %s papers from source", len(self.papers))
        return {"papers": len(self.papers)}

    def index_to_neo4j(self, source_tag: str, reset: bool = False) -> Dict[str, int]:
        if not self.papers:
            return {"papers": 0}

        self.graph_store.ensure_schema()
        if reset:
            self.graph_store.clear_source_tag(source_tag)
        self.graph_store.upsert_papers(self.papers, source_tag)
        self.graph_store.build_related_edges(source_tag)
        stats = self.graph_store.get_stats(source_tag)
        return {"papers": int(stats.get("papers", 0))}

    def build_paper_documents(self) -> List[Document]:
        docs: List[Document] = []
        for p in self.papers:
            text = (
                f"Title: {p.title}\n"
                f"Paper ID: {p.paper_id}\n"
                f"Published: {p.published}\n"
                f"URL: {p.url}\n"
                f"Authors: {', '.join(p.authors)}\n"
                f"Categories: {', '.join(p.categories)}\n"
                f"Keywords: {', '.join(p.keywords)}\n"
                f"Abstract: {p.summary}"
            )
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "paper_id": p.paper_id,
                        "title": p.title,
                        "url": p.url,
                        "published": p.published,
                        "authors": ", ".join(p.authors),
                        "categories": ", ".join(p.categories),
                        "keywords": ", ".join(p.keywords),
                        "doc_type": "paper",
                    },
                )
            )
        self.documents = docs
        return docs

    def chunk_documents(self, chunk_size: int = 1000, chunk_overlap: int = 120) -> List[Document]:
        if not self.documents:
            return []

        chunks: List[Document] = []
        step = max(1, chunk_size - chunk_overlap)
        for doc in self.documents:
            content = doc.page_content
            if len(content) <= chunk_size:
                chunks.append(
                    Document(
                        page_content=content,
                        metadata={
                            **doc.metadata,
                            "chunk_id": f"{doc.metadata.get('paper_id', 'unknown')}_0",
                            "chunk_index": 0,
                            "total_chunks": 1,
                        },
                    )
                )
                continue

            total = (len(content) - 1) // step + 1
            for i, start in enumerate(range(0, len(content), step)):
                part = content[start : start + chunk_size]
                chunks.append(
                    Document(
                        page_content=part,
                        metadata={
                            **doc.metadata,
                            "chunk_id": f"{doc.metadata.get('paper_id', 'unknown')}_{i}",
                            "chunk_index": i,
                            "total_chunks": total,
                        },
                    )
                )
        self.chunks = chunks
        return chunks

    def get_statistics(self) -> Dict[str, int]:
        return {
            "total_papers": len(self.papers),
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
        }

    def close(self):
        self.graph_store.close()
