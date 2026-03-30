"""
向量检索模块（ARXIV版本）
"""

from typing import List

from .types import Document


class HybridRetrievalModule:
    """
    当前流程下仅承担向量检索职责（Milvus chunks）。
    """

    def __init__(self, config, milvus_module, data_module, llm_client=None):
        self.config = config
        self.milvus_module = milvus_module
        self.data_module = data_module
        self.llm_client = llm_client

    def initialize(self, chunks: List[Document]):
        _ = chunks

    def vector_search_enhanced(self, query: str, top_k: int = 5) -> List[Document]:
        hits = self.milvus_module.similarity_search(query=query, k=top_k * 2, source_tag=self.config.source_tag)
        docs: List[Document] = []
        for hit in hits[:top_k]:
            docs.append(
                Document(
                    page_content=hit.get("text", ""),
                    metadata={
                        "paper_id": hit.get("paper_id", ""),
                        "title": hit.get("title", ""),
                        "url": hit.get("url", ""),
                        "published": hit.get("published", ""),
                        "authors": hit.get("authors", ""),
                        "categories": hit.get("categories", ""),
                        "keywords": hit.get("keywords", ""),
                        "relevance_score": float(hit.get("score", 0.0)),
                        "search_type": "vector_enhanced",
                    },
                )
            )
        return docs

    def close(self):
        # 当前模块无外部连接资源
        pass
