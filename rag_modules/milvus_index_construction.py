"""
Milvus 索引构建模块（ARXIV版本）
"""

import logging
from typing import Any, Dict, List, Optional

from app.data import ArxivPaper
from app.vector import MilvusPaperStore


logger = logging.getLogger(__name__)


class MilvusIndexConstructionModule:
    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        dimension: int,
        model_name: str,
    ):
        self.collection_name = collection_name
        self.store = MilvusPaperStore(
            host=host,
            port=port,
            collection_name=collection_name,
            embedding_model=model_name,
            embedding_dim=dimension,
        )

    def build_vector_index(self, papers: List[ArxivPaper], source_tag: str, reset: bool = False) -> bool:
        return self.store.build_index(papers, source_tag=source_tag, reset=reset)

    def similarity_search(self, query: str, k: int = 8, source_tag: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.store.search(query, top_k=k, source_tag=source_tag)

    def get_collection_stats(self) -> Dict[str, Any]:
        return self.store.get_stats()

    def delete_collection(self) -> bool:
        try:
            self.store.delete_collection()
            return True
        except Exception as exc:
            logger.error("Delete collection failed: %s", exc)
            return False

    def has_collection(self) -> bool:
        return self.store.client.has_collection(self.collection_name)

    def load_collection(self) -> bool:
        try:
            if not self.store.client.has_collection(self.collection_name):
                return False
            self.store.client.load_collection(self.collection_name)
            self.store.collection_ready = True
            return True
        except Exception as exc:
            logger.error("Load collection failed: %s", exc)
            return False

    def close(self):
        self.store.close()

