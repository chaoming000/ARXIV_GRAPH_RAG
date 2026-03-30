import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
from sentence_transformers import SentenceTransformer

from app.data import ArxivPaper


logger = logging.getLogger(__name__)


class MilvusPaperStore:
    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        embedding_model: str,
        embedding_dim: int,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim

        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}")
        self.embedder = SentenceTransformer(self.embedding_model, device="cpu")
        self.collection_ready = False

    def _schema(self) -> CollectionSchema:
        return CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=16000),
                FieldSchema(name="paper_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="published", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="authors", dtype=DataType.VARCHAR, max_length=4000),
                FieldSchema(name="categories", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="source_tag", dtype=DataType.VARCHAR, max_length=128),
            ],
            description="arXiv paper chunk vectors",
        )

    def create_collection(self, reset: bool = False) -> bool:
        try:
            if self.client.has_collection(self.collection_name):
                if reset:
                    self.client.drop_collection(self.collection_name)
                else:
                    self.collection_ready = True
                    return True

            self.client.create_collection(
                collection_name=self.collection_name,
                schema=self._schema(),
                metric_type="COSINE",
                consistency_level="Strong",
            )

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 200},
            )
            self.client.create_index(collection_name=self.collection_name, index_params=index_params)
            self.client.load_collection(self.collection_name)
            self.collection_ready = True
            return True
        except Exception as e:
            logger.error("Failed to create collection: %s", e)
            return False

    def _paper_to_chunk(self, paper: ArxivPaper, source_tag: str) -> Dict[str, Any]:
        text = f"Title: {paper.title}\n\nAbstract: {paper.summary}"
        return {
            "id": f"{paper.paper_id}_0",
            "text": text[:15000],
            "paper_id": paper.paper_id,
            "title": paper.title[:1900],
            "url": paper.url[:900],
            "published": paper.published[:60],
            "authors": ", ".join(paper.authors)[:3900],
            "categories": ", ".join(paper.categories)[:900],
            "keywords": ", ".join(paper.keywords)[:1900],
            "source_tag": source_tag,
        }

    def build_index(self, papers: List[ArxivPaper], source_tag: str, reset: bool = True, batch_size: int = 64) -> bool:
        if not papers:
            return False

        if not self.create_collection(reset=reset):
            return False

        chunks = [self._paper_to_chunk(p, source_tag) for p in papers]
        texts = [c["text"] for c in chunks]
        vectors = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

        for c, v in zip(chunks, vectors):
            c["vector"] = v

        try:
            for i in range(0, len(chunks), batch_size):
                self.client.insert(collection_name=self.collection_name, data=chunks[i : i + batch_size])
            self.client.load_collection(self.collection_name)
            return True
        except Exception as e:
            logger.error("Failed to insert vectors: %s", e)
            return False

    def search(self, query: str, top_k: int = 8, source_tag: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.collection_ready and not self.client.has_collection(self.collection_name):
            return []

        query_vec = self.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False).tolist()[0]
        search_kwargs = {
            "collection_name": self.collection_name,
            "data": [query_vec],
            "anns_field": "vector",
            "limit": top_k,
            "output_fields": [
                "text",
                "paper_id",
                "title",
                "url",
                "published",
                "authors",
                "categories",
                "keywords",
                "source_tag",
            ],
            "search_params": {"metric_type": "COSINE", "params": {"ef": 64}},
        }
        if source_tag:
            search_kwargs["filter"] = f"source_tag == '{source_tag}'"

        results = self.client.search(**search_kwargs)
        output: List[Dict[str, Any]] = []
        if results and len(results) > 0:
            for hit in results[0]:
                entity = hit.get("entity", {})
                output.append(
                    {
                        "id": hit.get("id"),
                        "score": hit.get("distance", 0.0),
                        "paper_id": entity.get("paper_id", ""),
                        "title": entity.get("title", ""),
                        "text": entity.get("text", ""),
                        "url": entity.get("url", ""),
                        "published": entity.get("published", ""),
                        "authors": entity.get("authors", ""),
                        "categories": entity.get("categories", ""),
                        "keywords": entity.get("keywords", ""),
                    }
                )
        return output

    def get_stats(self) -> Dict[str, Any]:
        if not self.client.has_collection(self.collection_name):
            return {"collection": self.collection_name, "exists": False}
        stats = self.client.get_collection_stats(self.collection_name)
        return {"collection": self.collection_name, "exists": True, "row_count": stats.get("row_count", 0)}

    def delete_collection(self):
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            self.collection_ready = False

    def close(self):
        # MilvusClient does not need explicit close.
        pass
