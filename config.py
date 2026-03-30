"""
ARXIV GraphRAG 配置（C9 架构风格）
"""

import os
from dataclasses import dataclass
from typing import Any, Dict


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class GraphRAGConfig:
    # Neo4j
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://172.28.123.0:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "all-in-rag")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # Milvus
    milvus_host: str = os.getenv("MILVUS_HOST", "172.28.123.0")
    milvus_port: int = int(os.getenv("MILVUS_PORT", "19530"))
    milvus_collection_name: str = os.getenv("MILVUS_COLLECTION", "arxiv_paper_chunks")
    milvus_dimension: int = int(os.getenv("EMBEDDING_DIM", "384"))

    # Models
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    llm_model: str = os.getenv("LLM_MODEL", "deepseek-chat")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    # Retrieval
    top_k: int = int(os.getenv("FINAL_TOP_K", "8"))

    # Generation
    temperature: float = float(os.getenv("GEN_TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("GEN_MAX_TOKENS", "1200"))

    # Data source
    arxiv_endpoint: str = os.getenv("ARXIV_ENDPOINT", "http://export.arxiv.org/api/query")
    arxiv_search_query: str = os.getenv(
        "ARXIV_SEARCH_QUERY",
        "(cat:cs.AI OR cat:cs.CL OR cat:cs.LG OR cat:cs.CV)",
    )
    arxiv_max_results: int = int(os.getenv("ARXIV_MAX_RESULTS", "250"))
    arxiv_page_size: int = int(os.getenv("ARXIV_PAGE_SIZE", "100"))
    arxiv_request_delay_sec: float = float(os.getenv("ARXIV_REQUEST_DELAY_SEC", "3.2"))

    # Runtime
    source_tag: str = os.getenv("SOURCE_TAG", "arxiv_recent_ai")
    cache_path: str = os.getenv("CACHE_PATH", "data/arxiv_papers.json")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    app_log_level: str = os.getenv("APP_LOG_LEVEL", "WARNING")
    show_route: bool = _env_bool("SHOW_ROUTE", False)
    stream_answer: bool = _env_bool("STREAM_ANSWER", True)
    max_reference_items: int = int(os.getenv("MAX_REFERENCE_ITEMS", "3"))

    # Router
    router_llm_enabled: bool = _env_bool("ROUTER_LLM_ENABLED", True)
    router_llm_model: str = os.getenv("ROUTER_LLM_MODEL", os.getenv("LLM_MODEL", "deepseek-chat"))

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GraphRAGConfig":
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


DEFAULT_CONFIG = GraphRAGConfig()
