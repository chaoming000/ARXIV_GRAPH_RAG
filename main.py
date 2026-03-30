"""
ARXIV GraphRAG 主程序（按 C9 架构重构）
"""

import logging
import sys
import time
from typing import List, Optional

from dotenv import load_dotenv

from config import DEFAULT_CONFIG, GraphRAGConfig
from rag_modules import (
    GenerationIntegrationModule,
    GraphDataPreparationModule,
    MilvusIndexConstructionModule,
)
from rag_modules.graph_rag_retrieval import GraphRAGRetrieval
from rag_modules.hybrid_retrieval import HybridRetrievalModule
from rag_modules.intelligent_query_router import IntelligentQueryRouter


_log_level_name = (DEFAULT_CONFIG.app_log_level or "WARNING").upper()
_log_level = getattr(logging, _log_level_name, logging.WARNING)
logging.basicConfig(level=_log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
for _name in ["httpx", "httpcore", "openai", "openai._base_client", "urllib3"]:
    _l = logging.getLogger(_name)
    _l.setLevel(logging.ERROR)
    _l.propagate = False

load_dotenv()
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


class AdvancedGraphRAGSystem:
    def __init__(self, config: Optional[GraphRAGConfig] = None):
        self.config = config or DEFAULT_CONFIG

        self.data_module = None
        self.index_module = None
        self.generation_module = None
        self.traditional_retrieval = None
        self.graph_rag_retrieval = None
        self.query_router = None
        self.system_ready = False

    def initialize_system(self):
        logger.info("启动 ARXIV GraphRAG 系统...")
        self.data_module = GraphDataPreparationModule(
            uri=self.config.neo4j_uri,
            user=self.config.neo4j_user,
            password=self.config.neo4j_password,
            database=self.config.neo4j_database,
            arxiv_endpoint=self.config.arxiv_endpoint,
            request_delay_sec=self.config.arxiv_request_delay_sec,
        )
        self.index_module = MilvusIndexConstructionModule(
            host=self.config.milvus_host,
            port=self.config.milvus_port,
            collection_name=self.config.milvus_collection_name,
            dimension=self.config.milvus_dimension,
            model_name=self.config.embedding_model,
        )
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=self.config.deepseek_api_key,
            base_url=self.config.deepseek_base_url,
            max_reference_items=self.config.max_reference_items,
        )
        self.traditional_retrieval = HybridRetrievalModule(
            config=self.config,
            milvus_module=self.index_module,
            data_module=self.data_module,
            llm_client=self.generation_module.client if self.config.router_llm_enabled else None,
        )
        self.graph_rag_retrieval = GraphRAGRetrieval(
            config=self.config,
            llm_client=self.generation_module.client if self.config.router_llm_enabled else None,
        )
        self.query_router = IntelligentQueryRouter(
            traditional_retrieval=self.traditional_retrieval,
            graph_rag_retrieval=self.graph_rag_retrieval,
            llm_client=self.generation_module.client if self.config.router_llm_enabled else None,
            config=self.config,
        )
        logger.info("系统初始化完成")

    def build_knowledge_base(self):
        print("\n检查知识库状态...")
        stats = self.index_module.get_collection_stats()
        row_count = int(stats.get("row_count", 0)) if isinstance(stats, dict) else 0
        collection_exists = bool(stats.get("exists", False)) if isinstance(stats, dict) else False

        if collection_exists and row_count > 0:
            print(f"发现已存在向量集合（{row_count} 条），尝试直接加载。")
            if self.index_module.load_collection():
                self._prepare_runtime_views(force_refresh=False)
                self.system_ready = True
                print("✅ 知识库加载完成")
                return

        print("未发现可用知识库，开始自动构建...")
        self._prepare_runtime_views(force_refresh=False)

        self.data_module.index_to_neo4j(source_tag=self.config.source_tag, reset=False)
        ok = self.index_module.build_vector_index(
            papers=self.data_module.papers,
            source_tag=self.config.source_tag,
            reset=False,
        )
        if not ok:
            raise RuntimeError("Milvus 向量索引构建失败")

        self._initialize_retrievers(self.data_module.chunks)
        self.system_ready = True
        self._show_knowledge_base_stats()
        print("✅ 知识库构建完成")

    def _prepare_runtime_views(self, force_refresh: bool = False):
        self.data_module.load_graph_data(
            cache_path=self.config.cache_path,
            search_query=self.config.arxiv_search_query,
            max_results=self.config.arxiv_max_results,
            page_size=self.config.arxiv_page_size,
            use_cache=True,
            force_refresh=force_refresh,
        )
        self.data_module.build_paper_documents()
        self.data_module.chunk_documents(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self._initialize_retrievers(self.data_module.chunks)

    def _initialize_retrievers(self, chunks: List):
        self.traditional_retrieval.initialize(chunks)
        self.graph_rag_retrieval.initialize()

    def _show_knowledge_base_stats(self):
        ds = self.data_module.get_statistics()
        vs = self.index_module.get_collection_stats()
        rs = self.query_router.get_route_statistics()
        print("\n知识库统计:")
        print(f"  papers: {ds.get('total_papers', 0)}")
        print(f"  documents: {ds.get('total_documents', 0)}")
        print(f"  chunks: {ds.get('total_chunks', 0)}")
        print(f"  milvus_row_count: {vs.get('row_count', 0)}")
        print(f"  routed_queries: {rs.get('total_queries', 0)}")

    def ask_question_with_routing(self, question: str, stream: bool = True, explain_routing: bool = False):
        if not self.system_ready:
            raise RuntimeError("系统未就绪")

        start = time.time()
        if explain_routing:
            print(self.query_router.explain_routing_decision(question))

        docs, analysis = self.query_router.route_query(question, self.config.top_k)
        if not docs:
            return "未检索到相关论文，请尝试更具体的主题关键词。", analysis

        if stream:
            chunks = []
            for part in self.generation_module.generate_adaptive_answer_stream(question, docs):
                print(part, end="", flush=True)
                chunks.append(part)
            print()
            answer = "".join(chunks)
        else:
            answer = self.generation_module.generate_adaptive_answer(question, docs)

        print(f"\n⏱️ 耗时: {time.time() - start:.2f}s")
        return answer, analysis

    def run_interactive(self):
        if not self.system_ready:
            print("系统未就绪")
            return

        print("\nARXIV GraphRAG 已启动")
        print("命令: stats | rebuild | quit")
        while True:
            try:
                user_input = input("\n问题> ").strip()
            except KeyboardInterrupt:
                break
            if not user_input:
                continue
            cmd = user_input.lower()
            if cmd in {"quit", "exit"}:
                break
            if cmd == "stats":
                self._show_system_stats()
                continue
            if cmd == "rebuild":
                self._rebuild_knowledge_base()
                continue

            use_stream = self.config.stream_answer
            print("\n回答> ")
            answer, analysis = self.ask_question_with_routing(
                user_input,
                stream=use_stream,
                explain_routing=False,
            )
            if not use_stream:
                print("\n" + answer)
            elif not answer:
                print("\n(空回答)")
            if analysis and self.config.show_route:
                print(
                    "\n[route] "
                    f"{analysis.recommended_strategy.value} | "
                    f"confidence={analysis.confidence:.2f} | "
                    f"reason={analysis.reasoning}"
                )

        self._cleanup()
        print("\n已退出")

    def _show_system_stats(self):
        self._show_knowledge_base_stats()
        rs = self.query_router.get_route_statistics()
        print(
            f"路由统计: traditional={rs.get('traditional_count', 0)}, "
            f"graph={rs.get('graph_rag_count', 0)}"
        )

    def _rebuild_knowledge_base(self):
        confirm = input("将重建索引，是否继续？(y/N): ").strip().lower()
        if confirm != "y":
            print("已取消")
            return
        self.index_module.delete_collection()
        self.build_knowledge_base()

    def _cleanup(self):
        if self.traditional_retrieval:
            self.traditional_retrieval.close()
        if self.graph_rag_retrieval:
            self.graph_rag_retrieval.close()
        if self.index_module:
            self.index_module.close()
        if self.data_module:
            self.data_module.close()


def main():
    try:
        print("启动 ARXIV GraphRAG...")
        system = AdvancedGraphRAGSystem()
        system.initialize_system()
        system.build_knowledge_base()
        system.run_interactive()
    except Exception as exc:
        logger.error("系统运行失败: %s", exc)
        import traceback

        traceback.print_exc()
        print(f"\n系统错误: {exc}")


if __name__ == "__main__":
    main()
