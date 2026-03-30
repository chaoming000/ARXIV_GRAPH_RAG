# ARXIV_GRAPH_RAG

基于 C9（图RAG 智能烹饪助手）流程改造的 ARXIV 论文检索问答项目。

## 架构流程

1. 数据准备 `rag_modules/graph_data_preparation.py`
- 从 arXiv 拉取论文
- 生成论文文档与分块
- 写入 Neo4j 图谱

2. 向量索引 `rag_modules/milvus_index_construction.py`
- 论文分块向量化
- 写入/加载 Milvus 集合

3. 检索引擎
- `rag_modules/hybrid_retrieval.py`：向量检索（Milvus chunks）
- `rag_modules/graph_rag_retrieval.py`：图检索（按策略分为 `MULTI_HOP` / `SUBGRAPH`）

4. 智能路由 `rag_modules/intelligent_query_router.py`
- LLM 提取实体与关系（强制 JSON）
- 路由决策：若命中图关系或多类型实体则走 `graph_rag`，否则走 `hybrid_traditional`（向量）
- LLM 失败时关键词降级（如“关系/涉及/关联”等）

5. 回答生成 `rag_modules/generation_integration.py`
- 默认流式生成
- 流式失败重试，超过次数后降级为一次生成

6. 主控入口 `main.py`
- 启动即自动初始化模块
- 自动加载或构建知识库
- 默认进入聊天模式

## 图模型（Neo4j）

节点：
- `Paper`
- `Author`
- `Category`
- `Keyword`
- `Source`

关系：
- `(:Author)-[:AUTHORED]->(:Paper)`
- `(:Paper)-[:IN_CATEGORY]->(:Category)`
- `(:Paper)-[:HAS_KEYWORD]->(:Keyword)`
- `(:Paper)-[:FROM_SOURCE]->(:Source)`
- `(:Paper)-[:RELATED_TO]->(:Paper)`

## 快速开始

```bash
cd ARXIV_GRAPH_RAG
python -m pip install -r requirements.txt
copy .env.example .env
python main.py
```

启动后可用命令：
- `stats` 查看统计
- `rebuild` 重建索引
- `quit` 退出

常用输出控制（`.env`）：
- `APP_LOG_LEVEL=WARNING`：压低运行日志噪声
- `SHOW_ROUTE=false`：默认不打印路由行
- `STREAM_ANSWER=true`：是否默认流式输出
- `MAX_REFERENCE_ITEMS=3`：引用列表最大条数

## 说明

- 这是 ARXIV 论文场景版本，非论文场景相关逻辑已清理。
- 路由与检索流程对齐 C9：查询分析 -> 路由 -> 检索 -> 生成。
