"""
生成集成模块（ARXIV版本）
"""

import logging
import re
from typing import List

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

from .types import Document


logger = logging.getLogger(__name__)


class GenerationIntegrationModule:
    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_tokens: int,
        api_key: str,
        base_url: str,
        max_reference_items: int = 3,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_reference_items = max(1, int(max_reference_items))
        self.client = OpenAI(api_key=api_key, base_url=base_url) if (api_key and OpenAI is not None) else None

    def generate_adaptive_answer(self, question: str, documents: List[Document]) -> str:
        if not documents:
            return "未检索到相关论文，请尝试更具体的主题关键词。"

        normalized_docs = self._normalize_documents(documents)

        if not self.client:
            lines = [f"问题：{question}", f"命中文档：{len(documents)}", ""]
            for i, d in enumerate(normalized_docs[: self.max_reference_items], start=1):
                lines.append(
                    f"{i}. {d.metadata.get('title', 'unknown')} | "
                    f"{d.metadata.get('paper_id', '')} | {d.metadata.get('url', '')}"
                )
            lines.append("\n未配置 DEEPSEEK_API_KEY，返回结构化检索结果。")
            return "\n".join(lines)

        prompt = self._build_generation_prompt(question, normalized_docs)
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            llm_answer = (resp.choices[0].message.content or "").strip()
            return self._append_reference_list(llm_answer, normalized_docs)
        except Exception as exc:
            logger.warning("LLM generation failed: %s", exc)
            fallback_lines = ["生成回答失败，返回检索到的论文：", ""]
            for i, d in enumerate(normalized_docs[:8], start=1):
                fallback_lines.append(
                    f"{i}. {d.metadata.get('title', 'unknown')} | "
                    f"{d.metadata.get('paper_id', '')} | {d.metadata.get('url', '')}"
                )
            return "\n".join(fallback_lines)

    def generate_adaptive_answer_stream(self, question: str, documents: List[Document], max_retries: int = 2):
        if not documents:
            yield "未检索到相关论文，请尝试更具体的主题关键词。"
            return

        normalized_docs = self._normalize_documents(documents)

        if not self.client:
            yield self.generate_adaptive_answer(question, normalized_docs)
            return

        prompt = self._build_generation_prompt(question, normalized_docs)
        last_error = None
        for _attempt in range(max(1, max_retries)):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                )
                chunks: List[str] = []
                for chunk in response:
                    piece = chunk.choices[0].delta.content or ""
                    if piece:
                        chunks.append(piece)
                        yield piece

                streamed_text = "".join(chunks)
                ref_block = self._format_reference_list(normalized_docs)
                if ref_block and not self._has_reference_section(streamed_text):
                    yield "\n\n" + ref_block
                return
            except Exception as exc:
                last_error = exc
                logger.warning("LLM stream generation failed, retrying: %s", exc)
                continue

        logger.warning("LLM stream failed after retries, fallback to non-stream: %s", last_error)
        yield self.generate_adaptive_answer(question, normalized_docs)

    def _normalize_documents(self, documents: List[Document]) -> List[Document]:
        normalized: List[Document] = []
        for doc in documents:
            meta = dict(doc.metadata or {})
            content = doc.page_content or ""

            if not meta.get("title"):
                extracted = self._extract_field_from_content(content, "title")
                if extracted:
                    meta["title"] = extracted
            if not meta.get("paper_id"):
                extracted = self._extract_field_from_content(content, "paper_id")
                if extracted:
                    meta["paper_id"] = extracted
            if not meta.get("url"):
                extracted = self._extract_field_from_content(content, "url")
                if extracted:
                    meta["url"] = extracted

            normalized.append(Document(page_content=content, metadata=meta))
        return normalized

    def _extract_field_from_content(self, content: str, field: str) -> str:
        prefix = field.lower() + ":"
        for line in content.splitlines():
            raw = line.strip()
            if raw.lower().startswith(prefix):
                return raw[len(prefix) :].strip()
        return ""

    def _append_reference_list(self, answer: str, documents: List[Document]) -> str:
        ref_block = self._format_reference_list(documents)
        if not ref_block:
            return answer
        if self._has_reference_section(answer):
            return answer

        lines = [answer.rstrip(), "", ref_block]
        return "\n".join(lines)

    def _has_reference_section(self, text: str) -> bool:
        if not text:
            return False
        # 支持多种常见写法，避免重复追加引用区：
        # [引用论文] / **引用论文** / ### 引用论文 / 引用论文：
        return bool(
            re.search(
                r"(?mi)^\s*(?:[-*]\s*)?(?:\*{1,2}|#{1,6}\s*|\[)?\s*引用论文\s*(?:\*{1,2}|\])?\s*[:：]?\s*$",
                text,
            )
        )

    def _format_reference_list(self, documents: List[Document]) -> str:
        refs = []
        seen = set()
        for doc in documents:
            meta = doc.metadata or {}
            title = str(meta.get("title", "")).strip()
            paper_id = str(meta.get("paper_id", "")).strip()
            url = str(meta.get("url", "")).strip()

            key = paper_id or title
            if not key or key in seen:
                continue
            seen.add(key)
            refs.append((title or "unknown", paper_id, url))
            if len(refs) >= self.max_reference_items:
                break

        if not refs:
            return ""

        lines = ["[引用论文]"]
        for i, (title, paper_id, url) in enumerate(refs, start=1):
            lines.append(f"{i}. {title} | {paper_id} | {url}")
        return "\n".join(lines)

    def _build_generation_prompt(self, question: str, documents: List[Document]) -> str:
        context = []
        for i, d in enumerate(documents[:12], start=1):
            meta = d.metadata or {}
            context.append(
                f"[{i}] title={meta.get('title', '')} | paper_id={meta.get('paper_id', '')} | "
                f"authors={meta.get('authors', '')} | published={meta.get('published', '')} | "
                f"url={meta.get('url', '')}\n{d.page_content[:1200]}"
            )
        return (
            "你是论文助手。基于候选文献回答问题，不要编造。\n"
            "必须满足：\n"
            "1) 对关键结论使用编号引用，如[1][2]，编号需要与“引用论文”中的一致\n"
            "2) 回答末尾必须有“引用论文”小节\n"
            "3) “引用论文”中每条都必须包含：标题、paper_id、arXiv链接\n"
            "4) 若候选文献不足以支持结论，要明确写“证据不足”\n\n"
            "5) 没有用到的候选文献不要提及，也不要写到“引用论文”中\n\n"
            "6) 不要提及我给你提供了候选论文这一事实\n\n"
            f"问题：{question}\n\n候选文献：\n" + "\n\n".join(context)
        )
