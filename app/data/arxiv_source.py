import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import requests


logger = logging.getLogger(__name__)


@dataclass
class ArxivPaper:
    paper_id: str
    title: str
    summary: str
    published: str
    updated: str
    url: str
    authors: List[str]
    categories: List[str]
    primary_category: str
    keywords: List[str]


_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "our", "using",
    "paper", "model", "models", "method", "methods", "approach", "results", "based",
    "into", "over", "towards", "under", "their", "than", "via", "new", "study",
    "analysis", "data", "learning", "neural", "network", "networks", "large",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _extract_keywords(title: str, summary: str, top_k: int = 10) -> List[str]:
    text = f"{title} {summary}".lower()
    tokens = re.findall(r"[a-z][a-z\-]{2,}", text)

    freq: Dict[str, int] = {}
    for tok in tokens:
        if tok in _STOPWORDS or tok.startswith("http"):
            continue
        freq[tok] = freq.get(tok, 0) + 1

    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in ranked[:top_k]]


class ArxivSourceClient:
    def __init__(self, endpoint: str, delay_sec: float = 3.2):
        self.endpoint = endpoint
        self.delay_sec = delay_sec

    def fetch_papers(self, search_query: str, max_results: int = 200, page_size: int = 100) -> List[ArxivPaper]:
        papers: List[ArxivPaper] = []
        start = 0

        while start < max_results:
            batch_size = min(page_size, max_results - start)
            params = {
                "search_query": search_query,
                "start": start,
                "max_results": batch_size,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            logger.info("Fetching arXiv entries: start=%s size=%s", start, batch_size)

            response = requests.get(self.endpoint, params=params, timeout=40)
            response.raise_for_status()

            batch = self._parse_atom(response.text)
            if not batch:
                break

            papers.extend(batch)
            start += len(batch)

            # arXiv 官方建议控制请求频率
            if start < max_results:
                time.sleep(self.delay_sec)

        logger.info("Fetched %s papers from arXiv", len(papers))
        return papers

    def save_cache(self, papers: List[ArxivPaper], cache_path: str):
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(p) for p in papers]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_cache(self, cache_path: str) -> List[ArxivPaper]:
        path = Path(cache_path)
        if not path.exists():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [ArxivPaper(**item) for item in raw]

    def _parse_atom(self, xml_text: str) -> List[ArxivPaper]:
        root = ET.fromstring(xml_text)
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        papers: List[ArxivPaper] = []

        for entry in root.findall("atom:entry", ns):
            paper_url = _normalize_text(entry.findtext("atom:id", default="", namespaces=ns))
            paper_id = paper_url.rsplit("/", 1)[-1]
            title = _normalize_text(entry.findtext("atom:title", default="", namespaces=ns))
            summary = _normalize_text(entry.findtext("atom:summary", default="", namespaces=ns))
            published = _normalize_text(entry.findtext("atom:published", default="", namespaces=ns))
            updated = _normalize_text(entry.findtext("atom:updated", default="", namespaces=ns))

            authors = []
            for author in entry.findall("atom:author", ns):
                name = _normalize_text(author.findtext("atom:name", default="", namespaces=ns))
                if name:
                    authors.append(name)

            categories = [
                c.attrib.get("term", "").strip()
                for c in entry.findall("atom:category", ns)
                if c.attrib.get("term", "").strip()
            ]
            primary = categories[0] if categories else "unknown"

            keywords = _extract_keywords(title, summary)

            if not paper_id or not title:
                continue

            papers.append(
                ArxivPaper(
                    paper_id=paper_id,
                    title=title,
                    summary=summary,
                    published=published,
                    updated=updated,
                    url=paper_url,
                    authors=list(dict.fromkeys(authors)),
                    categories=list(dict.fromkeys(categories)),
                    primary_category=primary,
                    keywords=keywords,
                )
            )

        return papers
