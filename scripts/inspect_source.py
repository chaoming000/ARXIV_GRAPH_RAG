"""
Inspect arXiv source data shape for ARXIV_GRAPH_RAG.

Usage examples:
  python scripts/inspect_source.py
  python scripts/inspect_source.py --mode fetch --max-results 5 --page-size 5
  python scripts/inspect_source.py --sample-size 3 --full-summary
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data import ArxivPaper, ArxivSourceClient
from config import DEFAULT_CONFIG


def _to_preview_dict(paper: ArxivPaper, full_summary: bool = False) -> Dict[str, Any]:
    item = asdict(paper)
    if not full_summary and item.get("summary"):
        summary = str(item["summary"])
        item["summary"] = summary[:240] + ("..." if len(summary) > 240 else "")
    return item


def _print_report(
    papers: List[ArxivPaper],
    mode: str,
    sample_size: int,
    cache_path: str,
    query: str,
    full_summary: bool,
):
    samples = [_to_preview_dict(p, full_summary=full_summary) for p in papers[:sample_size]]
    report = {
        "mode": mode,
        "total_records": len(papers),
        "sample_size": len(samples),
        "cache_path": str(Path(cache_path)),
        "query": query,
        "fields": [
            "paper_id",
            "title",
            "summary",
            "published",
            "updated",
            "url",
            "authors",
            "categories",
            "primary_category",
            "keywords",
        ],
        "samples": samples,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    load_dotenv()
    cfg = DEFAULT_CONFIG

    parser = argparse.ArgumentParser(description="Inspect arXiv source data shape.")
    parser.add_argument(
        "--mode",
        choices=["cache", "fetch"],
        default="cache",
        help="cache: read local cache file; fetch: request from arXiv API",
    )
    parser.add_argument("--cache-path", default=cfg.cache_path, help="Path to local cache JSON.")
    parser.add_argument("--query", default=cfg.arxiv_search_query, help="arXiv search_query.")
    parser.add_argument("--max-results", type=int, default=10, help="Max records for fetch mode.")
    parser.add_argument("--page-size", type=int, default=10, help="Page size for fetch mode.")
    parser.add_argument("--sample-size", type=int, default=3, help="How many sample records to print.")
    parser.add_argument(
        "--full-summary",
        action="store_true",
        help="Print full abstract instead of preview.",
    )
    args = parser.parse_args()

    client = ArxivSourceClient(endpoint=cfg.arxiv_endpoint, delay_sec=cfg.arxiv_request_delay_sec)

    if args.mode == "cache":
        papers = client.load_cache(args.cache_path)
        if not papers:
            print(
                json.dumps(
                    {
                        "mode": "cache",
                        "total_records": 0,
                        "message": "No cache data found. Try --mode fetch first.",
                        "cache_path": str(Path(args.cache_path)),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
    else:
        papers = client.fetch_papers(
            search_query=args.query,
            max_results=max(1, args.max_results),
            page_size=max(1, args.page_size),
        )

    _print_report(
        papers=papers,
        mode=args.mode,
        sample_size=max(1, args.sample_size),
        cache_path=args.cache_path,
        query=args.query,
        full_summary=args.full_summary,
    )


if __name__ == "__main__":
    main()
