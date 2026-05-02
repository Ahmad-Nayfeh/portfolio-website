"""scripts/discover_papers.py

Discover candidate papers for an AI-papers stream.

Primary source: Hugging Face Daily Papers (https://huggingface.co/papers).
Backup: arXiv (queried via its public API). Both return a list of normalized
PaperCandidate dicts that downstream code uses without caring which source
they came from.

Importable: no top-level side effects.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from typing import Any
from urllib.parse import quote

import requests

log = logging.getLogger(__name__)

HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
ARXIV_API = "http://export.arxiv.org/api/query"
USER_AGENT = "ahmadnayfeh-portfolio-bot/0.1 (+https://github.com/Ahmad-Nayfeh/portfolio-website)"
TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------------
# Hugging Face Daily Papers
# ---------------------------------------------------------------------------


def fetch_huggingface_daily(lookback_days: int = 7) -> list[dict[str, Any]]:
    """Fetch daily-papers entries for the last `lookback_days` days.

    The HF endpoint returns a list of {paper, ...} objects per day. We flatten
    across days and de-duplicate by arxiv id.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    today = dt.date.today()
    for offset in range(lookback_days):
        day = today - dt.timedelta(days=offset)
        url = f"{HF_DAILY_URL}?date={day.isoformat()}"
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_SECONDS)
            r.raise_for_status()
        except requests.RequestException as e:
            log.warning("HF daily papers fetch failed for %s: %s", day, e)
            continue

        for item in r.json() or []:
            paper = item.get("paper") or {}
            arxiv_id = paper.get("id") or ""
            if not arxiv_id or arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            out.append(_normalize_hf_paper(paper, item))
    log.info("HF Daily Papers: %d unique papers across %d days", len(out), lookback_days)
    return out


def _normalize_hf_paper(paper: dict[str, Any], wrapper: dict[str, Any]) -> dict[str, Any]:
    arxiv_id = paper.get("id") or ""
    return {
        "source": "huggingface_daily_papers",
        "arxiv_id": arxiv_id,
        "title": paper.get("title", "").strip(),
        "summary": paper.get("summary", "").strip(),
        "authors": [a.get("name", "") for a in (paper.get("authors") or [])],
        "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else "",
        "categories": paper.get("ai_categories") or paper.get("categories") or [],
        "upvotes": wrapper.get("numUniqueUpvotes") or wrapper.get("upvotes") or 0,
        "published": paper.get("publishedAt") or wrapper.get("publishedAt") or "",
    }


# ---------------------------------------------------------------------------
# arXiv (backup or `discovery.source: arxiv`)
# ---------------------------------------------------------------------------


def fetch_arxiv(
    categories: list[str],
    lookback_days: int = 7,
    max_results: int = 50,
) -> list[dict[str, Any]]:
    """Query arXiv for recent papers in the given categories."""
    if not categories:
        categories = ["cs.LG"]
    cutoff = (dt.datetime.utcnow() - dt.timedelta(days=lookback_days)).strftime("%Y%m%d%H%M%S")
    cat_query = "+OR+".join(f"cat:{quote(c)}" for c in categories)
    search = f"({cat_query})+AND+submittedDate:[{cutoff}+TO+999912312359]"
    url = f"{ARXIV_API}?search_query={search}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"

    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_SECONDS)
        r.raise_for_status()
    except requests.RequestException as e:
        log.warning("arXiv fetch failed: %s", e)
        return []
    return _parse_arxiv_atom(r.text)


def _parse_arxiv_atom(xml: str) -> list[dict[str, Any]]:
    """Parse arXiv's Atom feed without bringing in a heavy XML dep."""
    entries = re.findall(r"<entry>([\s\S]*?)</entry>", xml)
    out: list[dict[str, Any]] = []
    for entry in entries:
        idm = re.search(r"<id>([^<]+)</id>", entry)
        url = idm.group(1).strip() if idm else ""
        arxiv_id = url.rsplit("/", 1)[-1].replace("abs/", "") if url else ""
        title = _xml_text(entry, "title")
        summary = _xml_text(entry, "summary")
        authors = re.findall(r"<author>\s*<name>([^<]+)</name>", entry)
        cats = [m.group(1) for m in re.finditer(r'<category[^/]*term="([^"]+)"', entry)]
        published = _xml_text(entry, "published")
        out.append(
            {
                "source": "arxiv",
                "arxiv_id": arxiv_id,
                "title": title,
                "summary": summary,
                "authors": authors,
                "url": url,
                "pdf_url": url.replace("/abs/", "/pdf/") + ".pdf" if url else "",
                "categories": cats,
                "upvotes": 0,
                "published": published,
            }
        )
    return out


def _xml_text(entry: str, tag: str) -> str:
    m = re.search(rf"<{tag}>([\s\S]*?)</{tag}>", entry)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Filtering + selection
# ---------------------------------------------------------------------------


def apply_filters(
    candidates: list[dict[str, Any]],
    categories: list[str] | None = None,
    min_upvotes: int = 0,
) -> list[dict[str, Any]]:
    """Filter candidates by categories (if any match) and upvote threshold."""
    out: list[dict[str, Any]] = []
    for c in candidates:
        if min_upvotes and (c.get("upvotes") or 0) < min_upvotes:
            continue
        if categories:
            cats = c.get("categories") or []
            if cats and not any(cat in cats for cat in categories):
                continue
        out.append(c)
    return out


def discover_for_stream(stream_cfg: Any) -> list[dict[str, Any]]:
    """Discover candidates for a stream config object (StreamConfig from
    load_streams.py). Returns the top_n filtered candidates, ranked by
    upvotes desc.
    """
    discovery = stream_cfg.discovery
    source = discovery.source
    lookback = discovery.lookback_days
    cats = (discovery.filters or {}).get("categories") or []
    min_upvotes = (discovery.filters or {}).get("min_upvotes") or 0
    top_n = discovery.top_n or 10

    if source == "huggingface_daily_papers":
        candidates = fetch_huggingface_daily(lookback)
    elif source == "arxiv":
        candidates = fetch_arxiv(cats, lookback, max_results=max(top_n * 5, 50))
    elif source == "manual":
        log.info("Stream discovery=manual; skipping automatic discovery")
        return []
    elif source == "none":
        return []
    else:
        log.warning("Unknown discovery source %r; falling back to arxiv", source)
        candidates = fetch_arxiv(cats, lookback, max_results=max(top_n * 5, 50))

    candidates = apply_filters(candidates, cats, min_upvotes)
    candidates.sort(key=lambda c: c.get("upvotes", 0), reverse=True)
    return candidates[:top_n]
