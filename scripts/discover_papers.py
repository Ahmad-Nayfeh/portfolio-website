"""scripts/discover_papers.py

Discover candidate papers for an AI-papers stream.

Primary source: Hugging Face Daily Papers (https://huggingface.co/papers).
Backup: arXiv (queried via its public API). Both return a list of normalized
PaperCandidate dicts that downstream code uses without caring which source
they came from.

Deduplication state: when a stream config sets
`discovery.dedup_state_file`, we read that JSON file at discovery time and
drop any candidate whose `arxiv_id` already shipped in a prior post. The
file is appended to (via `mark_paper_as_used`) only after a post writes
successfully, so a failed run never permanently consumes a paper.

Importable: no top-level side effects.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

log = logging.getLogger(__name__)

HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
ARXIV_API = "http://export.arxiv.org/api/query"
USER_AGENT = "ahmadnayfeh-portfolio-bot/0.1 (+https://github.com/Ahmad-Nayfeh/portfolio-website)"
TIMEOUT_SECONDS = 30

# Repo root — used to resolve relative paths in stream configs (e.g. the
# dedup_state_file). Mirrors the convention in load_streams.REPO_ROOT.
REPO_ROOT = Path(__file__).resolve().parent.parent


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
    debug_logged = False  # log structure of the first HF item we see, for diagnosis
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
            if not debug_logged:
                # One-shot debug: HF reshuffled their JSON in the past, so when
                # the upvote field disappears we want to see exactly what keys
                # they're returning today.
                log.info(
                    "HF debug — wrapper keys: %s; paper keys: %s; sample upvote candidates: "
                    "wrapper.numUniqueUpvotes=%r wrapper.upvotes=%r paper.upvotes=%r",
                    sorted(item.keys()),
                    sorted(paper.keys()),
                    item.get("numUniqueUpvotes"),
                    item.get("upvotes"),
                    paper.get("upvotes"),
                )
                debug_logged = True
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
        # HF has moved this field around historically. Look in both the
        # wrapper and the nested paper object, and accept several aliases.
        "upvotes": (
            paper.get("upvotes")
            or paper.get("numUniqueUpvotes")
            or wrapper.get("numUniqueUpvotes")
            or wrapper.get("upvotes")
            or 0
        ),
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


# ---------------------------------------------------------------------------
# Deduplication state — stops the same paper from being picked twice across
# weekly runs. The state file is small (one record per published post), so
# we read it eagerly and write it eagerly. JSON keeps it diff-friendly in
# git, which matters when a maintainer wants to see the publishing history
# without grepping through MDX files.
# ---------------------------------------------------------------------------


_DEFAULT_DEDUP_STATE = "pipeline-state/used-papers.json"


def _resolve_state_path(stream_cfg: Any) -> Path:
    """Return the absolute path to the dedup state file for this stream.

    Looks first at `discovery.dedup_state_file` on the stream config, then
    falls back to the repo-level default. Either form may be relative; we
    always resolve against REPO_ROOT so the file is the same one whether
    we're running from the repo root or from inside scripts/.
    """
    discovery = getattr(stream_cfg, "discovery", None)
    raw = None
    if discovery is not None:
        raw = getattr(discovery, "dedup_state_file", None)
        if raw is None:
            # The pydantic model uses extra="allow", so unknown fields land
            # in the model_extra dict; check there too.
            extra = getattr(discovery, "model_extra", None) or {}
            raw = extra.get("dedup_state_file")
    raw = raw or _DEFAULT_DEDUP_STATE
    p = Path(raw)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def load_used_paper_ids(stream_cfg: Any) -> set[str]:
    """Read the dedup state file and return the set of arxiv ids already used.

    Missing file is treated as an empty set — first run shouldn't fail just
    because the ledger doesn't exist yet. Malformed JSON is logged and
    treated as empty for the same reason; a corrupted file shouldn't block
    publishing, only the dedup guarantee.
    """
    path = _resolve_state_path(stream_cfg)
    if not path.exists():
        log.info("Dedup state %s not found; treating as empty.", path)
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Dedup state %s is malformed (%s); treating as empty.", path, e)
        return set()
    used: set[str] = set()
    for entry in data.get("papers") or []:
        aid = (entry.get("arxiv_id") or "").strip()
        if aid:
            used.add(aid)
    log.info("Dedup: %d papers already used in past posts.", len(used))
    return used


def mark_paper_as_used(
    stream_cfg: Any,
    paper: dict[str, Any],
    *,
    slug: str | None = None,
    when: dt.datetime | None = None,
) -> None:
    """Append `paper` to the dedup state file.

    Called by main.py after `write_post` succeeds. We do NOT write to this
    file before publishing — a crashed run shouldn't permanently burn a
    paper. The file is created on first write.
    """
    arxiv_id = (paper.get("arxiv_id") or "").strip()
    if not arxiv_id:
        log.info("mark_paper_as_used: paper has no arxiv_id; skipping.")
        return
    path = _resolve_state_path(stream_cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(
                "Dedup state %s is malformed (%s); resetting to a fresh ledger.",
                path, e,
            )
            data = {}
    else:
        data = {}
    data.setdefault("schema_version", 1)
    papers_list = data.setdefault("papers", [])
    # Don't double-record a paper if (somehow) it's already in the ledger.
    for entry in papers_list:
        if (entry.get("arxiv_id") or "").strip() == arxiv_id:
            log.info("Dedup: %s already in ledger; not re-adding.", arxiv_id)
            return
    when = when or dt.datetime.utcnow()
    papers_list.append({
        "arxiv_id": arxiv_id,
        "title": (paper.get("title") or "").strip(),
        "url": (paper.get("url") or "").strip(),
        "stream_id": getattr(stream_cfg.stream, "id", "") if hasattr(stream_cfg, "stream") else "",
        "slug": slug or "",
        "added_at": when.replace(microsecond=0).isoformat() + "Z",
    })
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    log.info("Dedup: recorded %s -> %s", arxiv_id, path)


def _filter_used(
    candidates: list[dict[str, Any]],
    used_ids: set[str],
) -> list[dict[str, Any]]:
    """Drop candidates whose arxiv_id already shipped in a prior post."""
    if not used_ids:
        return candidates
    kept: list[dict[str, Any]] = []
    dropped = 0
    for c in candidates:
        if (c.get("arxiv_id") or "").strip() in used_ids:
            dropped += 1
            continue
        kept.append(c)
    if dropped:
        log.info("Dedup: dropped %d candidate(s) already used.", dropped)
    return kept


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def discover_for_stream(stream_cfg: Any) -> list[dict[str, Any]]:
    """Discover candidates for a stream config object (StreamConfig from
    load_streams.py). Returns the top_n filtered candidates, ranked by
    upvotes desc, with already-published papers removed.

    The umbrella selector (scripts/select_papers.py) consumes this list
    and decides which 1-3 papers to actually write about. We deliberately
    keep `top_n` wider than the final pick so the selector has room to
    look for a thematic group.
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

    # Apply the dedup filter AFTER ranking so the log message about how many
    # we dropped reflects the post-filter top of the pile.
    used_ids = load_used_paper_ids(stream_cfg)
    candidates = _filter_used(candidates, used_ids)

    return candidates[:top_n]
