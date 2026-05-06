"""scripts/select_papers.py

Umbrella-aware paper selection.

The discovery layer (`discover_papers.py`) returns a wide pool of recent,
high-upvote, dedup-filtered candidates from the last ~30 days. This module
asks Claude to look at that pool and pick the right SHAPE for the post:

  * 2-3 papers that share a clean thematic umbrella (e.g. "test-time
    scaling", "diffusion guidance", "long-context retrieval"), OR
  * a single best paper if no honest umbrella holds.

The "honest umbrella" rule matters: forcing a connection between unrelated
papers produces the kind of synthetic essay a reader can smell from a mile
away. We'd rather ship one strong paper writeup than two papers stapled
together with rhetorical filler.

Selection method is configurable per stream via
`discovery.selection.method`:
  - "claude_umbrella_picks"  -> this module
  - "claude_picks"           -> legacy: take top-N by upvotes
  - any other / unset        -> legacy fallback

Importable: no top-level side effects.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import cost_meter  # sibling module — same meter the generation stages use

log = logging.getLogger(__name__)


# Cap on how many candidates we send to Claude. With ~30-day HF Daily Papers
# lookback the pool is usually in the dozens; sending all of them is fine but
# burns tokens. Empirically the top 25 by upvotes gives us enough diversity
# for an umbrella without spending more than ~5k input tokens on this stage.
MAX_CANDIDATES_FOR_SELECTION = 25

# Cap on how many papers the selector may return. The user requested 2-3
# under an umbrella; we hard-clamp so a runaway model can't return 10.
MAX_PICKED_PAPERS = 3
MIN_UMBRELLA_PAPERS = 2


# ---------------------------------------------------------------------------
# Anthropic client (lazy import — same convention as generate_post.py)
# ---------------------------------------------------------------------------


def _client() -> Any:
    try:
        import anthropic  # noqa: WPS433
    except ImportError as e:
        raise RuntimeError(
            "anthropic package not installed. Run `pip install -r scripts/requirements.txt`."
        ) from e
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set."
        )
    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


_SELECTION_SYSTEM = (
    "You are an editor for a technical AI/ML blog. Your job is to look at "
    "a list of recent research papers and pick which ones to feature in "
    "this week's post. You care about thematic coherence and reader value, "
    "not novelty for its own sake. You are willing to recommend featuring "
    "only one paper if no honest connection between several papers exists "
    "— the worst outcome is an essay that fakes a theme."
)


_SELECTION_INSTRUCTIONS = """
You are choosing papers for ONE blog post. Pick whichever option produces the
better post:

OPTION A — Umbrella post (preferred when honest):
  Pick {min_n}-{max_n} papers that genuinely sit under one clear theme.
  Examples of good umbrellas: "long-context evaluation", "diffusion model
  inference-time control", "agentic tool use", "low-rank fine-tuning".
  A good umbrella is something a single paragraph could introduce without
  hand-waving. If you have to write "while these papers approach different
  problems, they share a spirit of..." — that's not an umbrella, that's
  filler. Reject it.

OPTION B — Single-paper deep dive:
  If no honest umbrella exists, pick the SINGLE strongest paper. Strength
  here means: novel result, clear methodology, and broad enough to
  interest a non-specialist engineer. Default to this when in doubt.

Rules:
  - You may NOT pick more than {max_n} papers.
  - You may NOT pick fewer than 1.
  - Prefer fewer papers over more when uncertain. One excellent writeup
    beats two stretched-thin writeups.
  - Use the listed upvote count as a SIGNAL but not a tiebreaker — a
    less-upvoted paper that fits the umbrella beats a top-upvoted outlier.

Return your decision as a single JSON object on its own. No prose before or
after, no markdown fences. Schema:

{{
  "mode": "umbrella" | "single",
  "umbrella_theme": "<short phrase, or null when mode=single>",
  "rationale": "<one or two sentences explaining the choice>",
  "selected_arxiv_ids": ["<id1>", "<id2>", ...]
}}

The arxiv ids must come from the candidate list below.
"""


def _format_candidates(candidates: list[dict[str, Any]]) -> str:
    """Render the candidates as a numbered, abstract-bearing block.

    Truncate each abstract to ~600 chars so the full input prompt stays
    well under the model's context window even when 25 papers are included.
    """
    lines: list[str] = []
    for i, c in enumerate(candidates, start=1):
        lines.append(f"### Candidate {i}")
        lines.append(f"- arxiv_id: {c.get('arxiv_id', '')}")
        lines.append(f"- title: {c.get('title', '(untitled)')}")
        lines.append(f"- upvotes: {c.get('upvotes', 0)}")
        if c.get("authors"):
            authors = c["authors"][:5]
            suffix = " et al." if len(c["authors"]) > 5 else ""
            lines.append(f"- authors: {', '.join(authors)}{suffix}")
        cats = c.get("categories") or []
        if cats:
            lines.append(f"- categories: {', '.join(cats[:6])}")
        summary = (c.get("summary") or "").strip().replace("\n", " ")
        if summary:
            if len(summary) > 600:
                summary = summary[:600].rstrip() + "..."
            lines.append("- abstract:")
            lines.append(f"  {summary}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first {...} JSON object out of `text`. Tolerant of stray
    markdown fences and prose wrappers around the JSON, since models
    sometimes ignore "no prose" instructions.
    """
    if not text:
        return None
    text = text.strip()
    # Strip markdown fences if Claude wrapped the JSON in ```json ... ```.
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        inner = fence_match.group(1).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass  # fall through to the broader regex
    m = _JSON_OBJECT_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        log.warning("Selector returned non-JSON: %s. Raw: %s", e, text[:200])
        return None


def _papers_by_id(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for c in candidates:
        aid = (c.get("arxiv_id") or "").strip()
        if aid:
            by_id[aid] = c
    return by_id


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def select_umbrella_picks(
    *,
    stream_cfg: Any,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ask Claude to pick {1, 2, 3} papers under an umbrella from `candidates`.

    Falls back to the legacy "top N by upvotes" behaviour if:
      - the candidate list is empty
      - the API call fails
      - Claude returns malformed JSON
      - Claude picks ids that aren't in the candidate list
    so a selector bug never blocks a publish run.

    Returns the chosen subset of `candidates` in the order Claude listed
    them (the umbrella reads more naturally when the model's preferred
    "lead" paper is first).
    """
    if not candidates:
        return []

    # Hard fallback path used by both `else` branches and the rescue clauses.
    desired_count = _legacy_count(stream_cfg)

    def _fallback(reason: str) -> list[dict[str, Any]]:
        log.info("Selector fallback (%s): taking top %d by upvotes.", reason, desired_count)
        return candidates[:desired_count]

    pool = candidates[:MAX_CANDIDATES_FOR_SELECTION]

    instructions = _SELECTION_INSTRUCTIONS.format(
        min_n=MIN_UMBRELLA_PAPERS, max_n=MAX_PICKED_PAPERS,
    )
    user_msg = (
        f"Candidate papers (ranked by community upvotes):\n\n"
        f"{_format_candidates(pool)}\n\n"
        f"---\n\n{instructions}"
    )

    try:
        client = _client()
    except Exception as e:
        return _fallback(f"client unavailable: {e}")

    model = stream_cfg.generation.model
    fallback_model = stream_cfg.generation.fallback_model
    # Selection only needs a tiny output (a JSON blob). Cap it so a model
    # that ignores the format instruction can't burn 16k tokens of prose.
    max_tokens = 800

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_SELECTION_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as e:  # pragma: no cover — depends on SDK exceptions
        log.warning("Selector call failed on %s: %s. Trying %s.", model, e, fallback_model)
        try:
            response = client.messages.create(
                model=fallback_model,
                max_tokens=max_tokens,
                system=_SELECTION_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
        except Exception as e2:
            return _fallback(f"both models failed: {e2}")

    # Record cost. Same pattern as generate_post._call_claude.
    meter = cost_meter.get_meter()
    if meter is not None:
        usage = getattr(response, "usage", None)
        if usage is not None:
            try:
                meter.record_claude(
                    label="paper_selection",
                    model=getattr(response, "model", model),
                    input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                    output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                )
            except cost_meter.CostCeilingExceeded:
                # Selection is the very first model call in the run — if the
                # ceiling tripped here something is misconfigured. Re-raise
                # so main.py can abort the stream cleanly.
                raise
            except Exception as e:  # pragma: no cover — defensive
                log.warning("cost_meter: failed to record selector usage (%s)", e)

    text = "".join(
        block.text for block in response.content
        if getattr(block, "type", None) == "text"
    ).strip()
    decision = _extract_json_object(text)
    if not decision:
        return _fallback("response was not valid JSON")

    mode = (decision.get("mode") or "").strip().lower()
    raw_ids = decision.get("selected_arxiv_ids") or []
    theme = (decision.get("umbrella_theme") or "").strip() if mode == "umbrella" else ""
    rationale = (decision.get("rationale") or "").strip()

    if not isinstance(raw_ids, list):
        return _fallback("selected_arxiv_ids was not a list")

    by_id = _papers_by_id(pool)
    picked: list[dict[str, Any]] = []
    for raw in raw_ids:
        if not isinstance(raw, str):
            continue
        aid = raw.strip()
        if aid in by_id and by_id[aid] not in picked:
            picked.append(by_id[aid])
        if len(picked) >= MAX_PICKED_PAPERS:
            break

    if not picked:
        return _fallback("no selected ids matched the candidate list")

    if mode == "umbrella" and len(picked) < MIN_UMBRELLA_PAPERS:
        # Model said "umbrella" but only one paper survived. Treat as single.
        log.info("Selector: umbrella mode collapsed to single paper after id matching.")
        picked = picked[:1]
        mode = "single"

    log.info(
        "Selector: mode=%s, theme=%r, picked %d paper(s); rationale=%r",
        mode, theme, len(picked), rationale[:200],
    )
    for p in picked:
        log.info("  - %s (%s, upvotes=%s)", p.get("arxiv_id"), p.get("title", "")[:80], p.get("upvotes"))
    return picked


def _legacy_count(stream_cfg: Any) -> int:
    """Pre-umbrella behaviour: honor `discovery.selection.count`, default 1.

    Used as a fallback when the selector can't run. We default to 1 (single
    paper) rather than 2 because a fallback should produce the safest post,
    not the most ambitious one.
    """
    discovery = getattr(stream_cfg, "discovery", None)
    if discovery is None:
        return 1
    selection = getattr(discovery, "selection", None) or {}
    if isinstance(selection, dict):
        count = selection.get("count")
    else:
        count = getattr(selection, "count", None)
    if isinstance(count, int) and count > 0:
        return min(count, MAX_PICKED_PAPERS)
    return 1
