"""scripts/generate_post.py

Multi-stage post generation. Each stage is a separate Claude API call with
narrow scope — this is a deliberate design choice to reduce hallucination
versus one giant prompt. The orchestrator stitches the stage outputs into a
single MDX document.

Stages (configurable in the stream YAML):
  1. paper_summary       — restate each paper's contribution in plain language.
  2. method_explanation  — explain the method, with $$...$$ for key math.
  3. quote_extraction    — pull verbatim quotes with section markers.
  4. critique            — strengths/weaknesses, no hype.
  5. demo_code           — minimal didactic Python demo.
  6. synthesis           — closing section connecting the papers.

The post body is assembled in the order the stages are listed.

Importable: no top-level side effects.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import re
from pathlib import Path
from typing import Any

import cost_meter  # sibling module — see scripts/cost_meter.py

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTENT_BLOG = REPO_ROOT / "content" / "blog"


# ---------------------------------------------------------------------------
# Anthropic client (lazy import so importing this module doesn't require the
# anthropic package — tests / dry runs may not have it installed).
# ---------------------------------------------------------------------------


def _client():
    try:
        import anthropic  # noqa: WPS433 (deliberate lazy import)
    except ImportError as e:
        raise RuntimeError(
            "anthropic package not installed. Run `pip install -r scripts/requirements.txt`."
        ) from e
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. Add it to your "
            "GitHub repo's Actions secrets (Settings -> Secrets and variables "
            "-> Actions -> New repository secret)."
        )
    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Stage execution
# ---------------------------------------------------------------------------


def _call_claude(
    *,
    client: Any,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    label: str = "claude",
) -> str:
    """One round-trip to Claude. Returns the assistant message text.

    Records token usage with the cost meter (if one is active) so the run
    can be aborted before a buggy stage burns the whole budget. The meter
    is module-level state initialized by main.py — if it's None (e.g.
    tests that import this module directly), recording is a no-op.

    `label` is what shows up in the cost-meter line entries — typically
    the stage name. The orchestrator stage loop overrides the default.
    """
    log.info("Calling %s (system=%d chars, user=%d chars)", model, len(system), len(user))
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    # Record cost. Wrapped defensively because (a) the meter may not be
    # initialized in tests, and (b) the SDK's `usage` shape could change in a
    # future version — we don't want a billing tracker to break generation.
    meter = cost_meter.get_meter()
    if meter is not None:
        usage = getattr(response, "usage", None)
        if usage is not None:
            try:
                meter.record_claude(
                    label=label,
                    model=model,
                    input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                    output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                )
            except cost_meter.CostCeilingExceeded:
                # Re-raise — main.py catches this and aborts the stream.
                raise
            except Exception as e:  # pragma: no cover — defensive
                log.warning("cost_meter: failed to record Claude usage (%s)", e)
        else:
            log.warning("cost_meter: response has no usage field; skipping record.")

    parts: list[str] = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip()


def _format_papers_for_prompt(papers: list[dict[str, Any]]) -> str:
    """Render the discovered papers as a structured block the model can cite."""
    lines: list[str] = []
    for i, p in enumerate(papers, start=1):
        lines.append(f"### Paper {i}: {p.get('title', '(untitled)')}")
        lines.append(f"- arXiv ID: {p.get('arxiv_id', '')}")
        lines.append(f"- URL: {p.get('url', '')}")
        if p.get("authors"):
            lines.append(f"- Authors: {', '.join(p['authors'])}")
        if p.get("summary"):
            lines.append("")
            lines.append("Abstract:")
            lines.append(p["summary"])
        lines.append("")
    return "\n".join(lines)


def run_stages(
    *,
    stream_cfg: Any,
    papers: list[dict[str, Any]],
) -> dict[str, str]:
    """Run each generation stage and return a dict of stage_name -> markdown."""
    client = _client()
    model = stream_cfg.generation.model
    fallback = stream_cfg.generation.fallback_model
    max_tokens = stream_cfg.generation.max_output_tokens
    system = stream_cfg.generation.system_prompt or ""

    paper_block = _format_papers_for_prompt(papers)
    stages = stream_cfg.generation.stages or [
        {"name": "main", "task": stream_cfg.generation.task_instructions or "Write the full post."}
    ]

    results: dict[str, str] = {}
    accumulated_context = ""

    for stage in stages:
        name = stage.get("name", "stage")
        task = stage.get("task", "")
        user_msg = (
            f"Source papers:\n\n{paper_block}\n\n"
            f"---\n\nTask for this stage ({name}):\n{task}\n\n"
        )
        if accumulated_context:
            user_msg += (
                f"---\n\nWhat you've written in earlier stages of this post "
                f"(for continuity — don't repeat, build on it):\n\n{accumulated_context}\n"
            )
        try:
            text = _call_claude(
                client=client, model=model, system=system, user=user_msg,
                max_tokens=max_tokens, label=f"stage:{name}",
            )
        except cost_meter.CostCeilingExceeded:
            # Don't fall back to a different model — we're already over budget.
            raise
        except Exception as e:  # pragma: no cover — depends on the SDK's exception classes
            log.warning("Stage %s failed on %s: %s. Trying fallback %s", name, model, e, fallback)
            text = _call_claude(
                client=client, model=fallback, system=system, user=user_msg,
                max_tokens=max_tokens, label=f"stage:{name}:fallback",
            )
        results[name] = text
        accumulated_context += f"\n\n## {name}\n\n{text}"

    return results


# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------


_QUOTE_BLOCK_RE = re.compile(r"^>[ \t]+\S", re.MULTILINE)


def count_verbatim_quotes(markdown: str) -> int:
    """Count distinct blockquote groups in `markdown`. Consecutive `>` lines
    count as one quote.
    """
    in_quote = False
    count = 0
    for line in markdown.splitlines():
        is_quote_line = bool(_QUOTE_BLOCK_RE.match(line))
        if is_quote_line and not in_quote:
            count += 1
            in_quote = True
        elif not is_quote_line:
            in_quote = False
    return count


def regenerate_quote_stage(
    *,
    stream_cfg: Any,
    papers: list[dict[str, Any]],
    stage_outputs: dict[str, str],
    current_count: int,
    target_count: int,
) -> dict[str, str]:
    """Re-run the quote-extraction stage with a stronger prompt.

    Called by the orchestrator when `count_verbatim_quotes(...)` came back
    below `quality_gates.require_verbatim_quotes`. Mutates a copy of
    `stage_outputs` and returns the new dict so the orchestrator can re-write
    the MDX from scratch.

    The function looks for a stage whose name contains "quote" (case-
    insensitive). If no such stage exists, it logs a warning and returns the
    inputs unchanged — there's nothing to regenerate.
    """
    stages = stream_cfg.generation.stages or []
    quote_stage = next(
        (s for s in stages if "quote" in (s.get("name") or "").lower()),
        None,
    )
    if quote_stage is None:
        log.warning(
            "Quote check failed (%d/%d) but no quote stage in stream config; skipping retry.",
            current_count, target_count,
        )
        return stage_outputs

    name = quote_stage.get("name", "quote_extraction")
    base_task = quote_stage.get("task", "")
    paper_block = _format_papers_for_prompt(papers)

    stronger_task = (
        f"{base_task}\n\n"
        f"IMPORTANT — RETRY CONTEXT: a previous attempt produced only "
        f"{current_count} verbatim blockquote(s) in markdown. We require at "
        f"least {target_count}. Each verbatim quote MUST:\n"
        f"  1. Be wrapped as a markdown blockquote (lines starting with `> `).\n"
        f"  2. Be a continuous span of text copied EXACTLY from the abstract or "
        f"     summary of one of the source papers above — no paraphrasing.\n"
        f"  3. Be followed by an attribution line citing which paper it came from.\n"
        f"Output the full revised quote section. Produce at least "
        f"{max(target_count, 2)} blockquote groups."
    )

    user_msg = (
        f"Source papers:\n\n{paper_block}\n\n"
        f"---\n\nTask for this stage ({name}, RETRY):\n{stronger_task}\n"
    )

    client = _client()
    model = stream_cfg.generation.model
    fallback = stream_cfg.generation.fallback_model
    max_tokens = stream_cfg.generation.max_output_tokens
    system = stream_cfg.generation.system_prompt or ""

    try:
        text = _call_claude(
            client=client, model=model, system=system, user=user_msg,
            max_tokens=max_tokens, label=f"stage:{name}:retry",
        )
    except cost_meter.CostCeilingExceeded:
        raise
    except Exception as e:  # pragma: no cover
        log.warning("Quote retry failed on %s: %s. Trying fallback %s", model, e, fallback)
        text = _call_claude(
            client=client, model=fallback, system=system, user=user_msg,
            max_tokens=max_tokens, label=f"stage:{name}:retry:fallback",
        )

    new_outputs = dict(stage_outputs)
    new_outputs[name] = text
    log.info("Quote retry produced %d new chars for stage %s", len(text), name)
    return new_outputs


# ---------------------------------------------------------------------------
# MDX assembly
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "post"


def make_slug(stream_cfg: Any, papers: list[dict[str, Any]], now: dt.datetime) -> str:
    """Format `slug_pattern` from the stream config."""
    pattern = stream_cfg.content.slug_pattern or "{year}-W{week}-{topic-slug}"
    iso = now.isocalendar()
    topic = "ai-papers"
    if papers:
        topic = slugify(papers[0].get("title", "ai-papers"))[:50]
    return (
        pattern.replace("{year}", str(iso[0]))
        .replace("{week}", f"{iso[1]:02d}")
        .replace("{topic-slug}", topic)
        .replace("{stream-id}", stream_cfg.stream.id)
    )


def _compute_read_time(body_parts: list[str]) -> str:
    """Estimate reading time at ~200 wpm.

    Strips fenced code blocks and math (block + inline) before counting,
    since those don't read at the same speed as prose. Always returns at
    least "1 min read" so empty/very-short posts don't display "0".
    """
    text = "\n".join(body_parts)
    # Strip fenced code blocks (```...```), block math ($$...$$), inline math ($...$).
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"\$\$[\s\S]*?\$\$", "", text)
    text = re.sub(r"\$[^$\n]+\$", "", text)
    word_count = len(text.split())
    minutes = max(1, round(word_count / 200))
    return f"{minutes} min read"


_KEBAB_SAFE_RE = re.compile(r"[^a-z0-9-]+")


def _normalize_tag(raw: str) -> str:
    """Normalize an extracted tag: lower, strip, hyphenate, dedupe."""
    s = raw.strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = _KEBAB_SAFE_RE.sub("-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def _parse_extracted_tags(raw: str, max_tags: int = 5) -> list[str]:
    """Parse the tag_extraction stage's output into a clean list.

    Expects a single line of comma-separated kebab-case tags. Tolerant of
    minor formatting variation (newlines, extra whitespace, bullets,
    trailing periods). Returns at most `max_tags` unique tags.
    """
    if not raw:
        return []
    # Take the densest non-empty line if the model snuck in a heading.
    candidates = [
        line for line in (l.strip().lstrip("-•*").strip() for l in raw.splitlines())
        if line and "," in line
    ]
    line = candidates[0] if candidates else raw.strip()
    tags: list[str] = []
    seen: set[str] = set()
    for piece in line.split(","):
        norm = _normalize_tag(piece)
        if norm and norm not in seen:
            seen.add(norm)
            tags.append(norm)
        if len(tags) >= max_tags:
            break
    return tags


def _is_metadata_stage(stage: dict[str, Any]) -> bool:
    """A stage with role: metadata is parsed for frontmatter, not rendered."""
    return (stage.get("role") or "").strip().lower() == "metadata"


def _format_title(template: str, papers: list[dict[str, Any]]) -> str:
    """Render the stream's title_template with the first paper's title."""
    topic = papers[0].get("title", "Notes") if papers else "Notes"
    return template.replace("{topic}", topic)


def assemble_mdx(
    *,
    stream_cfg: Any,
    papers: list[dict[str, Any]],
    stage_outputs: dict[str, str],
    slug: str,
    now: dt.datetime,
    cover_image: str | None = None,
) -> str:
    """Stitch stage outputs into one MDX document with frontmatter.

    `cover_image`, when provided, is written to the frontmatter as a
    public-relative path (e.g. `/blog-images/<slug>/cover.png`) so the
    blog post page can render it without further plumbing.
    """
    # Per-stream title template (falls back to legacy "Paper Notes:" prefix).
    title_template = (
        getattr(stream_cfg.content, "title_template", None) or "Paper Notes: {topic}"
    )
    title = _format_title(title_template, papers)

    # Excerpt: use the intro_framing if available (new structure), else fall
    # back to paper_summary (old structure), else first non-empty stage.
    excerpt_source = (
        stage_outputs.get("intro_framing")
        or stage_outputs.get("paper_summary")
        or next((v for v in stage_outputs.values() if v), "")
    )
    excerpt = excerpt_source[:280].replace("\n", " ").strip()

    # Build the body — skip metadata-only stages (e.g. tag_extraction).
    body_parts: list[str] = []
    extracted_tags: list[str] = []
    for stage in stream_cfg.generation.stages or []:
        name = stage.get("name", "")
        text = stage_outputs.get(name, "").strip()
        if not text:
            continue
        if _is_metadata_stage(stage):
            # tag_extraction is the only metadata stage we know about today;
            # if more appear later, parse by name here.
            if name == "tag_extraction":
                extracted_tags = _parse_extracted_tags(text)
            continue
        body_parts.append(text)
    if not body_parts:
        # Defensive fallback: if no stages produced output, emit whatever we
        # got (excluding any metadata stages we already pulled out).
        meta_names = {
            (s.get("name") or "")
            for s in (stream_cfg.generation.stages or [])
            if _is_metadata_stage(s)
        }
        body_parts = [v for k, v in stage_outputs.items() if v and k not in meta_names]

    read_time = _compute_read_time(body_parts)

    # Merge stream-base tags with extracted tags. Stream tags first
    # (broader), extracted tags after (more specific). Dedupe by normalized
    # form so "AI" doesn't appear twice if Claude also picks "ai".
    merged_tags: list[str] = []
    seen_tags: set[str] = set()
    for t in (stream_cfg.content.tags or []):
        norm = _normalize_tag(t) or t.strip()
        key = norm.lower()
        if key and key not in seen_tags:
            seen_tags.add(key)
            merged_tags.append(t)  # keep original casing for stream-level tags
    for t in extracted_tags:
        if t and t not in seen_tags:
            seen_tags.add(t)
            merged_tags.append(t)

    # Frontmatter (YAML, not JSON — readable on disk).
    #
    # `date` is written as ISO 8601 with the Riyadh offset (+03:00) so the
    # post page can render the publish *time* alongside the date. The site's
    # `formatDate`/`formatTime` helpers handle both this rich format and the
    # legacy "YYYY-MM-DD" used by hand-written posts.
    iso_date = _iso_riyadh(now)
    fm_lines = [
        "---",
        f'title: "{_yaml_escape(title)}"',
        f'slug: "{slug}"',
        f'date: "{iso_date}"',
        "tags:",
    ]
    for t in merged_tags:
        fm_lines.append(f'  - "{t}"')
    fm_lines.append(f'excerpt: "{_yaml_escape(excerpt)}"')
    fm_lines.append(f'readTime: "{read_time}"')
    fm_lines.append("featured: false")
    if cover_image:
        fm_lines.append(f'coverImage: "{cover_image}"')
    fm_lines.append(f'streamId: "{stream_cfg.stream.id}"')
    fm_lines.append(f'language: "{stream_cfg.stream.language}"')
    fm_lines.append(f'generatedBy: "{stream_cfg.generation.model}"')
    if papers:
        fm_lines.append("papers:")
        for p in papers:
            fm_lines.append(f'  - title: "{_yaml_escape(p.get("title", ""))}"')
            fm_lines.append(f"    url: \"{p.get('url', '')}\"")
            if p.get("arxiv_id"):
                fm_lines.append(f"    arxivId: \"{p['arxiv_id']}\"")
            if p.get("authors"):
                fm_lines.append("    authors:")
                for a in p["authors"]:
                    fm_lines.append(f'      - "{_yaml_escape(a)}"')
    fm_lines.append("---")
    fm_lines.append("")

    return "\n".join(fm_lines) + "\n\n".join(body_parts) + "\n"


def _yaml_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


# Asia/Riyadh = UTC+3, no DST. We don't import pytz/zoneinfo to avoid an
# extra dependency — a fixed offset is correct for Riyadh year-round.
_RIYADH_OFFSET = dt.timezone(dt.timedelta(hours=3))


def _iso_riyadh(now: dt.datetime) -> str:
    """Return `now` as an ISO 8601 string in Riyadh local time.

    Accepts either a naive datetime (assumed UTC, like `dt.datetime.utcnow()`)
    or a timezone-aware one. Output looks like `2026-05-04T13:00:00+03:00`.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    riyadh = now.astimezone(_RIYADH_OFFSET)
    # Strip microseconds for cleaner frontmatter; second resolution is plenty.
    riyadh = riyadh.replace(microsecond=0)
    return riyadh.isoformat()


def write_post(
    *,
    stream_cfg: Any,
    papers: list[dict[str, Any]],
    stage_outputs: dict[str, str],
    now: dt.datetime | None = None,
    cover_image: str | None = None,
) -> Path:
    """Write the assembled MDX to content/blog/<slug>.mdx and return the path.

    `cover_image` is forwarded to `assemble_mdx` and written to the
    frontmatter when present.
    """
    if now is None:
        now = dt.datetime.utcnow()
    slug = make_slug(stream_cfg, papers, now)
    mdx = assemble_mdx(
        stream_cfg=stream_cfg,
        papers=papers,
        stage_outputs=stage_outputs,
        slug=slug,
        now=now,
        cover_image=cover_image,
    )
    out_dir = REPO_ROOT / stream_cfg.content.output_path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slug}.mdx"
    out_path.write_text(mdx, encoding="utf-8")
    log.info("Wrote %s (%d chars)", out_path, len(mdx))
    return out_path
