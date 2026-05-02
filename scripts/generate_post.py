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
) -> str:
    """One round-trip to Claude. Returns the assistant message text."""
    log.info("Calling %s (system=%d chars, user=%d chars)", model, len(system), len(user))
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
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
                client=client, model=model, system=system, user=user_msg, max_tokens=max_tokens,
            )
        except Exception as e:  # pragma: no cover — depends on the SDK's exception classes
            log.warning("Stage %s failed on %s: %s. Trying fallback %s", name, model, e, fallback)
            text = _call_claude(
                client=client, model=fallback, system=system, user=user_msg, max_tokens=max_tokens,
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
            client=client, model=model, system=system, user=user_msg, max_tokens=max_tokens,
        )
    except Exception as e:  # pragma: no cover
        log.warning("Quote retry failed on %s: %s. Trying fallback %s", model, e, fallback)
        text = _call_claude(
            client=client, model=fallback, system=system, user=user_msg, max_tokens=max_tokens,
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


def assemble_mdx(
    *,
    stream_cfg: Any,
    papers: list[dict[str, Any]],
    stage_outputs: dict[str, str],
    slug: str,
    now: dt.datetime,
) -> str:
    """Stitch stage outputs into one MDX document with frontmatter."""
    title_root = papers[0].get("title", "AI Paper Notes") if papers else "AI Paper Notes"
    title = f"Paper Notes: {title_root}"
    excerpt = stage_outputs.get("paper_summary", "")[:280].replace("\n", " ").strip()

    # Frontmatter (YAML, not JSON — readable on disk).
    fm_lines = [
        "---",
        f'title: "{_yaml_escape(title)}"',
        f'slug: "{slug}"',
        f'date: "{now.strftime("%Y-%m-%d")}"',
        "tags:",
    ]
    for t in stream_cfg.content.tags or []:
        fm_lines.append(f'  - "{t}"')
    fm_lines.append(f'excerpt: "{_yaml_escape(excerpt)}"')
    fm_lines.append('readTime: "auto"')
    fm_lines.append("featured: false")
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

    body_parts: list[str] = []
    for stage in stream_cfg.generation.stages or []:
        name = stage.get("name", "")
        text = stage_outputs.get(name, "").strip()
        if text:
            body_parts.append(text)
    if not body_parts:
        body_parts = [v for v in stage_outputs.values() if v]

    return "\n".join(fm_lines) + "\n\n".join(body_parts) + "\n"


def _yaml_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def write_post(
    *,
    stream_cfg: Any,
    papers: list[dict[str, Any]],
    stage_outputs: dict[str, str],
    now: dt.datetime | None = None,
) -> Path:
    """Write the assembled MDX to content/blog/<slug>.mdx and return the path."""
    if now is None:
        now = dt.datetime.utcnow()
    slug = make_slug(stream_cfg, papers, now)
    mdx = assemble_mdx(
        stream_cfg=stream_cfg, papers=papers, stage_outputs=stage_outputs, slug=slug, now=now,
    )
    out_dir = REPO_ROOT / stream_cfg.content.output_path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slug}.mdx"
    out_path.write_text(mdx, encoding="utf-8")
    log.info("Wrote %s (%d chars)", out_path, len(mdx))
    return out_path
