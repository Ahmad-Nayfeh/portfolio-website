"""scripts/generate_cover.py

DALL-E 3 cover image generation stage.

Each generated post gets a cover image rendered from a short visual brief.
The brief is drawn from one of two sources, in priority order:

  1. A `cover_image_brief` stage in the stream config (role: metadata).
     If the model produced one, we use it verbatim — this gives the writing
     stage the chance to describe a scene specific to the paper.
  2. A fallback prompt built from the post title and the first ~500
     characters of the intro_framing stage. Crude but always available.

The final DALL-E prompt wraps the brief in a fixed style suffix so the
visual language is consistent across the whole stream — every cover should
look like it belongs to the same magazine.

Sizing & cost (DALL-E 3, May 2026):
  1024x1024 standard  = $0.04
  1792x1024 standard  = $0.08   <-- we use this (landscape blog cover)
  1024x1792 standard  = $0.08
  HD adds another ~$0.04–$0.08.

We use 1792x1024 standard. At one post per week, ~$4.16/year. The square
1024x1024 saves half that but looks wrong as a 16:9 blog header.

The function is fully optional — if the OpenAI key is missing or the API
fails, we log and return None so the rest of the pipeline keeps going.
"""
from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import Any

import cost_meter  # sibling module — see scripts/cost_meter.py

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

# A fixed style suffix that defines the "look" of every cover. Edit this in
# one place to re-skin the entire stream.
STYLE_SUFFIX = (
    "Editorial magazine cover illustration. Minimal, abstract, conceptual. "
    "Clean composition with negative space. Restrained color palette: warm "
    "off-white background, deep navy, with a single electric cobalt blue "
    "accent. Soft analog texture. Geometric and graphic, not photorealistic. "
    "No text, no letters, no captions, no logos. Engineering-notebook aesthetic."
)


def _openai_client():
    """Lazy import so dry runs and tests don't require openai installed."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "openai package not installed. Run `pip install -r scripts/requirements.txt`."
        ) from e
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. Add it to your "
            "GitHub repo's Actions secrets (Settings -> Secrets and variables "
            "-> Actions -> New repository secret)."
        )
    return OpenAI(api_key=api_key)


def _build_fallback_brief(papers: list[dict[str, Any]], stage_outputs: dict[str, str]) -> str:
    """Synthesize a one-line visual brief from the post's own content.

    Picked when the stream config doesn't include a cover_image_brief stage.
    Aims for "abstract scene about <topic>", not literal description.
    """
    title = papers[0].get("title") if papers else None
    intro = (stage_outputs.get("intro_framing") or "").strip()
    intro_excerpt = intro[:400].replace("\n", " ")
    if title and intro_excerpt:
        return (
            f"An abstract conceptual scene illustrating the central idea of "
            f"the paper '{title}'. Context: {intro_excerpt}"
        )
    if title:
        return f"An abstract conceptual scene illustrating: {title}"
    return "An abstract conceptual scene about a recent AI research result."


def _resolve_brief(
    *,
    stream_cfg: Any,
    papers: list[dict[str, Any]],
    stage_outputs: dict[str, str],
) -> str:
    """Find the visual brief, preferring a model-written one if present."""
    # Look for a stage named cover_image_brief OR with role: cover_brief.
    brief = (stage_outputs.get("cover_image_brief") or "").strip()
    if brief:
        # Take the first non-empty line so the model can't smuggle in a
        # 500-word essay (DALL-E max prompt is ~4000 chars but shorter is
        # always sharper).
        line = next(
            (ln.strip() for ln in brief.splitlines() if ln.strip()),
            "",
        )
        if line:
            return line[:800]
    return _build_fallback_brief(papers, stage_outputs)[:800]


def _build_dalle_prompt(brief: str) -> str:
    return f"{brief.rstrip('. ')}. {STYLE_SUFFIX}"


def _download(url: str, target: Path) -> None:
    """Stream the image bytes from `url` to `target`. Atomic-ish."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as f:
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(target)


def generate_cover_image(
    *,
    stream_cfg: Any,
    papers: list[dict[str, Any]],
    stage_outputs: dict[str, str],
    slug: str,
) -> str | None:
    """Generate a cover image and return the public path (e.g.
    `/blog-images/2026-W18-foo/cover.png`), or None if disabled / failed.

    The pipeline treats failure here as non-fatal: the post still ships
    without a cover image. We log loud enough that it's noticeable in the
    Actions log but not loud enough to break the build.
    """
    cover_cfg = getattr(stream_cfg, "cover_image", None)
    enabled = getattr(cover_cfg, "enabled", False) if cover_cfg else False
    if not enabled:
        log.info("Cover image stage disabled for stream %s", stream_cfg.stream.id)
        return None

    out_dir_pattern = getattr(cover_cfg, "output_dir", "public/blog-images/{slug}")
    out_dir = REPO_ROOT / out_dir_pattern.replace("{slug}", slug)
    out_path = out_dir / "cover.png"

    brief = _resolve_brief(stream_cfg=stream_cfg, papers=papers, stage_outputs=stage_outputs)
    prompt = _build_dalle_prompt(brief)
    log.info(
        "Cover image: brief=%d chars, full prompt=%d chars (target: %s)",
        len(brief), len(prompt), out_path.relative_to(REPO_ROOT),
    )

    model = getattr(cover_cfg, "model", "dall-e-3")
    size = getattr(cover_cfg, "size", "1792x1024")
    quality = getattr(cover_cfg, "quality", "standard")
    style = getattr(cover_cfg, "style", "natural")

    try:
        client = _openai_client()
    except Exception as e:
        log.warning("Cover image: OpenAI client unavailable (%s) — skipping.", e)
        return None

    try:
        response = client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            n=1,
            response_format="url",
        )
    except Exception as e:
        log.warning("Cover image: DALL-E call failed (%s) — skipping.", e)
        return None

    # Record cost. We bill on response (not before) because a failed call costs
    # nothing. Wrapped defensively so a meter bug never blocks a successful
    # image from shipping; CostCeilingExceeded does propagate, since the next
    # stage shouldn't fire if we've already exceeded the budget.
    meter = cost_meter.get_meter()
    if meter is not None:
        try:
            meter.record_dalle(
                label="cover_image",
                model=model,
                size=size,
                quality=quality,
            )
        except cost_meter.CostCeilingExceeded:
            raise
        except Exception as e:  # pragma: no cover — defensive
            log.warning("cost_meter: failed to record DALL-E usage (%s)", e)

    if not response.data:
        log.warning("Cover image: DALL-E returned no data — skipping.")
        return None
    image_url = response.data[0].url
    if not image_url:
        log.warning("Cover image: DALL-E returned no url — skipping.")
        return None

    try:
        _download(image_url, out_path)
    except Exception as e:
        log.warning("Cover image: download failed (%s) — skipping.", e)
        return None

    public_path = "/" + str(out_path.relative_to(REPO_ROOT / "public")).replace(
        os.sep, "/"
    )
    log.info("Cover image: wrote %s (public path %s)", out_path, public_path)
    return public_path
