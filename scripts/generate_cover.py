"""scripts/generate_cover.py

DALL-E 3 cover image generation stage — Nocturne style (May 2026).
"""
from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import Any

import cost_meter

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

STYLE_SUFFIX = (
    "Flat editorial vector illustration. Geometric and abstract. Minimal "
    "composition with generous negative space. Dark background (approximately "
    "#080812). Strict three-color palette: electric teal (#00d4aa), warm "
    "amber (#ffba08), and hot magenta (#ff3cac) accents. Crisp vector shapes "
    "with clean edges and subtle glow. No gradients, no shading, no rasterized "
    "textures, no photorealism. No people, no faces, no human figures. No text, "
    "no letters, no captions, no logos. Scientific magazine cover energy, "
    "nocturnal and luminous."
)


def _openai_client():
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
    brief = (stage_outputs.get("cover_image_brief") or "").strip()
    if brief:
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
        except Exception as e:
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
