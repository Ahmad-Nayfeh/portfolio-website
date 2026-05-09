"""scripts/generate_inline_images.py

Inline illustration generation — Nocturne style (May 2026).
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

import cost_meter

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

INLINE_STYLE_SUFFIX = (
    "Flat editorial vector illustration in the style of Distill.pub explainer "
    "figures. Diagrammatic and abstract. Centered composition filling roughly "
    "70 percent of the canvas with breathing room around. Dark background "
    "(approximately #080812). Strict three-color palette: electric teal "
    "(#00d4aa), warm amber (#ffba08), and hot magenta (#ff3cac) accents. "
    "Crisp vector shapes, hairline strokes, subtle glow effects, no gradients, "
    "no shading, no rasterized textures, no photorealism. No people, no faces, "
    "no human figures. No text, no letters, no captions, no logos. Schematic "
    "and symbolic, like a luminous margin sketch in a research notebook."
)

PLACEHOLDER_RE = re.compile(r"\{\{IMG:([A-Z0-9_]+)\}\}")

MAX_INLINE_IMAGES = 4

INLINE_SIZE = "1024x1024"
INLINE_QUALITY = "standard"


def _openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "openai package not installed. Run `pip install -r scripts/requirements.txt`."
        ) from e
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set."
        )
    return OpenAI(api_key=api_key)


def _build_dalle_prompt(brief: str) -> str:
    return f"{brief.rstrip('. ')}. {INLINE_STYLE_SUFFIX}"


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


_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    m = _JSON_OBJECT_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        log.warning("inline image briefs not parseable as JSON: %s. Raw: %s", e, text[:200])
        return None


def parse_briefs(stage_text: str) -> list[dict[str, str]]:
    obj = _extract_json_object(stage_text)
    if not obj or not isinstance(obj.get("images"), list):
        return []
    out: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for raw in obj["images"]:
        if not isinstance(raw, dict):
            continue
        rid = str(raw.get("id") or "").strip().upper()
        rid = re.sub(r"[^A-Z0-9_]", "_", rid)
        if not rid or rid in seen_ids:
            continue
        brief = str(raw.get("brief") or "").strip()
        if not brief:
            continue
        alt = str(raw.get("alt") or "").strip() or "Inline illustration"
        out.append({"id": rid, "brief": brief, "alt": alt})
        seen_ids.add(rid)
        if len(out) >= MAX_INLINE_IMAGES:
            break
    return out


def _render_one(
    *,
    client: Any,
    brief: dict[str, str],
    out_dir: Path,
) -> str | None:
    image_id = brief["id"]
    target = out_dir / f"inline_{image_id.lower()}.png"
    prompt = _build_dalle_prompt(brief["brief"])
    log.info("Inline image %s: prompt=%d chars", image_id, len(prompt))

    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=INLINE_SIZE,
            quality=INLINE_QUALITY,
            style="natural",
            n=1,
            response_format="url",
        )
    except Exception as e:
        log.warning("Inline image %s: DALL-E call failed (%s).", image_id, e)
        return None

    meter = cost_meter.get_meter()
    if meter is not None:
        try:
            meter.record_dalle(
                label=f"inline_image:{image_id}",
                model="dall-e-3",
                size=INLINE_SIZE,
                quality=INLINE_QUALITY,
            )
        except cost_meter.CostCeilingExceeded:
            raise
        except Exception as e:
            log.warning("cost_meter: failed to record DALL-E inline usage (%s)", e)

    if not response.data:
        log.warning("Inline image %s: DALL-E returned no data.", image_id)
        return None
    image_url = response.data[0].url
    if not image_url:
        log.warning("Inline image %s: DALL-E returned no url.", image_id)
        return None

    try:
        _download(image_url, target)
    except Exception as e:
        log.warning("Inline image %s: download failed (%s).", image_id, e)
        return None

    public_path = "/" + str(target.relative_to(REPO_ROOT / "public")).replace(os.sep, "/")
    return public_path


def render_inline_images(
    *,
    stage_outputs: dict[str, str],
    slug: str,
    skip: bool = False,
) -> dict[str, dict[str, str]]:
    if skip:
        log.info("Inline image generation skipped (skip=True).")
        return {}

    raw = stage_outputs.get("inline_image_briefs", "").strip()
    if not raw:
        log.info("No inline_image_briefs stage output; skipping inline rendering.")
        return {}

    briefs = parse_briefs(raw)
    if not briefs:
        log.info("inline_image_briefs parsed as 0 valid entries; skipping.")
        return {}

    log.info("Inline images: rendering %d brief(s)...", len(briefs))

    try:
        client = _openai_client()
    except Exception as e:
        log.warning("Inline images: OpenAI client unavailable (%s) — skipping all.", e)
        return {}

    out_dir = REPO_ROOT / "public" / "blog-images" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, str]] = {}
    for brief in briefs:
        public_path = _render_one(client=client, brief=brief, out_dir=out_dir)
        if public_path:
            results[brief["id"]] = {"path": public_path, "alt": brief["alt"]}
            log.info("Inline image %s -> %s", brief["id"], public_path)
    return results


def substitute_placeholders(
    mdx: str,
    rendered: dict[str, dict[str, str]],
) -> tuple[str, int, int]:
    replaced = 0
    missing = 0

    def _sub(match: re.Match[str]) -> str:
        nonlocal replaced, missing
        placeholder_id = match.group(1)
        entry = rendered.get(placeholder_id)
        if entry:
            replaced += 1
            alt = entry.get("alt", "Inline illustration")
            path = entry.get("path", "")
            return f"\n\n![{alt}]({path})\n\n"
        missing += 1
        return f"<!-- IMG:{placeholder_id} not rendered -->"

    new_mdx = PLACEHOLDER_RE.sub(_sub, mdx)
    new_mdx = re.sub(r"\n{3,}", "\n\n", new_mdx)
    return new_mdx, replaced, missing


def render_and_substitute(
    *,
    stream_cfg: Any,
    stage_outputs: dict[str, str],
    mdx_path: Path,
    slug: str,
    skip: bool = False,
) -> dict[str, Any]:
    inline_cfg = getattr(stream_cfg, "inline_images", None)
    enabled = getattr(inline_cfg, "enabled", True) if inline_cfg else True
    if not enabled or skip:
        mdx = mdx_path.read_text(encoding="utf-8")
        new_mdx, _, missing = substitute_placeholders(mdx, {})
        if new_mdx != mdx:
            mdx_path.write_text(new_mdx, encoding="utf-8")
        return {
            "ran": False,
            "reason": "disabled" if not enabled else "skipped",
            "rendered": {},
            "replaced": 0,
            "missing": missing,
        }

    rendered = render_inline_images(
        stage_outputs=stage_outputs, slug=slug, skip=skip,
    )
    mdx = mdx_path.read_text(encoding="utf-8")
    new_mdx, replaced, missing = substitute_placeholders(mdx, rendered)
    if new_mdx != mdx:
        mdx_path.write_text(new_mdx, encoding="utf-8")
        log.info(
            "Inline images: substituted %d placeholder(s); %d remained unrendered.",
            replaced, missing,
        )
    return {
        "ran": True,
        "reason": None,
        "rendered": {k: v["path"] for k, v in rendered.items()},
        "replaced": replaced,
        "missing": missing,
    }
