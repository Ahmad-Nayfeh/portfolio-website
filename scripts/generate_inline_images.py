"""scripts/generate_inline_images.py

Inline illustration generation (H4 — May 2026).

Each generated post gets ONE big editorial cover image at the top (see
generate_cover.py). H4 adds 2-4 SMALLER inline illustrations woven into
the prose, so the post reads as a visual essay rather than a wall of
text broken only by code blocks.

How it works:

1. The `inline_image_briefs` metadata stage (defined in
   streams/ai-papers.yaml) asks Claude to output JSON describing 2-4
   illustrations with stable IDs (e.g. "OPENING", "DIALECTIC_1") and
   one-sentence visual briefs each.

2. The prose stages (umbrella_opening, dialectical_walk, ...) drop
   `{{IMG:ID}}` placeholder tokens at natural narrative beats. Each
   placeholder corresponds to one entry in the briefs JSON.

3. THIS module reads the briefs, renders each through DALL-E with the
   inline style suffix (similar palette to the cover but looser
   composition — "Distill.pub explainer figure" rather than "magazine
   cover"), saves the PNGs under `public/blog-images/<slug>/`, then
   rewrites the MDX file replacing each `{{IMG:ID}}` placeholder with
   a real markdown image reference.

Failure handling:
  - A failed render leaves its placeholder in the MDX as a HTML comment
    (`<!-- IMG:ID failed to render -->`) so the post still ships and
    the omission is visible in review.
  - Cost-meter integration mirrors generate_cover.py — billed on
    response, ceiling-checked, raises CostCeilingExceeded to abort
    cleanly.

Sizing & cost (DALL-E 3, May 2026):
  1024x1024 standard = $0.04
  1024x1024 HD       = $0.08
  We use 1024x1024 standard for inline. Three inline images per post +
  one HD cover = $0.04*3 + $0.12 = $0.24/post; at weekly cadence ~$12/yr.
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

# Locked inline style suffix. INTENTIONALLY DIFFERENT from the cover
# suffix (generate_cover.py:STYLE_SUFFIX) — same palette, but the
# composition language shifts from "magazine cover" to "in-text
# explainer figure". Cover images need to carry visual weight at
# 1792x1024 with the headline; inline images need to be readable at
# ~600px column width and feel like Distill.pub margin sketches.
INLINE_STYLE_SUFFIX = (
    "Flat editorial vector illustration in the style of Distill.pub "
    "explainer figures. Diagrammatic and abstract. Centered composition "
    "filling roughly 70 percent of the canvas with breathing room "
    "around. Strict three-color palette: warm off-white background "
    "(approximately #fbf6ec), deep navy primary (approximately "
    "#0c1e3e), and a single electric cobalt blue accent (approximately "
    "#2754d8). Crisp vector shapes, hairline strokes, no gradients, "
    "no shading, no rasterized textures, no photorealism. No people, "
    "no faces, no human figures. No text, no letters, no captions, "
    "no logos. Schematic and symbolic, like a margin sketch in a "
    "research notebook."
)

# Placeholder regex. Matches `{{IMG:OPENING}}` or `{{IMG:DIALECTIC_1}}`.
# The ID is captured for lookup against the briefs JSON. Uppercase
# letters, digits, underscores only — keeps the syntax noise-resistant
# in markdown (no spaces, no special chars).
PLACEHOLDER_RE = re.compile(r"\{\{IMG:([A-Z0-9_]+)\}\}")

# How many briefs we'll honor per post. The stage prompt asks for 2-4;
# this is the hard cap on the rendering side so a runaway model can't
# burn the cost ceiling on twenty illustrations.
MAX_INLINE_IMAGES = 4

# Default size + quality for inline. 1024x1024 standard balances cost
# vs visual quality at the inline image dimensions we'll display.
INLINE_SIZE = "1024x1024"
INLINE_QUALITY = "standard"


# ---------------------------------------------------------------------------
# OpenAI client (lazy import — same convention as generate_cover.py)
# ---------------------------------------------------------------------------


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
            "OPENAI_API_KEY environment variable is not set. Add it to your "
            "GitHub repo's Actions secrets."
        )
    return OpenAI(api_key=api_key)


def _build_dalle_prompt(brief: str) -> str:
    return f"{brief.rstrip('. ')}. {INLINE_STYLE_SUFFIX}"


def _download(url: str, target: Path) -> None:
    """Stream an image from `url` to `target`. Atomic-ish."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as f:
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(target)


# ---------------------------------------------------------------------------
# Brief parsing
# ---------------------------------------------------------------------------


_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first {...} JSON object out of `text`, fenced or not.

    Same tolerance pattern as scripts/select_papers.py — models
    sometimes wrap JSON in ```json fences or in a prose preamble despite
    being told not to.
    """
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
    """Parse the inline_image_briefs stage output into a clean list.

    Expected schema:
        {
          "images": [
            {"id": "OPENING",     "brief": "An abstract scene of...", "alt": "..."},
            {"id": "DIALECTIC_1", "brief": "...",                      "alt": "..."},
            ...
          ]
        }

    Tolerant of: missing `alt`, ids in lowercase, briefs with trailing
    periods, extra unknown keys.

    Returns at most `MAX_INLINE_IMAGES` entries with normalized keys.
    """
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


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_one(
    *,
    client: Any,
    brief: dict[str, str],
    out_dir: Path,
) -> str | None:
    """Render one inline illustration. Returns its public path, or None on failure."""
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

    # Cost meter — billed on response, same pattern as the cover stage.
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
        except Exception as e:  # pragma: no cover — defensive
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
    """Render every inline brief and return {id -> {public_path, alt}}.

    Failures are surfaced as missing entries in the returned dict so
    `substitute_placeholders` can decide what to leave behind in the MDX
    (a comment, an empty alt-text, etc).

    `skip=True` short-circuits — used for `--skip-cover` style runs and
    for tests that don't want to hit the OpenAI API.
    """
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


# ---------------------------------------------------------------------------
# Placeholder substitution
# ---------------------------------------------------------------------------


def substitute_placeholders(
    mdx: str,
    rendered: dict[str, dict[str, str]],
) -> tuple[str, int, int]:
    """Replace `{{IMG:ID}}` tokens in `mdx` with markdown image syntax.

    For each placeholder:
      - If `rendered[id]` exists, swap for `![alt](public_path)`.
        We wrap it in a paragraph break so the image renders on its
        own line in the prose flow (markdown would otherwise inline it).
      - If not, swap for an HTML comment so the gap is visible to a
        human reviewer but doesn't show up to readers.

    Returns: (new_mdx, replaced_count, missing_count).
    """
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
            # Surrounding blank lines force markdown to render the image
            # as its own block element rather than splicing it inline
            # with the preceding/following text.
            return f"\n\n![{alt}]({path})\n\n"
        missing += 1
        return f"<!-- IMG:{placeholder_id} not rendered -->"

    new_mdx = PLACEHOLDER_RE.sub(_sub, mdx)
    # Collapse any triple+ newlines we may have introduced down to 2.
    new_mdx = re.sub(r"\n{3,}", "\n\n", new_mdx)
    return new_mdx, replaced, missing


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_and_substitute(
    *,
    stream_cfg: Any,
    stage_outputs: dict[str, str],
    mdx_path: Path,
    slug: str,
    skip: bool = False,
) -> dict[str, Any]:
    """Top-level: render briefs, rewrite the MDX in place. Returns a summary.

    Wired into main.py after `write_post` succeeds. Failure here is
    non-fatal — the post ships with placeholders rewritten as comments,
    and the run summary records what didn't render.
    """
    inline_cfg = getattr(stream_cfg, "inline_images", None)
    enabled = getattr(inline_cfg, "enabled", True) if inline_cfg else True
    if not enabled or skip:
        # Even when disabled we sub the placeholders to comments so a
        # post that contains `{{IMG:OPENING}}` never ships visibly broken.
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
