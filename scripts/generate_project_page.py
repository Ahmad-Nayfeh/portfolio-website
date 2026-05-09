"""
generate_project_page.py — Generate cover image + project markdown page.

Steps:
  1. Generate DALL-E 3 cover image (reuse blog's generate_cover.py logic)
  2. Copy validation figures from work_dir into portfolio's public/images/
  3. Produce content/projects/<slug>.md with embedded figures
"""

from __future__ import annotations

import logging
import shutil
from datetime import date
from pathlib import Path
from typing import Optional

import requests

from openai import OpenAI

logger = logging.getLogger(__name__)

# Reuse the Nocturne style suffix from the blog cover pipeline
NOCTURNE_STYLE_SUFFIX = (
    "Flat editorial vector illustration. Geometric and abstract. "
    "Minimal composition with generous negative space. "
    "Dark background (~#080812). "
    "Strict three-color palette: electric teal (#00d4aa), "
    "warm amber (#ffba08), and hot magenta (#ff3cac) accents."
)

COVER_BRIEF_SYSTEM = """You are generating a cover image brief for a project page.
Write ONE sentence (max 25 words) describing the abstract scene for the cover.
Be VISUAL: describe shapes, composition, lighting — not technical jargon.
Do NOT mention text, letters, or faces.
Output the sentence and nothing else."""


def _generate_cover_image(
    openai_client: OpenAI,
    project_name: str,
    description: str,
    slug: str,
    output_dir: Path,
) -> Optional[str]:
    """Generate a DALL-E 3 cover image. Returns relative path or None."""
    try:
        brief_resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": COVER_BRIEF_SYSTEM},
                {
                    "role": "user",
                    "content": f"Project: {project_name}. Description: {description}",
                },
            ],
            max_tokens=60,
        )
        brief = brief_resp.choices[0].message.content.strip()

        prompt = f"{brief} {NOCTURNE_STYLE_SUFFIX}"

        img_resp = openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1792x1024",
            quality="hd",
            n=1,
        )

        img_url = img_resp.data[0].url

        # Download and save
        cover_dir = output_dir / "public" / "images" / "projects"
        cover_dir.mkdir(parents=True, exist_ok=True)
        cover_path = cover_dir / f"{slug}.jpg"

        r = requests.get(img_url, timeout=60)
        r.raise_for_status()
        cover_path.write_bytes(r.content)

        relative = f"/images/projects/{slug}.jpg"
        logger.info("Cover saved: %s", relative)
        return relative

    except Exception as e:
        logger.warning("Cover generation failed (non-fatal): %s", e)
        return None


def _copy_figures_to_portfolio(
    work_dir: Path,
    slug: str,
    portfolio_root: Path,
) -> list[str]:
    """Copy validation figures from work_dir to portfolio/public/images/projects/<slug>/.
    Returns list of relative paths for the markdown page."""
    src_fig_dir = work_dir / "output"
    dst_dir = portfolio_root / "public" / "images" / "projects" / slug

    if not src_fig_dir.exists():
        logger.warning("No figures found in %s", src_fig_dir)
        return []

    dst_dir.mkdir(parents=True, exist_ok=True)
    relative_paths = []

    for f in sorted(src_fig_dir.iterdir()):
        if f.suffix.lower() in {".png", ".pdf", ".jpg", ".jpeg", ".svg"}:
            shutil.copy2(f, dst_dir / f.name)
            relative_paths.append(f"/images/projects/{slug}/{f.name}")

    logger.info("Copied %d figures to portfolio", len(relative_paths))
    return relative_paths


def generate_project_page(
    openai_client: Optional[OpenAI],
    project_name: str,
    slug: str,
    description: str,
    tags: list[str],
    repo_url: str,
    paper_title: str,
    arxiv_url: str,
    figures: list[str],
    demo_stdout: str,
    work_dir: Path,
    portfolio_root: Path,
) -> Path:
    """Generate the project markdown page. Returns path to the .md file."""

    # Generate cover image
    cover_path = None
    if openai_client:
        cover_path = _generate_cover_image(
            openai_client, project_name, description, slug, portfolio_root
        )

    # Copy figures into portfolio
    figure_urls = _copy_figures_to_portfolio(work_dir, slug, portfolio_root)

    # Build frontmatter
    tags_yaml = "\n".join(f'  - "{t}"' for t in tags)
    frontmatter = f"""---
title: "{project_name}"
slug: "{slug}"
date: "{date.today().isoformat()}"
coverImage: "{cover_path or ''}"
tags:
{tags_yaml}
excerpt: "{description[:200]}"
category: "AI Implementation"
githubLink: "{repo_url}"
featured: false
paperTitle: "{paper_title}"
paperUrl: "{arxiv_url}"
---
"""

    # Build body
    body_parts = [
        f'<div class="project-prose-container">\n',
        f"## Overview\n\n{description}\n\n",
        f"> This project was auto-generated from the paper: [{paper_title}]({arxiv_url})\n\n",
        f"> GitHub repository: [{repo_url}]({repo_url})\n\n",
    ]

    # Add figure gallery
    if figure_urls:
        body_parts.append("## Visual Results\n\n")
        body_parts.append('<div class="project-gallery">\n')
        for fig_url in figure_urls:
            body_parts.append(f'    <img src="{fig_url}" alt="{Path(fig_url).stem}" />\n')
        body_parts.append('</div>\n\n')

    # Add demo log (last 20 lines)
    if demo_stdout:
        lines = demo_stdout.strip().splitlines()
        tail = lines[-30:] if len(lines) > 30 else lines
        body_parts.append("## Demo Output\n\n```\n")
        body_parts.extend(line + "\n" for line in tail)
        body_parts.append("```\n\n")

    body_parts.append("## Repository\n\n")
    body_parts.append(f"The full source code is available at [{repo_url}]({repo_url}).\n")
    body_parts.append("Clone and run `python demo.py` to reproduce the results.\n")

    body_parts.append('\n</div>\n')

    body = "".join(body_parts)

    # Write the markdown file
    content_dir = portfolio_root / "content" / "projects"
    content_dir.mkdir(parents=True, exist_ok=True)
    md_path = content_dir / f"{slug}.md"
    md_path.write_text(frontmatter + body, encoding="utf-8")

    logger.info("Project page written: %s", md_path)
    return md_path
