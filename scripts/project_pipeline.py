"""
project_pipeline.py — Main orchestrator for the project automation pipeline.

Flow:
  1. Discover papers from Hugging Face Daily Papers
  2. For each paper (sorted by upvotes), check feasibility
  3. First feasible paper → generate project code
  4. Validate: run demo.py, collect figures
  5. Create GitHub repo with the code
  6. Generate project page (cover + markdown)
  7. Print run summary for the GitHub Action to consume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from anthropic import Anthropic
from openai import OpenAI

from cost_meter import CostMeter
from discover_papers import fetch_huggingface_daily
from generate_project_code import check_feasibility

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("project_pipeline")


def main():
    parser = argparse.ArgumentParser(description="Project automation pipeline")
    parser.add_argument("--output-summary", default="project-run-summary.json")
    parser.add_argument("--dry-run", action="store_true", help="Discover + plan only, no generation")
    parser.add_argument("--paper-url", help="Override: specific paper URL to implement")
    args = parser.parse_args()

    portfolio_root = Path(__file__).resolve().parent.parent
    work_dir = portfolio_root / ".project-work"
    work_dir.mkdir(parents=True, exist_ok=True)

    cost_meter = CostMeter(cost_ceiling_usd=2.00)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not anthropic_key:
        logger.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    anthropic_client = Anthropic(api_key=anthropic_key)
    openai_client = OpenAI(api_key=openai_key) if openai_key else None

    # ── Stage 1: Discover papers ──────────────────────────────────────────
    candidates = []
    if args.paper_url:
        logger.info("Specific paper URL provided: %s", args.paper_url)
        # We'll build a single-entry candidate from the URL.
        # In practice the workflow_dispatch passes a URL that main.py
        # would need to resolve. For now, skip discovery and bail.
        # TODO: resolve paper URL to paper info
        candidates = []
    else:
        logger.info("Discovering papers from Hugging Face Daily Papers...")
        raw = fetch_huggingface_daily(lookback_days=14)
        # Deduplicate against used-projects.json
        used_path = Path(__file__).resolve().parent.parent / "pipeline-state" / "used-projects.json"
        used_ids = set()
        if used_path.exists():
            import json as _json
            used_ids = {e.get("arxiv_id") for e in _json.loads(used_path.read_text())}
        for p in raw:
            arxiv_id = p.get("paper", {}).get("id", "")
            if arxiv_id and arxiv_id not in used_ids:
                candidates.append({
                    "title": p.get("paper", {}).get("title", ""),
                    "abstract": p.get("paper", {}).get("summary", ""),
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                    "arxiv_id": arxiv_id,
                    "upvotes": p.get("upvotes", 0),
                })
        candidates.sort(key=lambda c: c.get("upvotes", 0), reverse=True)
        candidates = candidates[:20]
        logger.info("Found %d candidates after dedup", len(candidates))

    if not candidates and not args.paper_url:
        logger.info("No candidates found. Nothing to do.")
        _write_summary(args.output_summary, [{"status": "no_candidates"}])
        return

    # ── Stage 2: Check feasibility ────────────────────────────────────────
    selected = None
    for paper in candidates:
        logger.info("Checking: %s", paper.get("title", "Unknown"))
        try:
            result = check_feasibility(
                anthropic_client,
                title=paper.get("title", ""),
                abstract=paper.get("abstract", ""),
                arxiv_url=paper.get("url", ""),
                cost_meter=cost_meter,
            )
            if result.feasible:
                selected = {**paper, "feasibility": result}
                logger.info("SELECTED: %s (difficulty=%s)", paper["title"], result.difficulty)
                break
            else:
                logger.info("Not feasible: %s", result.reasoning)
        except Exception as e:
            logger.warning("Feasibility check failed: %s", e)
            continue

    if not selected and args.dry_run:
        logger.info("Dry run: no feasible paper found")
        _write_summary(args.output_summary, [{"status": "no_feasible_paper", "candidates_checked": len(candidates)}])
        return

    if not selected:
        logger.info("No feasible paper found among %d candidates", len(candidates))
        _write_summary(args.output_summary, [{"status": "no_feasible_paper", "candidates_checked": len(candidates)}])
        return

    # ── Stage 3: Generate code (skip if dry run) ──────────────────────────
    if args.dry_run:
        logger.info("Dry run — would generate project: %s", selected["feasibility"].suggested_project_name)
        summary = [{
            "status": "dry_run",
            "project_name": selected["feasibility"].suggested_project_name,
            "paper_title": selected["title"],
            "arxiv_url": selected.get("url"),
            "tags": selected["feasibility"].suggested_tags,
            "difficulty": selected["feasibility"].difficulty,
            "estimated_figures": selected["feasibility"].estimated_figures,
        }]
        _write_summary(args.output_summary, summary)
        return

    logger.info("Generating project code for: %s", selected["title"])

    from generate_project_code import run as generate_project
    from validate_project import validate
    from repo_setup import create_repo
    from generate_project_page import generate_project_page

    project = generate_project(
        anthropic_client,
        title=selected["title"],
        abstract=selected.get("abstract", ""),
        arxiv_url=selected.get("url", ""),
        cost_meter=cost_meter,
    )

    if not project:
        logger.error("Code generation failed")
        _write_summary(args.output_summary, [{"status": "generation_failed", "paper": selected["title"]}])
        sys.exit(1)

    # ── Stage 4: Validate ─────────────────────────────────────────────────
    logger.info("Validating project: %s", project.project_name)
    validation = validate(
        files=project.files,
        work_dir=work_dir / project.project_name,
    )

    if not validation["success"]:
        logger.error("Validation failed: %s", validation.get("error"))
        _write_summary(args.output_summary, [{
            "status": "validation_failed",
            "project_name": project.project_name,
            "error": validation.get("error"),
            "figure_count": validation["figure_count"],
        }])
        sys.exit(1)

    # ── Stage 5: Create GitHub repo ───────────────────────────────────────
    logger.info("Creating GitHub repo: %s", project.project_name)
    repo_result = create_repo(
        project_name=project.project_name,
        description=project.description,
        tags=project.tags,
        files=project.files,
        work_dir=work_dir,
    )

    if not repo_result["success"]:
        logger.error("Repo creation failed: %s", repo_result.get("error"))
        _write_summary(args.output_summary, [{
            "status": "repo_failed",
            "project_name": project.project_name,
            "error": repo_result.get("error"),
        }])
        sys.exit(1)

    # ── Stage 6: Generate project page ────────────────────────────────────
    logger.info("Generating project page...")
    md_path = generate_project_page(
        openai_client=openai_client,
        project_name=project.project_name,
        slug=project.slug,
        description=project.description,
        tags=project.tags,
        repo_url=repo_result["repo_url"],
        paper_title=project.paper_title,
        arxiv_url=project.arxiv_url,
        figures=validation["figures"],
        demo_stdout=validation["stdout"],
        work_dir=work_dir / project.project_name,
        portfolio_root=portfolio_root,
    )

    # ── Write summary ─────────────────────────────────────────────────────
    summary = [{
        "status": "ok",
        "project_name": project.project_name,
        "slug": project.slug,
        "paper_title": project.paper_title,
        "arxiv_url": project.arxiv_url,
        "repo_url": repo_result["repo_url"],
        "tags": project.tags,
        "figure_count": validation["figure_count"],
        "duration_seconds": validation["duration_seconds"],
        "cost_usd": cost_meter.total_cost,
        "md_path": str(md_path),
    }]

    _write_summary(args.output_summary, summary)
    logger.info("Done. Project: %s", repo_result["repo_url"])


def _write_summary(path: str, data: list[dict]):
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info("Summary written to %s", path)


if __name__ == "__main__":
    main()
