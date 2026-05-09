"""
project_pipeline.py — Main orchestrator for the project automation pipeline.

Flow:
  1. Discover papers from Hugging Face Daily Papers (14-day lookback)
  2. For each paper (sorted by upvotes), check feasibility via Claude
  3. If no HF paper is feasible, fall back to arXiv (diverse categories, 30-day lookback)
  4. First feasible paper → generate project code
  5. Validate: run demo.py, collect figures
  6. Create GitHub repo with the code
  7. Generate project page (cover + markdown)
  8. Print run summary for the GitHub Action to consume
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path

from anthropic import Anthropic
from openai import OpenAI

import cost_meter
from discover_papers import fetch_arxiv, fetch_huggingface_daily
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

    cost_meter.init_meter(ceiling_usd=2.00)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not anthropic_key:
        logger.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    anthropic_client = Anthropic(api_key=anthropic_key)
    openai_client = OpenAI(api_key=openai_key) if openai_key else None

    # ── Stage 1: Discover papers ──────────────────────────────────────────
    candidates = []
    used_path = portfolio_root / "pipeline-state" / "used-projects.json"
    used_ids: set[str] = set()
    if used_path.exists():
        used_ids = _load_used_project_ids(used_path)
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
        for p in raw:
            arxiv_id = p.get("arxiv_id", "")
            if arxiv_id and arxiv_id not in used_ids:
                candidates.append({
                    "title": p.get("title", ""),
                    "abstract": p.get("summary", ""),
                    "url": p.get("url", ""),
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
        logger.info("No feasible paper from HF Daily Papers (%d checked). Trying arXiv fallback...", len(candidates))
        arxiv_categories = [
            # Signal processing — filtering, spectral analysis, compression
            "eess.SP",
            # Audio/speech processing — features, filtering, separation
            "eess.AS",
            # Statistical ML — kernel methods, Gaussian processes, Bayesian methods
            "stat.ML",
            # Optimization — convex, non-convex, gradient methods, submodular
            "math.OC",
            # Information theory — coding, compression, rate-distortion
            "cs.IT",
            # Numerical analysis — linear algebra, integration, approximation
            "cs.NA", "math.NA",
            # Computational statistics — MCMC, bootstrap, EM
            "stat.CO",
            # Statistical methodology — regression, testing, estimation
            "stat.ME",
            # Machine learning — theory papers, classical methods
            "cs.LG",
        ]
        arxiv_candidates = fetch_arxiv(
            categories=arxiv_categories,
            lookback_days=30,
            max_results=50,
        )
        # Deduplicate against used-projects.json
        arxiv_added: list[dict] = []
        for p in arxiv_candidates:
            arxiv_id = p.get("arxiv_id", "")
            if arxiv_id and arxiv_id not in used_ids:
                entry = {
                    "title": p.get("title", ""),
                    "abstract": p.get("summary", ""),
                    "url": p.get("url", ""),
                    "arxiv_id": arxiv_id,
                    "upvotes": 0,
                }
                candidates.append(entry)
                arxiv_added.append(entry)
        logger.info("arXiv fallback: %d additional candidates", len(arxiv_added))

        for paper in arxiv_added:
            if selected:
                break
            logger.info("Checking arXiv: %s", paper.get("title", "Unknown"))
            try:
                result = check_feasibility(
                    anthropic_client,
                    title=paper.get("title", ""),
                    abstract=paper.get("abstract", ""),
                    arxiv_url=paper.get("url", ""),
                )
                if result.feasible:
                    selected = {**paper, "feasibility": result}
                    logger.info("SELECTED (arXiv): %s (difficulty=%s)", paper["title"], result.difficulty)
                    break
                else:
                    logger.info("Not feasible: %s", result.reasoning)
            except Exception as e:
                logger.warning("Feasibility check failed: %s", e)
                continue

    if not selected:
        logger.info("No feasible paper found across all sources")
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

    # Add generated figures to the project files so they get committed to the repo
    output_dir = work_dir / project.project_name / "output"
    if output_dir.exists():
        for fig_path in sorted(output_dir.iterdir()):
            if fig_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
                rel = f"output/{fig_path.name}"
                project.files[rel] = fig_path.read_bytes()

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

    # ── Record the paper as used so it won't be picked again ──────────────
    _mark_project_used(
        path=used_path,
        arxiv_id=selected.get("arxiv_id", ""),
        title=selected.get("title", ""),
        url=selected.get("url", ""),
        slug=project.slug,
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
        "cost_usd": cost_meter.get_meter().total_usd if cost_meter.get_meter() else 0,
        "md_path": str(md_path),
    }]

    _write_summary(args.output_summary, summary)
    logger.info("Done. Project: %s", repo_result["repo_url"])


def _write_summary(path: str, data: list[dict]):
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info("Summary written to %s", path)


def _load_used_project_ids(path: Path) -> set[str]:
    """Read used-projects.json, handling both list and dict formats."""
    if not path.exists():
        return set()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not parse %s (%s); treating as empty.", path, e)
        return set()
    if isinstance(raw, dict):
        return {e.get("arxiv_id", "") for e in raw.get("papers", []) if isinstance(e, dict)}
    if isinstance(raw, list):
        return {e.get("arxiv_id", "") for e in raw if isinstance(e, dict)}
    return set()


def _mark_project_used(path: Path, arxiv_id: str, title: str, url: str, slug: str):
    """Append a paper to used-projects.json so it won't be picked again."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {"schema_version": 1, "papers": []}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                # Migrate old list format to dict format
                data["papers"] = existing
            elif isinstance(existing, dict):
                data = existing
        except Exception:
            pass
    # Check if already recorded
    for entry in data.get("papers", []):
        if isinstance(entry, dict) and entry.get("arxiv_id", "") == arxiv_id:
            logger.info("Paper %s already in used-projects.json", arxiv_id)
            return
    data["papers"].append({
        "arxiv_id": arxiv_id,
        "title": title,
        "url": url,
        "slug": slug,
        "added_at": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    })
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    logger.info("Recorded %s in %s", arxiv_id, path)


if __name__ == "__main__":
    main()
