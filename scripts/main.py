"""scripts/main.py

Pipeline entry point. Run via the GitHub Actions workflow (or locally for
testing):

    python scripts/main.py                    # all due streams
    python scripts/main.py --stream ai-papers # one specific stream
    python scripts/main.py --dry-run          # discovery + plan only, no API calls

Per stream, the flow is:

    load_streams.load_due_streams
        -> discover_papers.discover_for_stream
        -> generate_post.run_stages
        -> generate_post.regenerate_quote_stage   (if quote count is short)
        -> generate_post.write_post               (writes content/blog/<slug>.mdx)
        -> run_demo.run_demo_for_post             (optional, mutates the MDX)
        -> validate_build.run_next_build          (optional, gates the PR)

The pipeline writes its outputs into the working tree. The GitHub Actions
job is what creates the branch + PR after `main.py` exits (see
.github/workflows/publish.yml).

Exit codes:
    0 = at least one stream produced a post (or no streams were due)
    1 = a fatal error (bad config, network, model failure)
    2 = a stream's post failed a quality gate (build, quote count, ...)

The workflow turns exit code 2 into a labelled PR rather than a failed run,
so the post can still be reviewed even when something is off.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path

# Local imports — these are siblings in scripts/.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import discover_papers  # noqa: E402
import generate_post  # noqa: E402
import load_streams  # noqa: E402
import run_demo  # noqa: E402
import validate_build  # noqa: E402

log = logging.getLogger("publish")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publish pipeline orchestrator.")
    p.add_argument(
        "--stream",
        help="Run only this stream id (skips cron check). Useful for workflow_dispatch.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover papers and print the plan, but skip Claude API calls and writes.",
    )
    p.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip `next build` validation (orchestrator gate). The CI will still run its own build.",
    )
    p.add_argument(
        "--skip-demo",
        action="store_true",
        help="Skip running the embedded Python demo, even if the stream has demo.enabled.",
    )
    p.add_argument(
        "--output-summary",
        help="Write a JSON summary of what happened to this path (consumed by the workflow).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _select_streams(args: argparse.Namespace, now: dt.datetime) -> list:
    if args.stream:
        cfg = load_streams.get_stream_by_id(args.stream)
        if cfg is None:
            log.error("Stream %s not found in streams/", args.stream)
            sys.exit(1)
        if not cfg.stream.enabled:
            log.warning("Stream %s is disabled; running anyway because --stream was passed.", args.stream)
        return [cfg]
    return load_streams.load_due_streams(now=now)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    now = dt.datetime.utcnow()

    streams = _select_streams(args, now)
    if not streams:
        log.info("No streams due at %s UTC. Nothing to do.", now.isoformat(timespec="minutes"))
        _write_summary(args, [])
        return 0

    summaries: list[dict] = []
    worst_exit = 0

    for cfg in streams:
        log.info("=" * 70)
        log.info("Stream %s (%s)", cfg.stream.id, cfg.stream.name)

        summary: dict = {
            "stream_id": cfg.stream.id,
            "stream_name": cfg.stream.name,
            "started_at": dt.datetime.utcnow().isoformat() + "Z",
            "status": "pending",
        }

        # 1) Discover.
        try:
            papers = discover_papers.discover_for_stream(cfg)
        except Exception as e:
            log.exception("Discovery failed for %s: %s", cfg.stream.id, e)
            summary["status"] = "discovery_failed"
            summary["error"] = str(e)
            summaries.append(summary)
            worst_exit = max(worst_exit, 1)
            continue

        log.info("Discovered %d candidate papers", len(papers))
        summary["candidate_count"] = len(papers)
        summary["candidates"] = [
            {"title": p.get("title"), "arxiv_id": p.get("arxiv_id"), "upvotes": p.get("upvotes", 0)}
            for p in papers
        ]

        if not papers and cfg.discovery.source not in {"none", "manual"}:
            log.warning("No candidates found for %s; skipping post.", cfg.stream.id)
            summary["status"] = "no_candidates"
            summaries.append(summary)
            continue

        # Apply selection.count (if set) — keep the top N for the post.
        sel = (cfg.discovery.selection or {}).get("count")
        if sel and isinstance(sel, int):
            papers = papers[:sel]
        summary["selected_count"] = len(papers)

        if args.dry_run:
            log.info("[dry-run] Would generate post from %d papers; stopping here.", len(papers))
            summary["status"] = "dry_run"
            summaries.append(summary)
            continue

        # 2) Generate.
        try:
            stage_outputs = generate_post.run_stages(stream_cfg=cfg, papers=papers)
        except Exception as e:
            log.exception("Generation failed for %s: %s", cfg.stream.id, e)
            summary["status"] = "generation_failed"
            summary["error"] = str(e)
            summaries.append(summary)
            worst_exit = max(worst_exit, 1)
            continue

        # 2b) Quote-count retry. We assemble the body (without writing it),
        # count quotes, and re-run the quote stage once if we're below target.
        # Retrying BEFORE writing to disk keeps the published post clean if
        # the retry succeeds — no rewrite, no second commit.
        required_quotes = cfg.quality_gates.require_verbatim_quotes or 0
        retried_quotes = False
        if required_quotes > 0:
            preview_body = "\n\n".join(
                stage_outputs.get(s.get("name", ""), "").strip()
                for s in (cfg.generation.stages or [])
                if stage_outputs.get(s.get("name", ""), "").strip()
            )
            preview_count = generate_post.count_verbatim_quotes(preview_body)
            if preview_count < required_quotes:
                log.info(
                    "Quote count %d < %d; retrying quote stage once.",
                    preview_count, required_quotes,
                )
                try:
                    stage_outputs = generate_post.regenerate_quote_stage(
                        stream_cfg=cfg, papers=papers, stage_outputs=stage_outputs,
                        current_count=preview_count, target_count=required_quotes,
                    )
                    retried_quotes = True
                except Exception as e:
                    log.warning("Quote retry crashed: %s — proceeding with original output.", e)
        summary["quote_retry_used"] = retried_quotes

        # 3) Write MDX.
        try:
            mdx_path = generate_post.write_post(
                stream_cfg=cfg, papers=papers, stage_outputs=stage_outputs, now=now,
            )
        except Exception as e:
            log.exception("Writing MDX failed for %s: %s", cfg.stream.id, e)
            summary["status"] = "write_failed"
            summary["error"] = str(e)
            summaries.append(summary)
            worst_exit = max(worst_exit, 1)
            continue

        slug = mdx_path.stem
        summary["slug"] = slug
        summary["mdx_path"] = str(mdx_path.relative_to(load_streams.REPO_ROOT))

        # 3b) Quality gate: verbatim quotes (final check after retry).
        quote_count = generate_post.count_verbatim_quotes(mdx_path.read_text(encoding="utf-8"))
        summary["quote_count"] = quote_count
        summary["quote_count_required"] = required_quotes
        if required_quotes and quote_count < required_quotes:
            log.warning(
                "Stream %s: post has %d verbatim quotes, %d required. PR will be flagged.",
                cfg.stream.id, quote_count, required_quotes,
            )
            summary["quote_check_failed"] = True
            worst_exit = max(worst_exit, 2)
        else:
            summary["quote_check_failed"] = False

        # 4) Demo.
        if args.skip_demo:
            log.info("Skipping demo (--skip-demo)")
            summary["demo"] = {"ran": False, "reason": "skipped"}
        else:
            try:
                demo_result = run_demo.run_demo_for_post(stream_cfg=cfg, mdx_path=mdx_path, slug=slug)
            except Exception as e:
                log.exception("Demo step crashed: %s", e)
                demo_result = {"ran": True, "success": False, "figures": [], "reason": "crashed", "error": str(e)}
            # Don't pickle Path objects into JSON.
            summary["demo"] = {
                "ran": demo_result.get("ran", False),
                "success": demo_result.get("success", False),
                "reason": demo_result.get("reason"),
                "figure_count": len(demo_result.get("figures", []) or []),
            }

        # 5) Build validation.
        if args.skip_build or not cfg.quality_gates.build_validation:
            log.info("Skipping next-build validation")
            summary["build"] = {"ran": False, "reason": "skipped"}
        else:
            build_result = validate_build.run_next_build()
            summary["build"] = {
                "ran": True,
                "success": build_result["success"],
                "returncode": build_result["returncode"],
                "duration_seconds": round(build_result["duration_seconds"], 1),
            }
            if not build_result["success"]:
                log.warning("Stream %s: next build failed; PR will be flagged.", cfg.stream.id)
                summary["build_failure_excerpt"] = validate_build.format_failure_excerpt(build_result)
                worst_exit = max(worst_exit, 2)

        if summary.get("status") == "pending":
            summary["status"] = "ok" if worst_exit == 0 else "flagged"
        summary["finished_at"] = dt.datetime.utcnow().isoformat() + "Z"
        summaries.append(summary)

    _write_summary(args, summaries)
    return worst_exit


def _write_summary(args: argparse.Namespace, summaries: list[dict]) -> None:
    if not args.output_summary:
        return
    out = Path(args.output_summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    log.info("Wrote run summary to %s", out)


if __name__ == "__main__":
    sys.exit(main())
