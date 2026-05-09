"""
validate_project.py — Run the generated project code and capture outputs.

Steps:
  1. Create temp directory
  2. Write all project files into it
  3. Install dependencies
  4. Run demo.py with a timeout
  5. Collect all generated figures
  6. Return figure paths + run log
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TIMEOUT_MINUTES = 10


def validate(
    files: dict[str, str],
    work_dir: Path,
) -> dict:
    """
    Write files, install deps, run demo.py, collect outputs.

    Returns:
        {
            "success": bool,
            "figures": ["output/figure1.png", ...],
            "figure_count": int,
            "stdout": str,
            "stderr": str,
            "error": str | None,
            "duration_seconds": float,
        }
    """
    result: dict = {
        "success": False,
        "figures": [],
        "figure_count": 0,
        "stdout": "",
        "stderr": "",
        "error": None,
        "duration_seconds": 0.0,
    }

    # Wipe and recreate work_dir
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Write all files
    for rel_path, content in files.items():
        full_path = work_dir / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    # Install dependencies
    req_path = work_dir / "requirements.txt"
    if req_path.exists():
        logger.info("Installing dependencies...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req_path)],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            result["error"] = f"Dependency installation failed: {e.stderr[:500]}"
            result["stderr"] = e.stderr
            return result
        except subprocess.TimeoutExpired:
            result["error"] = "Dependency installation timed out (120s)"
            return result

    # Run demo.py
    demo_path = work_dir / "demo.py"
    if not demo_path.exists():
        result["error"] = "demo.py not found"
        return result

    logger.info("Running demo.py...")
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(demo_path)],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_MINUTES * 60,
        )
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr

        if proc.returncode != 0:
            result["error"] = (
                f"demo.py exited with code {proc.returncode}. "
                f"Stderr: {proc.stderr[:500]}"
            )
            return result
    except subprocess.TimeoutExpired:
        result["error"] = f"demo.py timed out after {TIMEOUT_MINUTES} minutes"
        return result
    finally:
        result["duration_seconds"] = time.time() - start

    # Collect figures
    output_dir = work_dir / "output"
    if output_dir.exists():
        for f in sorted(output_dir.iterdir()):
            if f.suffix.lower() in {".png", ".pdf", ".jpg", ".jpeg", ".svg"}:
                result["figures"].append(str(f.relative_to(work_dir)))

    result["figure_count"] = len(result["figures"])
    result["success"] = result["figure_count"] >= 4

    if result["figure_count"] < 4:
        result["error"] = (
            f"Only {result['figure_count']} figures produced (need at least 4)"
        )

    logger.info(
        "Validation: %s (%d figures in %.1fs)",
        "PASS" if result["success"] else "FAIL",
        result["figure_count"],
        result["duration_seconds"],
    )

    return result
