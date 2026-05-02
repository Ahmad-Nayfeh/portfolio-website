"""scripts/validate_build.py

Run a Next.js production build against the current repo state. Used as a
quality gate after a stream has written its post: if `next build` fails on
the new MDX (e.g. unparseable JSX, missing import, KaTeX error), we flag the
PR rather than letting it auto-merge.

The function returns a structured result so the orchestrator can decide
whether to label the PR `build-failed` or proceed.

Importable: no top-level side effects.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _which_pkg_manager() -> tuple[str, list[str]]:
    """Pick a package manager command.

    Order of preference:
      pnpm > npm. We don't try yarn — the repo uses pnpm.
    Returns (label, [cmd, ...args_prefix]) where ...args_prefix is what to put
    before "build" / "next" etc.
    """
    if shutil.which("pnpm"):
        return "pnpm", ["pnpm"]
    if shutil.which("npm"):
        return "npm", ["npm", "run"]
    raise RuntimeError("Neither pnpm nor npm is on PATH; cannot run a build.")


def run_next_build(timeout_seconds: int = 600) -> dict[str, Any]:
    """Run `<pkg-manager> build` and capture the result.

    Returns: {
        'success': bool,
        'returncode': int,
        'stdout': str,
        'stderr': str,
        'duration_seconds': float,
        'pkg_manager': str,
    }
    """
    label, prefix = _which_pkg_manager()
    cmd = prefix + ["build"]
    log.info("Running %s in %s", " ".join(cmd), REPO_ROOT)

    env = os.environ.copy()
    # Cut down log noise; we only care about pass/fail + the failure tail.
    env.setdefault("CI", "1")
    env.setdefault("NEXT_TELEMETRY_DISABLED", "1")

    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
        duration = time.monotonic() - started
        return {
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "duration_seconds": duration,
            "pkg_manager": label,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "success": False,
            "returncode": 124,
            "stdout": e.stdout or "",
            "stderr": (e.stderr or "") + "\n[BUILD TIMEOUT]",
            "duration_seconds": time.monotonic() - started,
            "pkg_manager": label,
        }


def format_failure_excerpt(result: dict[str, Any], max_chars: int = 4000) -> str:
    """Return the most useful tail of stderr+stdout for a PR comment / issue."""
    blob = (result.get("stderr") or "") + "\n" + (result.get("stdout") or "")
    blob = blob.strip()
    if len(blob) <= max_chars:
        return blob
    return "...\n" + blob[-max_chars:]
