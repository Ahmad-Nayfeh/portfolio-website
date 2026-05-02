"""scripts/run_demo.py

Extract the Python demo from a generated MDX post, run it in a subprocess
under a timeout, capture any matplotlib figures it produces, and rewrite the
demo section in the MDX to reference the rendered images.

The demo block is identified by a fenced code block tagged ```python or
```python demo. The first such block in the post is treated as the demo.

Outputs:
  - public/blog-images/<slug>/figure_*.png  (any figures the demo saved)
  - The MDX is updated to insert a markdown image link per figure right
    below the demo's code block.

On failure (timeout, non-zero exit), the demo section is either kept (with a
warning) or stripped from the MDX, depending on `stream_cfg.demo.on_failure`.

Importable: no top-level side effects.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Match a fenced python block. We capture the body so we can rewrite the
# original block with figures appended underneath.
_PY_BLOCK_RE = re.compile(
    r"^```python(?:[^\n]*)\n([\s\S]*?)\n```",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Demo extraction + rewriting
# ---------------------------------------------------------------------------


def extract_first_python_block(mdx: str) -> tuple[str, int, int] | None:
    """Find the first ```python ... ``` block. Return (code, start, end) or None."""
    m = _PY_BLOCK_RE.search(mdx)
    if not m:
        return None
    return m.group(1), m.start(), m.end()


def _wrap_demo_for_capture(code: str, out_dir: Path) -> str:
    """Wrap user demo code so matplotlib figures are auto-saved.

    The wrapper:
      1. Forces a non-interactive backend so the demo can't open a GUI window.
      2. After the demo runs, saves every open figure to figure_<n>.png.

    We intentionally append rather than transform the user's code so we don't
    silently rewrite something the model might rely on (e.g. plt.show()).
    """
    header = (
        "import os, sys\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as _plt_demo_capture\n"
        "\n"
    )
    footer = (
        "\n"
        "import matplotlib.pyplot as _plt\n"
        f"_out_dir = r'''{out_dir}'''\n"
        "os.makedirs(_out_dir, exist_ok=True)\n"
        "for _i, _num in enumerate(_plt.get_fignums(), start=1):\n"
        "    _fig = _plt.figure(_num)\n"
        "    _path = os.path.join(_out_dir, f'figure_{_i}.png')\n"
        "    _fig.savefig(_path, dpi=120, bbox_inches='tight')\n"
        "    print(f'SAVED_FIGURE:{_path}')\n"
    )
    return header + code + footer


def _run_subprocess(code: str, timeout_seconds: int) -> tuple[int, str, str]:
    """Run `code` in a fresh Python subprocess with a hard timeout."""
    with tempfile.TemporaryDirectory(prefix="demo_") as tmp:
        script = Path(tmp) / "demo.py"
        script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            return 124, e.stdout or "", (e.stderr or "") + "\n[TIMEOUT]"
        return proc.returncode, proc.stdout, proc.stderr


def _images_section(image_paths: list[Path], slug: str) -> str:
    if not image_paths:
        return ""
    lines = ["", "**Generated figures:**", ""]
    for p in image_paths:
        # Public path used by Next.js: /blog-images/<slug>/<file>
        public_url = f"/blog-images/{slug}/{p.name}"
        lines.append(f"![Demo figure]({public_url})")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run_demo_for_post(
    *,
    stream_cfg: Any,
    mdx_path: Path,
    slug: str,
) -> dict[str, Any]:
    """Run the demo embedded in `mdx_path`. Mutate the file in-place.

    Returns a dict describing what happened: {
        'ran': bool,
        'success': bool,
        'figures': [Path, ...],
        'stdout': str,
        'stderr': str,
        'reason': str | None,
    }
    """
    if not stream_cfg.demo.enabled:
        log.info("Demo disabled for stream %s; skipping", stream_cfg.stream.id)
        return {"ran": False, "success": False, "figures": [], "stdout": "", "stderr": "", "reason": "disabled"}

    mdx = mdx_path.read_text(encoding="utf-8")
    block = extract_first_python_block(mdx)
    if block is None:
        log.info("No python demo block found in %s; skipping", mdx_path.name)
        return {"ran": False, "success": False, "figures": [], "stdout": "", "stderr": "", "reason": "no_block"}
    code, _, end = block

    out_dir_template = stream_cfg.demo.output_dir or "public/blog-images/{slug}"
    out_dir = REPO_ROOT / out_dir_template.replace("{slug}", slug)
    out_dir.mkdir(parents=True, exist_ok=True)

    timeout_seconds = (stream_cfg.demo.timeout_minutes or 15) * 60
    wrapped = _wrap_demo_for_capture(code, out_dir)
    rc, stdout, stderr = _run_subprocess(wrapped, timeout_seconds)
    log.info("Demo subprocess rc=%d stdout=%d stderr=%d", rc, len(stdout), len(stderr))

    figures = sorted(out_dir.glob("figure_*.png"))

    if rc != 0:
        log.warning("Demo failed (rc=%d). Policy: %s", rc, stream_cfg.demo.on_failure)
        # Clean up partial figures so a half-rendered demo doesn't leak into the post.
        if figures:
            for f in figures:
                f.unlink(missing_ok=True)
            figures = []
        if stream_cfg.demo.on_failure == "strip_demo_section":
            new_mdx = _strip_demo_block(mdx)
            mdx_path.write_text(new_mdx, encoding="utf-8")
        # else: leave the post untouched.
        return {
            "ran": True,
            "success": False,
            "figures": [],
            "stdout": stdout,
            "stderr": stderr,
            "reason": "nonzero_exit" if rc != 124 else "timeout",
        }

    # Success: insert image links right after the demo block.
    if figures:
        insertion = _images_section(figures, slug)
        new_mdx = mdx[:end] + "\n" + insertion + mdx[end:]
        mdx_path.write_text(new_mdx, encoding="utf-8")

    return {
        "ran": True,
        "success": True,
        "figures": figures,
        "stdout": stdout,
        "stderr": stderr,
        "reason": None,
    }


def _strip_demo_block(mdx: str) -> str:
    """Remove the first python fenced block + a one-line note explaining why."""
    m = _PY_BLOCK_RE.search(mdx)
    if not m:
        return mdx
    note = "\n> _The runnable demo for this post failed in CI and has been removed. The text above still describes the method._\n"
    return mdx[: m.start()] + note + mdx[m.end() :]


# ---------------------------------------------------------------------------
# Sanity helper for manual runs
# ---------------------------------------------------------------------------


def have_python_runtime() -> bool:
    """Quick check used by the orchestrator to skip the demo on hosts without Python."""
    return shutil.which(sys.executable) is not None
