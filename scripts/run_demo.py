"""scripts/run_demo.py

Extract Python demo blocks from a generated MDX post, run each in its own
subprocess under a timeout, capture the matplotlib figures each block
produces, and rewrite the MDX in place to reference the rendered images and
captured stdout.

H5 extension (May 2026): supports MULTIPLE python blocks per post. The
closing_demo_and_critique stage has the main big experimental block; the
dialectical_walk stage may now include up to 3 shorter (<=40 line) blocks
at natural beats in the argument. Each block runs in its own subdir of the
per-post image folder so figure names never collide (demo_0/, demo_1/, ...).
MDX rewriting proceeds from the last block to the first so that inserting
content at position `end` never shifts the offsets of earlier blocks.

Demo output capture design:
  The subprocess CWD is set to the block's own subdir. ANY PNG the demo
  writes -- via plt.savefig() or our autosave footer -- lands there. We
  sweep the subdir for *.png after the subprocess exits and inject them
  all under the code block.

On failure (timeout, non-zero exit): the failed block is replaced with a
short warning note. Other blocks still run.
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

# Match any fenced python block. Capture the body for execution + position
# tracking.
_PY_BLOCK_RE = re.compile(
    r"^```python(?:[^\n]*)\n([\s\S]*?)\n```",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Block extraction
# ---------------------------------------------------------------------------


def extract_all_python_blocks(mdx: str) -> list[tuple[str, int, int]]:
    """Find all ```python ... ``` blocks. Return [(code, start, end), ...].

    Ordered by position in the document (earliest first). The caller uses
    this ordering to run blocks in document order, then rewrites the MDX
    from last to first to preserve byte offsets.
    """
    return [(m.group(1), m.start(), m.end()) for m in _PY_BLOCK_RE.finditer(mdx)]


def extract_first_python_block(mdx: str) -> tuple[str, int, int] | None:
    """Compatibility alias -- returns only the first block, or None."""
    blocks = extract_all_python_blocks(mdx)
    return blocks[0] if blocks else None


# ---------------------------------------------------------------------------
# Demo wrapper
# ---------------------------------------------------------------------------


def _read_plot_style_source() -> str:
    """Read scripts/plot_style.py and return its source text.

    We inline the module body into the demo wrapper rather than rely on
    PYTHONPATH because the subprocess CWD is the per-block image subdir,
    not scripts/. Inlining is heavier but bulletproof.
    """
    style_path = Path(__file__).resolve().parent / "plot_style.py"
    try:
        return style_path.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("plot_style.py not readable (%s); demos will run with bare matplotlib.", e)
        return ""


def _wrap_demo_for_capture(code: str) -> str:
    """Wrap user demo code so matplotlib figures are auto-saved AND styled.

    The wrapper:
      1. Forces a non-interactive backend.
      2. Inlines scripts/plot_style.py BEFORE the demo runs, exposing
         PALETTE / figure() / lead_color() / annotate_callout() in scope.
      3. After the demo runs, saves any still-open figures as figure_<n>.png
         in the current working directory (the per-block subdir).
    """
    plot_style = _read_plot_style_source()
    header = (
        "import os, sys\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as _plt_demo_capture\n"
        "\n"
        "# --- Editorial style (inlined from scripts/plot_style.py) ----\n"
        + plot_style
        + "\n"
        "# --- End editorial style. Demo code follows. -----------------\n"
        "\n"
    )
    footer = (
        "\n"
        "# --- Auto-capture: save any figures still open. -------------\n"
        "import matplotlib.pyplot as _plt\n"
        "for _i, _num in enumerate(_plt.get_fignums(), start=1):\n"
        "    _fig = _plt.figure(_num)\n"
        "    _path = f'figure_{_i}.png'\n"
        "    _fig.savefig(_path, dpi=200, bbox_inches='tight')\n"
        "    print(f'SAVED_FIGURE:{_path}')\n"
    )
    return header + code + footer


def _run_subprocess(
    code: str,
    timeout_seconds: int,
    cwd: Path,
) -> tuple[int, str, str]:
    """Run `code` in a fresh Python subprocess with a hard timeout.

    CWD is set to the per-block image subdir so figures land there.
    """
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
                cwd=str(cwd),
            )
        except subprocess.TimeoutExpired as e:
            return 124, e.stdout or "", (e.stderr or "") + "\n[TIMEOUT]"
        return proc.returncode, proc.stdout, proc.stderr


def _clean_stdout_for_display(stdout: str) -> str:
    """Strip internal SAVED_FIGURE: marker lines from stdout."""
    keep = [
        ln for ln in stdout.splitlines()
        if not ln.strip().startswith("SAVED_FIGURE:")
    ]
    while keep and not keep[-1].strip():
        keep.pop()
    return "\n".join(keep)


def _build_output_section(
    image_paths: list[Path],
    cleaned_stdout: str,
    slug: str,
    subdir_name: str,
) -> str:
    """Compose the markdown injected after a code block.

    Images are served from /blog-images/<slug>/<subdir>/<file>.
    Returns empty string if there is nothing to show.
    """
    if not image_paths and not cleaned_stdout.strip():
        return ""

    lines: list[str] = ["", "**Demo output**", ""]
    for p in image_paths:
        public_url = f"/blog-images/{slug}/{subdir_name}/{p.name}"
        lines.append(f"![Generated figure]({public_url})")
        lines.append("")
    if cleaned_stdout.strip():
        lines.append("```text")
        lines.append(cleaned_stdout.rstrip())
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_demo_for_post(
    *,
    stream_cfg: Any,
    mdx_path: Path,
    slug: str,
) -> dict[str, Any]:
    """Run every Python demo block in `mdx_path`. Mutate the file in-place.

    Each block runs in its own subdir (demo_0/, demo_1/, ...) of the
    per-post image folder so figure filenames never collide.

    Returns: {
        'ran': bool,
        'success': bool,         # True if ALL blocks succeeded
        'figures': [Path, ...],  # all PNGs produced across all blocks
        'stdout': str,           # concatenated stdout from all blocks
        'stderr': str,
        'reason': str | None,
    }
    """
    if not stream_cfg.demo.enabled:
        log.info("Demo disabled for stream %s; skipping", stream_cfg.stream.id)
        return {"ran": False, "success": False, "figures": [], "stdout": "", "stderr": "", "reason": "disabled"}

    mdx = mdx_path.read_text(encoding="utf-8")
    blocks = extract_all_python_blocks(mdx)
    if not blocks:
        log.info("No python demo blocks found in %s; skipping", mdx_path.name)
        return {"ran": False, "success": False, "figures": [], "stdout": "", "stderr": "", "reason": "no_block"}

    log.info("Found %d python block(s) in %s", len(blocks), mdx_path.name)

    out_dir_template = stream_cfg.demo.output_dir or "public/blog-images/{slug}"
    out_dir = REPO_ROOT / out_dir_template.replace("{slug}", slug)
    out_dir.mkdir(parents=True, exist_ok=True)

    timeout_seconds = (stream_cfg.demo.timeout_minutes or 15) * 60

    # Run every block in document order, capturing results. We separate
    # execution from MDX rewriting so that later rewriting (which modifies
    # string offsets) doesn't interfere with block position tracking.
    block_results: list[dict] = []
    for i, (code, _start, _end) in enumerate(blocks):
        subdir_name = f"demo_{i}"
        subdir = out_dir / subdir_name
        subdir.mkdir(parents=True, exist_ok=True)

        wrapped = _wrap_demo_for_capture(code)
        rc, stdout, stderr = _run_subprocess(wrapped, timeout_seconds, cwd=subdir)
        log.info(
            "Block %d/%d: rc=%d stdout=%d stderr=%d cwd=%s",
            i + 1, len(blocks), rc, len(stdout), len(stderr), subdir,
        )

        new_pngs = sorted(subdir.glob("*.png"))

        if rc != 0:
            log.warning("Block %d failed (rc=%d). Policy: %s", i, rc, stream_cfg.demo.on_failure)
            for p in new_pngs:
                p.unlink(missing_ok=True)
            block_results.append({
                "success": False,
                "figures": [],
                "stdout": stdout,
                "stderr": stderr,
                "reason": "timeout" if rc == 124 else "nonzero_exit",
                "subdir_name": subdir_name,
                "section": None,
            })
        else:
            cleaned = _clean_stdout_for_display(stdout)
            section = _build_output_section(new_pngs, cleaned, slug, subdir_name)
            block_results.append({
                "success": True,
                "figures": new_pngs,
                "stdout": stdout,
                "stderr": stderr,
                "reason": None,
                "subdir_name": subdir_name,
                "section": section,
            })

    # Rewrite the MDX from LAST block to FIRST so inserting content at
    # position `end` doesn't shift the start/end offsets of earlier blocks.
    current_mdx = mdx
    for (code, start, end), result in reversed(list(zip(blocks, block_results))):
        if result["success"]:
            section = result["section"] or ""
            if section:
                current_mdx = current_mdx[:end] + "\n" + section + current_mdx[end:]
        else:
            # Failed block: replace with warning note if policy says strip.
            if stream_cfg.demo.on_failure == "strip_demo_section":
                note = (
                    "\n> _A code demo for this section failed in CI and has "
                    "been removed. The text above still describes the method._\n"
                )
                current_mdx = current_mdx[:start] + note + current_mdx[end:]

    if current_mdx != mdx:
        mdx_path.write_text(current_mdx, encoding="utf-8")

    all_figures = [p for r in block_results for p in r["figures"]]
    all_stdout = "\n".join(r["stdout"] for r in block_results)
    all_stderr = "\n".join(r["stderr"] for r in block_results)
    all_success = all(r["success"] for r in block_results)

    return {
        "ran": True,
        "success": all_success,
        "figures": all_figures,
        "stdout": all_stdout,
        "stderr": all_stderr,
        "reason": None if all_success else "some_blocks_failed",
    }


def _strip_demo_block(mdx: str) -> str:
    """Remove ALL python fenced blocks + insert a short warning note.

    Used as a fallback when the entire demo phase needs to be stripped
    (e.g. when the stream config says to strip on failure).
    """
    note = (
        "\n> _The runnable demo for this post failed in CI and has been "
        "removed. The text above still describes the method._\n"
    )
    # Replace from the start of the first block to the end of the last block.
    blocks = extract_all_python_blocks(mdx)
    if not blocks:
        return mdx
    first_start = blocks[0][1]
    last_end = blocks[-1][2]
    return mdx[:first_start] + note + mdx[last_end:]


# ---------------------------------------------------------------------------
# Sanity helper for manual runs
# ---------------------------------------------------------------------------


def have_python_runtime() -> bool:
    """Quick check used by the orchestrator to skip the demo on hosts without Python."""
    return shutil.which(sys.executable) is not None
