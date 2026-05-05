"""scripts/run_demo.py

Extract the Python demo from a generated MDX post, run it in a subprocess
under a timeout, capture any matplotlib figures it produces, and rewrite
the demo section in the MDX to reference the rendered images and the
captured stdout.

Demo output capture (May 2026 rewrite):
  We run the demo subprocess with its CWD set to the per-post image
  folder (public/blog-images/<slug>/). That way ANY PNG the demo writes
  — whether via our wrapped `figure_<n>.png` autosave OR the demo's own
  `plt.savefig('whatever.png')` call — lands in the right folder. We
  then sweep the folder for `*.png` and inject them all under the demo
  block in the MDX.

  The previous implementation only captured figures still open at the
  end of the demo via `plt.get_fignums()`. That broke any demo that
  called `plt.savefig(...); plt.close()` (which is most of them — that
  is the standard matplotlib idiom). The new approach is forgiving: as
  long as the demo writes PNGs to the CWD, we pick them up.

Outputs:
  - public/blog-images/<slug>/<any>.png  (everything the demo wrote)
  - The MDX is updated with:
      a. a "Demo output" markdown image block per figure (in filename
         order, so the demo controls ordering by naming)
      b. a fenced text block with the captured stdout (cleaned of our
         internal SAVED_FIGURE: marker lines).

On failure (timeout, non-zero exit) the demo section is either kept (with
a warning) or stripped, depending on stream_cfg.demo.on_failure.
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
# original block with figures + stdout appended underneath.
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


def _wrap_demo_for_capture(code: str) -> str:
    """Wrap user demo code so matplotlib figures are auto-saved.

    The wrapper:
      1. Forces a non-interactive backend so the demo can't open a GUI
         window.
      2. After the demo runs, saves every still-open figure as
         figure_<n>.png in the CURRENT WORKING DIRECTORY. The caller
         (run_demo_for_post) sets CWD to the per-post image folder so
         these files land where Next.js can serve them.

    NOTE: We deliberately don't try to prevent the demo from calling
    plt.savefig() / plt.close() itself. Most matplotlib code does. The
    new design simply sweeps the CWD for ALL .png files after the
    subprocess exits, so we catch both the demo's own saves and our
    fallback autosaves.
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
        "# --- Auto-capture: save any figures still open. -------------\n"
        "import matplotlib.pyplot as _plt\n"
        "for _i, _num in enumerate(_plt.get_fignums(), start=1):\n"
        "    _fig = _plt.figure(_num)\n"
        "    _path = f'figure_{_i}.png'\n"
        "    _fig.savefig(_path, dpi=120, bbox_inches='tight')\n"
        "    print(f'SAVED_FIGURE:{_path}')\n"
    )
    return header + code + footer


def _run_subprocess(
    code: str,
    timeout_seconds: int,
    cwd: Path,
) -> tuple[int, str, str]:
    """Run `code` in a fresh Python subprocess with a hard timeout.

    The subprocess is launched with `cwd` set to the per-post image
    folder so any PNG the demo writes lands there directly.
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
    """Strip our internal SAVED_FIGURE marker lines from stdout."""
    keep = [
        ln for ln in stdout.splitlines()
        if not ln.strip().startswith("SAVED_FIGURE:")
    ]
    # Trim trailing blank lines but preserve internal whitespace.
    while keep and not keep[-1].strip():
        keep.pop()
    return "\n".join(keep)


def _build_output_section(
    image_paths: list[Path],
    cleaned_stdout: str,
    slug: str,
) -> str:
    """Compose the markdown that gets injected after the demo code block.

    Layout:
        ## Output
        ![Generated figure](/blog-images/<slug>/foo.png)
        ![Generated figure](/blog-images/<slug>/bar.png)

        ```text
        <captured stdout, with marker lines removed>
        ```

    If there are no figures and no stdout, return empty string so we
    don't inject a blank "## Output" header.
    """
    if not image_paths and not cleaned_stdout.strip():
        return ""

    lines: list[str] = ["", "**Demo output**", ""]
    for p in image_paths:
        # Public path used by Next.js: /blog-images/<slug>/<file>
        public_url = f"/blog-images/{slug}/{p.name}"
        lines.append(f"![Generated figure]({public_url})")
        lines.append("")
    if cleaned_stdout.strip():
        lines.append("```text")
        lines.append(cleaned_stdout.rstrip())
        lines.append("```")
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

    Returns: {
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

    # Snapshot pre-existing PNGs (e.g. cover.png) so we don't include
    # them in the demo-output section.
    pre_existing = {p.name for p in out_dir.glob("*.png")}

    timeout_seconds = (stream_cfg.demo.timeout_minutes or 15) * 60
    wrapped = _wrap_demo_for_capture(code)
    rc, stdout, stderr = _run_subprocess(wrapped, timeout_seconds, cwd=out_dir)
    log.info("Demo subprocess rc=%d stdout=%d stderr=%d cwd=%s",
             rc, len(stdout), len(stderr), out_dir)

    # Pick up everything new the demo wrote — either via its own
    # plt.savefig(...) OR via our autosave footer.
    new_pngs = sorted(
        p for p in out_dir.glob("*.png")
        if p.name not in pre_existing
    )

    if rc != 0:
        log.warning("Demo failed (rc=%d). Policy: %s", rc, stream_cfg.demo.on_failure)
        # Clean up anything the failed demo wrote so a half-rendered
        # demo doesn't leak into the post.
        for p in new_pngs:
            p.unlink(missing_ok=True)
        if stream_cfg.demo.on_failure == "strip_demo_section":
            new_mdx = _strip_demo_block(mdx)
            mdx_path.write_text(new_mdx, encoding="utf-8")
        return {
            "ran": True,
            "success": False,
            "figures": [],
            "stdout": stdout,
            "stderr": stderr,
            "reason": "nonzero_exit" if rc != 124 else "timeout",
        }

    # Success: inject figures + cleaned stdout right after the demo block.
    cleaned = _clean_stdout_for_display(stdout)
    section = _build_output_section(new_pngs, cleaned, slug)
    if section:
        new_mdx = mdx[:end] + "\n" + section + mdx[end:]
        mdx_path.write_text(new_mdx, encoding="utf-8")
        log.info("Demo output: injected %d figure(s) and %d chars stdout",
                 len(new_pngs), len(cleaned))
    else:
        log.info("Demo output: nothing to inject (no figures, no stdout)")

    return {
        "ran": True,
        "success": True,
        "figures": new_pngs,
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
