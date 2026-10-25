"""Notebook pipeline: drafts one article with a real, executed experiment, for human review.

Steps:
  1. Fetch recent arXiv papers in signal processing, image processing and medical imaging.
  2. Ask the model to pick one whose central idea can be tested on a CPU in minutes, and to
     write the experiment script.
  3. Run the script for real. On failure, send the error back for a fix (up to 3 attempts).
  4. Ask the model to write the article from results.json only.
  5. Validate the article (numbers vs results.json, structure, banned words, length).
  6. Write the files. The GitHub workflow then opens a pull request; nothing is merged
     without the owner's approval.

Model: Google Gemini (free tier of Google AI Studio). Needs GEMINI_API_KEY.
Set NOTEBOOK_FAKE_LLM=path/to/fixture.json to test the plumbing without a key.

Usage:  python scripts/notebook/pipeline.py [--report report.md]
"""
from __future__ import annotations

import argparse
import datetime as dt
import functools
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BLOG = ROOT / "content/blog"
CODE = ROOT / "notebook-code"
IMAGES = ROOT / "public/images/blog"
STATE = ROOT / "pipeline-state/notebook-used.json"
STYLE = (BLOG / "STYLE.md").read_text()
EXAMPLE = (BLOG / "lms-vs-nlms-ecg.mdx").read_text()

# Model names get retired often, so the newest available "flash" models are discovered at run
# time; GEMINI_MODEL (a repository variable) can still pin one.
FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash"]


@functools.lru_cache(maxsize=1)
def available_models(key: str) -> list[str]:
    """Stable Flash models that support generateContent, newest version first."""
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000&key=" + urllib.parse.quote(key)
        with urllib.request.urlopen(url, timeout=60) as r:
            models = json.load(r).get("models", [])
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
        print(f"  could not list models ({e!r}); using the fallback list", flush=True)
        return FALLBACK_MODELS
    names = []
    for m in models:
        name = m.get("name", "").removeprefix("models/")
        if "generateContent" not in m.get("supportedGenerationMethods", []):
            continue
        if not re.fullmatch(r"gemini-\d+(\.\d+)?-flash", name):   # skips lite, preview, image, tts, dated variants
            continue
        names.append(name)
    names.sort(key=lambda n: [int(x) for x in re.findall(r"\d+", n)], reverse=True)
    return names or FALLBACK_MODELS
ALLOWED_LIBS = "numpy, scipy, matplotlib, pandas, scikit-learn, opencv-python-headless, torch (CPU)"
BANNED = ["revolutionary", "game-changer", "game changer", "groundbreaking", "paradigm shift", "cutting-edge", "delve"]
TOPICS = "adaptive filtering, spectral analysis, denoising, image processing, medical imaging, self-supervised learning, vibration and condition monitoring"


# ---------------------------------------------------------------- model

def ask(prompt: str, system: str, want_json: bool = True) -> dict | str:
    fake = os.environ.get("NOTEBOOK_FAKE_LLM")
    if fake:
        return _fake(fake, prompt)
    key = os.environ["GEMINI_API_KEY"]
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, **({"responseMimeType": "application/json"} if want_json else {})},
    }
    last = None
    models = [m for m in [os.environ.get("GEMINI_MODEL")] if m] + available_models(key)
    print(f"  models to try: {list(dict.fromkeys(models))[:4]}", flush=True)
    for model in dict.fromkeys(models):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={urllib.parse.quote(key)}"
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = json.load(r)
                text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
                print(f"  [model {model}] {len(text)} chars", flush=True)
                return json.loads(_strip_fence(text)) if want_json else text
            except urllib.error.HTTPError as e:
                last = f"{model}: HTTP {e.code} {e.read()[:300]!r}"
                if e.code in (429, 500, 503):
                    time.sleep(20 * (attempt + 1))
                    continue
                break  # 400/403/404: try the next model
            except (KeyError, json.JSONDecodeError, urllib.error.URLError, http.client.HTTPException, OSError) as e:
                # Includes the server closing a long request without answering: wait and try again.
                last = f"{model}: {e!r}"
                print(f"  {last}; retrying", flush=True)
                time.sleep(15 * (attempt + 1))
        print(f"  model failed: {last}", flush=True)
    raise RuntimeError(f"No model answered. Last error: {last}")


def _strip_fence(t: str) -> str:
    t = t.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", t, re.S)
    return m.group(1) if m else t


def _fake(path: str, prompt: str):
    fx = json.loads(Path(path).read_text())
    for k in ["fix", "plan", "article"]:
        if f"[[{k}]]" in prompt:
            return fx[k]
    raise KeyError("fixture has no answer for this prompt")


# ---------------------------------------------------------------- papers

def recent_papers(n: int = 60) -> list[dict]:
    q = "cat:eess.SP OR cat:eess.IV OR (cat:cs.CV AND abs:medical)"
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": q, "sortBy": "submittedDate", "sortOrder": "descending", "max_results": n})
    with urllib.request.urlopen(url, timeout=60) as r:
        root = ET.fromstring(r.read())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    used = set(json.loads(STATE.read_text())) if STATE.exists() else set()
    out = []
    for e in root.findall("a:entry", ns):
        pid = e.findtext("a:id", "", ns).rsplit("/", 1)[-1].split("v")[0]
        if pid in used:
            continue
        out.append({
            "id": pid,
            "title": " ".join(e.findtext("a:title", "", ns).split()),
            "authors": [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)][:6],
            "published": e.findtext("a:published", "", ns)[:10],
            "abstract": " ".join(e.findtext("a:summary", "", ns).split())[:1500],
        })
    return out


# ---------------------------------------------------------------- experiment

SYSTEM = f"""You write experiments and articles for the Notebook of an electrical engineer's portfolio.
Honesty over impressiveness: never invent numbers, quotes or citations. Follow these rules exactly:

{STYLE}"""

PLAN_PROMPT = """[[plan]]
Below are recent arXiv papers. Pick ONE whose central idea you can test yourself, honestly, with a
small experiment that runs on a laptop CPU in under 10 minutes. Prefer topics in: {topics}.
The experiment must compare the paper's idea with a simple, strong baseline, and say plainly if the
baseline wins.

Data must be one of: synthetic data generated in the script; datasets bundled with SciPy or
scikit-learn (e.g. scipy.datasets.electrocardiogram, sklearn.datasets.load_digits); or a small public
file (< 100 MB) downloaded by the script from a stable URL with no login.
Allowed libraries: {libs}. Fixed random seeds.

The script:
- reads OUT_DIR from the environment and saves every figure there as PNG (2 to 4 figures, labelled axes),
- writes every number the article may quote to results.json in the current directory (keys with
  clear names, values as plain numbers or lists/dicts of numbers),
- prints nothing important that is not also in results.json,
- uses matplotlib with the Agg backend and never calls plt.show().

Answer as JSON:
{{"paper_id": "...", "slug": "kebab-case-slug", "title": "plain English title",
  "question": "the question the article answers", "why_testable": "...",
  "figures": ["file names the script saves"], "code": "the full experiment.py"}}

Papers:
{papers}
"""

FIX_PROMPT = """[[fix]]
The experiment script failed. Fix it and return the full corrected script, keeping the same goal,
figures and results.json keys where possible.

Error (last lines):
{error}

Script:
{code}

Answer as JSON: {{"code": "the full corrected experiment.py"}}"""

ARTICLE_PROMPT = """[[article]]
Write the Notebook article for this experiment, as an MDX file with frontmatter, following the rules
and the structure of the example. Use ONLY numbers that appear in results.json (round them the way
the example does, but never change them). If a result goes against the paper's claim, say so plainly.
Figures are at /images/blog/{slug}/<file>. The full script will be at notebook-code/{slug}/experiment.py
in https://github.com/Ahmad-Nayfeh/portfolio-website. Cite the paper (authors, arXiv id, year) under
"## Sources", and only sources you are certain exist.

Frontmatter fields: title, slug ("{slug}"), date ("{date}"), tags (3 to 4), excerpt (two sentences,
states the main finding), coverImage ("/images/blog/{slug}/<one of the figures>"), featured (false).

Paper: {paper}
Question: {question}
Figures: {figures}

experiment.py:
```python
{code}
```

results.json:
{results}

Example article (structure and tone to follow, not content):
{example}

Answer as JSON: {{"mdx": "the full article"}}"""


def run_experiment(code: str, workdir: Path, out_dir: Path, timeout: int = 900) -> tuple[bool, str]:
    (workdir / "experiment.py").write_text(code)
    env = {**os.environ, "OUT_DIR": str(out_dir), "MPLBACKEND": "Agg"}
    try:
        p = subprocess.run([sys.executable, "experiment.py"], cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout} s. Make the experiment smaller."
    log = (p.stdout + "\n" + p.stderr)[-4000:]
    if p.returncode != 0:
        return False, log
    if not (workdir / "results.json").exists():
        return False, log + "\nNo results.json was written."
    pngs = list(out_dir.glob("*.png"))
    if not pngs:
        return False, log + "\nNo PNG figures were saved to OUT_DIR."
    return True, log


# ---------------------------------------------------------------- validation

def _numbers_in(obj) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in _numbers_in(v)]
    if isinstance(obj, (list, tuple)):
        return [x for v in obj for x in _numbers_in(v)]
    return []


def validate(mdx: str, results: dict, code: str) -> list[str]:
    problems = []
    fm = re.match(r"^---\n(.*?)\n---\n", mdx, re.S)
    if not fm:
        return ["No frontmatter."]
    for field in ["title:", "slug:", "date:", "excerpt:", "tags:"]:
        if field not in fm.group(1):
            problems.append(f"Frontmatter is missing {field}")
    body = mdx[fm.end():]
    prose = re.sub(r"```.*?```", "", body, flags=re.S)
    prose = re.sub(r"\$\$.*?\$\$", "", prose, flags=re.S)
    prose = prose.replace("\u2212", "-")   # typographic minus
    words = len(re.findall(r"\w+", prose))
    if not 1100 <= words <= 2200:
        problems.append(f"Prose length is {words} words (target 1200-1800).")
    if "```python" not in body:
        problems.append("No python code block.")
    for heading in ["does not show", "## Sources"]:
        if heading.lower() not in body.lower():
            problems.append(f"Missing section: {heading}")
    low = prose.lower()
    for w in BANNED:
        if w in low:
            problems.append(f"Banned phrase: {w}")
    # Every decimal number, percentage, or number with 3+ digits in the prose must come from the run.
    known = _numbers_in(results)
    in_code = set(re.findall(r"\d+(?:\.\d+)?", code))
    for m in re.finditer(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?=%|\b)", prose):
        tok = m.group(0).replace(",", "")
        v = float(tok)
        is_pct = prose[m.end():m.end() + 1] == "%"
        if "." not in tok and not is_pct and abs(v) < 100:
            continue                     # small integers: counts, section numbers
        if 1900 <= v <= 2100 and "." not in tok:
            continue                     # years
        if tok.lstrip("-") in in_code:
            continue                     # a parameter of the experiment
        decimals = len(tok.split(".")[1]) if "." in tok else 0
        tol = 0.5 * 10 ** -decimals + 1e-9
        ok = any(abs(abs(k) - abs(v)) <= tol or abs(abs(k) * 100 - abs(v)) <= tol for k in known)
        if not ok:
            problems.append(f"Number not found in results.json: {m.group(0)}{'%' if is_pct else ''}")
    return problems


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="notebook-report.md")
    args = ap.parse_args()
    report = [f"# Notebook draft, {dt.date.today().isoformat()}", ""]

    papers = recent_papers()
    print(f"{len(papers)} candidate papers")
    listing = "\n\n".join(f"[{p['id']}] {p['title']} ({p['published']})\n{p['abstract'][:700]}" for p in papers[:25])
    plan = ask(PLAN_PROMPT.format(topics=TOPICS, libs=ALLOWED_LIBS, papers=listing), SYSTEM)
    slug = re.sub(r"[^a-z0-9-]", "", plan["slug"].lower())[:60].strip("-")
    paper = next((p for p in papers if p["id"] == plan["paper_id"]), None)
    if not paper or not slug:
        raise SystemExit(f"Bad plan: {plan.get('paper_id')} / {plan.get('slug')}")
    if (BLOG / f"{slug}.mdx").exists():
        slug += "-" + dt.date.today().strftime("%Y%m%d")
    print(f"Paper {paper['id']}: {paper['title']}\nSlug {slug}")

    work = Path(tempfile.mkdtemp())
    out = work / "figures"
    out.mkdir()
    code = plan["code"]
    ok, log = run_experiment(code, work, out)
    attempts = 1
    while not ok and attempts < 4:
        print(f"Run {attempts} failed:\n{log[-1500:]}")
        code = ask(FIX_PROMPT.format(error=log[-3000:], code=code), SYSTEM)["code"]
        shutil.rmtree(out); out.mkdir()
        ok, log = run_experiment(code, work, out)
        attempts += 1
    if not ok:
        raise SystemExit(f"The experiment still fails after {attempts} runs:\n{log}")
    results = json.loads((work / "results.json").read_text())
    figures = sorted(p.name for p in out.glob("*.png"))
    print(f"Experiment ran in {attempts} attempt(s); figures: {figures}")

    article = ask(ARTICLE_PROMPT.format(
        slug=slug, date=dt.date.today().isoformat(), paper=json.dumps(paper), question=plan["question"],
        figures=figures, code=code, results=json.dumps(results, indent=1)[:12000], example=EXAMPLE), SYSTEM)["mdx"]
    article = re.sub(r'^slug: ".*?"$', f'slug: "{slug}"', article, count=1, flags=re.M)
    problems = validate(article, results, code)

    # Write everything; the workflow commits it to a branch and opens the pull request.
    (CODE / slug).mkdir(parents=True, exist_ok=True)
    (CODE / slug / "experiment.py").write_text(code)
    (CODE / slug / "results.json").write_text(json.dumps(results, indent=2))
    (IMAGES / slug).mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.png"):
        shutil.copy(f, IMAGES / slug / f.name)
    (BLOG / f"{slug}.mdx").write_text(article)
    STATE.parent.mkdir(exist_ok=True)
    used = json.loads(STATE.read_text()) if STATE.exists() else []
    STATE.write_text(json.dumps(sorted(set(used) | {paper["id"]}), indent=1))

    report += [
        f"**Paper:** [{paper['title']}](https://arxiv.org/abs/{paper['id']}) ({paper['published']})",
        f"**Question:** {plan['question']}",
        f"**Experiment:** ran successfully on attempt {attempts}; {len(figures)} figures.",
        "",
        "## Automatic checks",
        "All passed." if not problems else "\n".join(f"- {p}" for p in problems),
        "",
        "## Before approving",
        "- Read the article: is every claim backed by the results and the figures?",
        "- Is the paper's idea described fairly? Open the paper link and check.",
        "- Merge to publish; close to discard. Nothing goes live without your approval.",
    ]
    Path(args.report).write_text("\n".join(report))
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as f:
            f.write(f"slug={slug}\ntitle={paper['title'][:120]}\nchecks={'passed' if not problems else 'flagged'}\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
