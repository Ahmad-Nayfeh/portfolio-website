"""
generate_project_code.py — Claude generates a complete project from a paper.

Pipeline:
  1. Feasibility check: Claude decides if the paper is implementable
     without GPUs, using only numpy/matplotlib/scipy/sklearn/seaborn.
  2. Code generation: Claude produces a full Python project with a
     visual demo (4-8 figures using the Nocturne palette).
  3. Output: returns a dict mapping relative paths → file contents.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from anthropic import Anthropic

import cost_meter

logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-sonnet-4-6"
FALLBACK_MODEL = "claude-haiku-4-5"

NOCTURNE_COLORS = """
#080812  (dark background)
#00d4aa  (electric teal — primary accent)
#ffba08  (warm amber — secondary accent)
#ff3cac  (hot magenta — tertiary accent)
#e8e8ed  (text / warm white)
#8888a0  (muted cool grey)
"""

FEASIBILITY_PROMPT = """You are a research engineer evaluating whether a paper's core IDEA can be demonstrated as a clean Python project.

The goal is an EDUCATIONAL DEMONSTRATION — not a reproduction of the original results.
We want to teach the core concept through simplified, pedagogical code and visualizations.

Say YES ("feasible": true) when:

1. The core CONCEPT can be demonstrated with ONLY: numpy, matplotlib, scipy, scikit-learn, seaborn
   → Examples of concepts that work: adaptive filtering, spectral methods, kernel methods, PCA variants, manifold learning, optimization algorithms, MCMC sampling, compression algorithms, information-theoretic metrics, signal denoising, source separation, dimensionality reduction, clustering, change-point detection, graph algorithms, random walks, stochastic processes, numerical integration, root-finding, linear system solvers, statistical tests, bootstrapping, etc.
2. Even if the original used GPUs/torch/tensorflow, a SIMPLIFIED CPU-ONLY pedagogical version is feasible — implement a toy version that illustrates the core idea using synthetic 2D/3D data, small-scale examples, or simplified problem settings.
3. Does NOT require downloading a large dataset (>100MB) — synthetic data or sklearn.datasets is fine.
4. Can produce AT LEAST 4 informative visualizations — parameter sweeps, convergence plots, comparison charts, ablation studies, statistical visualizations.
5. Can run in under 5 minutes on a CPU.
6. Has a clear, self-contained algorithm or technique worth demonstrating — NOT just a benchmark result, dataset release, survey/taxonomy, or purely empirical study with no algorithmic substance.

Say NO only when the paper genuinely has no algorithmic core separable from its GPU implementation, or when it's purely a benchmark/dataset/survey paper.

Output JSON ONLY with this schema:
{
  "feasible": true/false,
  "reasoning": "One-sentence explanation",
  "estimated_figures": 4-8,
  "suggested_project_name": "kebab-case-project-name",
  "suggested_tags": ["tag1", "tag2", "tag3"],
  "difficulty": "easy" | "medium" | "hard"
}
"""

CODE_GENERATION_PROMPT = """You are generating a complete Python project that implements the algorithm from the paper below.

The project MUST use ONLY: numpy, matplotlib, scipy, scikit-learn, seaborn.
NO torch, tensorflow, jax, or any GPU library.
NO large external datasets — use synthetic data or sklearn.datasets.

COLOUR PALETTE (Nocturne) — every figure MUST use these colours:
{nocturne_colours}

OUTPUT STRUCTURE — Produce a JSON object with keys = relative file paths, values = file contents.

Required files:

1. "src/implementation.py"
   - Clean, well-structured Python module implementing the core algorithm
   - Importable: `from src.implementation import ...`
   - Include a main class or function that users would call
   - Type hints on all functions
   - Docstrings explaining the algorithm

1. "README.md"
   - PROFESSIONAL, polished README — the kind you'd see on a top GitHub repo.
   - Must include ALL of these sections:
     a) **Title + badge row** — project name, license badge, Python version badge
     b) **Overview** — 2-3 paragraphs explaining what this project does, what paper it implements (with link to arxiv), and why it matters
     c) **Visual Results** — embed the figures using markdown: `![Figure 1](output/figure_1.png)`. Show ALL figures inline with brief captions explaining what each shows. THIS IS CRITICAL — the figures are the star of the README.
     d) **Installation** — clear step-by-step: `git clone`, `cd`, `pip install -r requirements.txt`
     e) **Quick Start** — `python demo.py` and what to expect
     f) **Usage** — how to import and use the library in your own code, with a short code example
     g) **Project Structure** — tree view of files
     h) **How It Works** — brief technical explanation of the algorithm
     i) **Results Summary** — table of key metrics or findings from the demo
     j) **License** — MIT
   - Use markdown headings, code blocks, tables, and embedded images throughout
   - Every figure filename in "output/" must match exactly what demo.py produces

2. "demo.py"
   - THE MOST IMPORTANT FILE. This is a standalone script that:
     a) Imports from src.implementation
     b) Applies the algorithm to synthetic data
     c) Produces 6-10 matplotlib figures saved to "output/" directory
     d) Each figure explores a DIFFERENT aspect:
        - Figure 1: Main result / algorithm output visualization
        - Figure 2: Comparison with baseline or alternative
        - Figure 3: Parameter sensitivity analysis  
        - Figure 4: Performance metrics or convergence
        - Figure 5: Edge case or boundary behavior
        - Figure 6+: Additional insights (ablation, statistical analysis, etc.)
     e) Every figure uses the Nocturne palette (dark background, teal/amber/magenta accents)
     f) Figures are publication-quality: proper labels, legends, titles, grid
     g) Uses seaborn style for statistical plots where appropriate
     h) Saves each figure as .png (300 DPI) to "output/" directory
     i) Prints progress to stdout  
   - Must NOT call plt.show()
   - Must handle errors gracefully
   - FIGURE FILENAMES must be descriptive, like "main_result.png", "comparison.png", "parameter_sensitivity.png", "convergence.png", "boundary_behavior.png", "ablation.png"

3. "requirements.txt"
   - numpy
   - matplotlib
   - scipy
   - scikit-learn
   - seaborn

4. "tests/test_basic.py"
   - 3-5 basic unit tests using assertions (no pytest needed)
   - Tests that the implementation runs and produces sensible outputs

CRITICAL RULES:
- The demo MUST produce AT LEAST 6 distinct figures saved to "output/"
- The code MUST run start-to-finish in under 5 minutes on a CPU
- No external file dependencies — generate synthetic data inline
- Use numpy.random.seed(42) for reproducibility
- Every figure must have a descriptive title, axis labels, and legend where applicable
- The Nocturne dark palette MUST be applied to every figure

Output ONLY valid JSON. No markdown fences, no preamble.
"""


@dataclass
class FeasibilityResult:
    feasible: bool
    reasoning: str
    estimated_figures: int
    suggested_project_name: str
    suggested_tags: list[str]
    difficulty: str


@dataclass
class GeneratedProject:
    project_name: str
    slug: str
    description: str
    tags: list[str]
    files: dict[str, str]  # relative path -> content
    figures_expected: int
    paper_title: str
    arxiv_url: str


def _call_claude(
    client: Anthropic,
    system: str,
    prompt: str,
    label: str,
    max_tokens: int = 16000,
    model: str = CLAUDE_MODEL,
) -> str:
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        meter = cost_meter.get_meter()
        if meter and resp.usage:
            meter.record_claude(
                label=label,
                model=model,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
            )
        return resp.content[0].text
    except Exception as e:
        logger.warning(f"Claude call failed with {model}: {e}")
        if model != FALLBACK_MODEL:
            logger.info("Falling back to haiku")
            return _call_claude(client, system, prompt, label, max_tokens, FALLBACK_MODEL)
        raise


def check_feasibility(
    client: Anthropic,
    title: str,
    abstract: str,
    arxiv_url: str,
) -> FeasibilityResult:
    label = f"feasibility:{title[:40]}"
    prompt = f"Title: {title}\n\nAbstract: {abstract}\n\nURL: {arxiv_url}\n\n"
    prompt += "Evaluate feasibility per the criteria above."

    raw = _call_claude(
        client,
        system=FEASIBILITY_PROMPT,
        prompt=prompt,
        label=label,
        max_tokens=1000,
    )

    # Strip markdown fences if Claude wraps the JSON
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)

    data = json.loads(raw)
    return FeasibilityResult(
        feasible=data["feasible"],
        reasoning=data["reasoning"],
        estimated_figures=data["estimated_figures"],
        suggested_project_name=data["suggested_project_name"],
        suggested_tags=data["suggested_tags"],
        difficulty=data["difficulty"],
    )


def generate_code(
    client: Anthropic,
    title: str,
    abstract: str,
    arxiv_url: str,
    project_name: str,
) -> dict[str, str]:
    label = f"code_gen:{project_name}"
    prompt = (
        f"Paper title: {title}\n\n"
        f"Abstract: {abstract}\n\n"
        f"URL: {arxiv_url}\n\n"
        f"Project name: {project_name}\n\n"
        f"Generate the complete project code."
    )

    system = CODE_GENERATION_PROMPT.format(nocturne_colours=NOCTURNE_COLORS)

    raw = _call_claude(
        client,
        system=system,
        prompt=prompt,
        label=label,
        max_tokens=32000,
    )

    # Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)

    files = json.loads(raw)
    if not isinstance(files, dict):
        raise ValueError(f"Expected JSON dict, got {type(files)}")

    return files


def run(
    client: Anthropic,
    title: str,
    abstract: str,
    arxiv_url: str,
) -> Optional[GeneratedProject]:
    """Full pipeline: check feasibility -> generate -> return project."""

    # Stage 1: Feasibility
    logger.info("Checking feasibility for: %s", title)
    feasibility = check_feasibility(client, title, abstract, arxiv_url)

    if not feasibility.feasible:
        logger.info("Paper not feasible: %s", feasibility.reasoning)
        return None

    project_name = feasibility.suggested_project_name
    logger.info(
        "Feasible (difficulty=%s, ~%d figures). Generating project: %s",
        feasibility.difficulty,
        feasibility.estimated_figures,
        project_name,
    )

    # Stage 2: Code generation
    files = generate_code(client, title, abstract, arxiv_url, project_name)

    # Sanity check: demo.py must exist and produce figures
    if "demo.py" not in files:
        logger.error("Generated code missing demo.py")
        return None

    # Count expected figure references in demo.py
    fig_count = len(re.findall(r"(?:savefig|output/[\w-]+\.png)", files["demo.py"]))

    description = f"Implementation of the paper '{title}' — {feasibility.reasoning}"

    return GeneratedProject(
        project_name=project_name,
        slug=project_name,
        description=description,
        tags=feasibility.suggested_tags,
        files=files,
        figures_expected=max(fig_count, feasibility.estimated_figures),
        paper_title=title,
        arxiv_url=arxiv_url,
    )
