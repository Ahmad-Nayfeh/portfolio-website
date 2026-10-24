# Notebook pipeline

Drafts one Notebook article every two weeks (1st and 15th of the month), with an experiment that
really runs, and opens a pull request. Nothing is published until the pull request is merged.

1. Recent arXiv papers in signal processing, image processing and medical imaging are fetched.
2. Gemini picks one whose idea can be tested on a CPU in minutes, and writes `experiment.py`.
3. The script runs in GitHub Actions. If it fails, the error goes back to Gemini (up to 3 fixes).
4. Gemini writes the article from `results.json` only, following `content/blog/STYLE.md`.
5. Automatic checks: every result number in the prose must be in `results.json`, required sections,
   length, banned words. Problems are listed in the pull request; the title says `passed` or `flagged`.

Setup (once): add the repository secret `GEMINI_API_KEY` (free key from
https://aistudio.google.com/apikey). Optionally set the repository variable `GEMINI_MODEL`.
Run it now: Actions, then "notebook-draft", then "Run workflow".

Test without a key: `NOTEBOOK_FAKE_LLM=fixture.json python scripts/notebook/pipeline.py`
(the fixture holds canned "plan", "fix" and "article" answers).

The older Claude + DALL-E pipeline (`publish.yml`, `project-publish.yml`, `scripts/*.py`) is
disabled and not used.
