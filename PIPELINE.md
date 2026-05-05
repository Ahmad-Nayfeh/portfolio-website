# PIPELINE.md

End-to-end reference for the auto-publish pipeline that takes a *stream*
config, calls Claude + DALL-E, and opens a pull request with a finished
blog post. Read this once and you'll know what every moving part does.

If you want to change something (add a new stream, tweak the cost ceiling,
add a stage), see `EXTENDING_PIPELINE.md`. This file just describes what
already exists.

---

## 1. High-level flow

```
GitHub UI: "Run workflow"     (manual workflow_dispatch — see §6)
        |
        v
.github/workflows/publish.yml
        |
        |-- 1. checkout + setup pnpm/node/python
        |-- 2. install deps
        |-- 3. python scripts/main.py --output-summary run-summary.json
        |        |
        |        |-- a. load streams/*.yaml          (load_streams.py)
        |        |-- b. for each due stream:
        |        |       cost_meter.init_meter($1 ceiling)
        |        |       discover papers              (HF Daily Papers)
        |        |       generate_post.run_stages()   (Claude, multi-stage)
        |        |       generate_cover_image()       (DALL-E 3)
        |        |       write .mdx + frontmatter
        |        |       run demo subprocess          (run_demo.py)
        |        |       run build validator          (validate_build.py)
        |        |       record per-stream summary
        |        |       cost_meter.reset_meter()
        |        |-- c. emit run-summary.json
        |
        |-- 4. inspect summary -> GITHUB_OUTPUT (slug, status, cost_usd, ...)
        |-- 5. peter-evans/create-pull-request opens auto/<stream>/<slug>
        |-- 6. gh pr merge --auto      (only if status == "ok")
        |-- 7. dawidd6/action-send-mail success email
        |
        +-- on failure ---> peter-evans/create-issue-from-file + failure email
```

The key invariant: **one workflow run → at most one pull request**. Even if
multiple streams ran, only the first stream that produced a post (status
`ok` or `flagged`) becomes a PR. This keeps debugging tractable.

---

## 2. File map

| Path                                         | Purpose                                                                  |
|----------------------------------------------|--------------------------------------------------------------------------|
| `.github/workflows/publish.yml`              | The orchestrator. Manual-only right now (workflow_dispatch).             |
| `streams/_defaults.yaml`                     | Defaults merged into every stream config.                                 |
| `streams/<stream-id>.yaml`                   | One file per stream. Defines schedule, discovery, stages, gates.         |
| `streams/_disabled/`                         | Park disabled streams here; loader skips this directory.                 |
| `scripts/load_streams.py`                    | Reads `streams/*.yaml`, validates with pydantic, returns typed configs.  |
| `scripts/main.py`                            | Top-level driver. Iterates streams, calls each stage, writes summary.    |
| `scripts/generate_post.py`                   | Calls Claude per-stage, assembles the MDX body + frontmatter.            |
| `scripts/generate_cover.py`                  | Calls DALL-E 3, downloads the cover PNG into `public/blog-images/`.      |
| `scripts/run_demo.py`                        | Executes the `demo_code` block in a subprocess with a timeout.           |
| `scripts/validate_build.py`                  | Runs `pnpm build` to catch MDX/Next breakage before the PR opens.        |
| `scripts/cost_meter.py`                      | Per-stream USD accumulator. Raises `CostCeilingExceeded` past the cap.   |
| `scripts/requirements.txt`                   | Python deps (anthropic, openai, pydantic, scientific stack for demos).   |
| `content/blog/<YYYY-WNN-slug>.mdx`           | The actual published posts. The pipeline writes here.                    |
| `public/blog-images/<slug>/cover.png`        | Cover images. Referenced by frontmatter `image:` field.                  |

---

## 3. The stream config

A stream is a YAML file under `streams/`. It tells the pipeline: where
to discover content, what stages to run with what prompts, what quality
gates to apply, and what the cover image should look like.

The current example is `streams/ai-papers.yaml` — a weekly AI paper
deep-dive stream. Its top-level shape:

```yaml
stream:
  id: ai-papers              # used in branch names: auto/<id>/<slug>
  name: "Paper Notes"
  enabled: true

schedule:
  cron: "0 7 * * 0"          # 10:00 Riyadh = 07:00 UTC, every Sunday
  timezone: "Asia/Riyadh"

discovery:
  source: hf_daily_papers
  pick_top_n: 1              # how many papers to bundle per post
  min_upvotes: 5

content:
  output_dir: "content/blog"
  filename_template: "{year}-W{week}-{slug}.mdx"

generation:
  model: claude-sonnet-4-6
  stages:                    # ordered. each one is a Claude call
    - name: intro_framing
      role: prose
      prompt: |
        ...
    - name: per_paper_section
      role: prose
      prompt: |
        ...
    - name: unified_demo
      role: demo
      prompt: |
        ...
    - name: closing_synthesis
      role: prose
      prompt: |
        ...
    - name: tag_extraction
      role: metadata           # consumed by frontmatter, not body
    - name: cover_image_brief
      role: metadata           # consumed by generate_cover.py

demo:
  enabled: true
  timeout_seconds: 60
  on_failure: strip_demo_section

cover_image:
  enabled: true
  model: dall-e-3
  size: "1792x1024"
  quality: "standard"
  style: "natural"

quality_gates:
  require_quotes: true
  min_word_count: 800
```

`scripts/load_streams.py` parses this with pydantic so any typo gets
flagged with a clear validation error before any API call is made.

---

## 4. Stage roles

Stage `role` controls how the output is consumed:

| role        | purpose                                           | shown in post body? |
|-------------|---------------------------------------------------|---------------------|
| `prose`     | Standard body text. Concatenated in stage order.  | yes                 |
| `demo`      | Python code block. Run by `run_demo.py`.          | yes (rendered MDX)  |
| `metadata`  | Output not part of the body. Examples: `tag_extraction` (parsed for frontmatter `tags:`), `cover_image_brief` (handed to DALL-E). | no |

The current ai-papers layout (C7 redesign):

1. `intro_framing` — frames why these papers were picked.
2. `per_paper_section` — for each paper: intuition → method → critique → quote.
3. `unified_demo` — Python demo illustrating the central idea.
4. `closing_synthesis` — alignment / divergence / what the papers are *not* saying.
5. `tag_extraction` — metadata only, becomes `tags:` in frontmatter.
6. `cover_image_brief` — metadata only, becomes the DALL-E prompt.

---

## 5. Quality gates

Three checks run before the PR opens. Each is non-blocking individually
but together they decide whether the post is `ok` (auto-mergeable) or
`flagged` (PR opens with a label, human review required).

| Gate                    | Where                          | Trigger                                                    |
|-------------------------|--------------------------------|------------------------------------------------------------|
| Demo execution          | `scripts/run_demo.py`          | Demo subprocess exits non-zero or times out (60s default). |
| Build validation        | `scripts/validate_build.py`    | `pnpm build` fails on the new MDX (e.g. broken JSX, missing component). |
| Quote sanity check      | `scripts/generate_post.py`     | A paper quote doesn't appear verbatim in the abstract / paper text. |

Failures don't stop the run — they flip the stream's `status` from `ok`
to `flagged`. The workflow then opens a PR with `auto_merge=false` so you
review it manually.

A failed quote check additionally triggers a one-shot regeneration of
the offending stage before flagging.

---

## 6. The cost meter

`scripts/cost_meter.py` is a per-stream USD accumulator. Wired into
`generate_post.py` (Claude calls) and `generate_cover.py` (DALL-E calls).
Every API response is converted to USD via the `PRICING` dict and added
to the running total. After every record, `_check_ceiling()` runs — if
the total exceeds the ceiling (default $1), it raises
`CostCeilingExceeded` and `main.py` aborts that stream cleanly.

Cost is captured per-stream in the run summary as `cost_usd` and surfaced
in the success email's "Cost:" line. The total across all streams in a
run is exposed as `total_cost_usd` to the workflow.

To change the ceiling: pass `--cost-ceiling 2.0` to `scripts/main.py`,
or edit the default in `main.py`'s argument parser.

A cost-aborted stream produces no PR (status: `cost_aborted`) and the
workflow exits 1 so you get a failure issue + email with the partial
cost shown in `run-summary.json`.

---

## 7. Triggers

Currently **manual-only**:

```
GitHub UI -> Actions -> publish -> Run workflow
  Inputs:
    stream_id   (default: "ai-papers", blank = all due streams)
    dry_run     (default: "false")
```

A weekly schedule is wired up but commented out. To resume automatic
weekly publishing:

1. Uncomment the `schedule:` block at the top of `publish.yml`.
2. Confirm each stream's `schedule.cron` is what you want (the workflow
   ticks every 30 min; the per-stream cron is what actually fires).

With both in place, the first 30-min tick that lands inside a stream's
cron window produces a post.

---

## 8. Auto-merge

Three things must be true for `gh pr merge --auto` to actually merge a
PR on its own:

1. **Repo setting** "Allow auto-merge" is on (`Settings → General →
   Pull Requests`).
2. **Branch protection** on `main` requires the **Vercel** check, and
   that check is currently passing on the PR.
3. The pipeline produced `status: ok` (no quality gate flagged the post).

Quality-flagged PRs get the `flagged` label and skip the auto-merge
call entirely — you review them by hand.

---

## 9. Notifications

Two paths via `dawidd6/action-send-mail` over Gmail SMTP:

- **Success email**: title, stream, slug, status, cost, PR URL,
  workflow run URL. Sent only when `has_post == true`.
- **Failure email + GitHub issue**: pipeline crashed (exit 1) or a
  cost ceiling was hit. Issue body inlines `run-summary.json` so you
  can see exactly what broke.

Required secrets in repo settings:

| Secret                | Used by                               |
|-----------------------|---------------------------------------|
| `ANTHROPIC_API_KEY`   | scripts/generate_post.py              |
| `OPENAI_API_KEY`      | scripts/generate_cover.py (optional)  |
| `GMAIL_USER`          | success + failure emails              |
| `GMAIL_APP_PASSWORD`  | success + failure emails              |

---

## 10. Running locally

```
# Dry run — discovery only, no API calls, no files written.
python scripts/main.py --dry-run --output-summary /tmp/summary.json

# Real run for one stream.
ANTHROPIC_API_KEY=... OPENAI_API_KEY=... \
  python scripts/main.py --stream ai-papers \
    --output-summary /tmp/summary.json

# Cap spend lower while iterating.
python scripts/main.py --stream ai-papers --cost-ceiling 0.20
```

---

## 11. Where things go wrong (and where to look)

| Symptom                                              | First place to look                                |
|------------------------------------------------------|----------------------------------------------------|
| No post produced, exit 0                             | `run-summary.json` — likely no stream was due.     |
| Pipeline exit 1, "CostCeilingExceeded"               | `cost_meter._check_ceiling`. Bump ceiling or trim stages. |
| MDX build fails on Vercel but not locally            | `validate_build.py` may have skipped — re-run `pnpm build`. |
| Quote check fails repeatedly                         | The paper abstract uses smart quotes / unicode — check the regen prompt. |
| Demo timeout                                         | Bump `demo.timeout_seconds` or simplify the demo prompt. |
| HF Daily Papers returns no candidates                | `min_upvotes` too high, or HF schema changed (see B10). |
| Cover image missing                                  | `OPENAI_API_KEY` unset, or DALL-E rate-limited. The post still ships, just without a header. |
