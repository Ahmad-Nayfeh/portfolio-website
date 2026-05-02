# streams/

Each YAML file in this directory is one publishing stream. The pipeline
(`scripts/main.py`, run by `.github/workflows/publish.yml`) iterates the
enabled streams, decides which ones are due based on their cron, and for each
due stream it discovers source material, calls the model, runs the demo (if
configured), and opens a PR that auto-merges on a green Vercel preview build.

Adding a new topic is a YAML file, not a code change.

## Layout

| Path | What |
|---|---|
| `_defaults.yaml` | Settings inherited by every stream. Streams override only what differs. |
| `<stream-id>.yaml` | One stream per file. The filename (sans `.yaml`) is the stream id. |
| `_disabled/` | Paused streams. Move a YAML here instead of deleting it. |

## Schema (the short version)

A stream YAML always has these top-level keys:

- **`stream`** — `id`, `name`, `enabled`, `language` (`en` or `ar`)
- **`schedule`** — cron expression in UTC, plus an informational timezone
- **`discovery`** — where the source material comes from (`huggingface_daily_papers`, `arxiv`, `semantic_scholar`, `manual`, `none`), how far back to look, and how the candidates are filtered/selected
- **`content`** — output path, slug pattern, default tags, cover image strategy
- **`generation`** — model, fallback model, system prompt, task instructions (or multi-stage `stages`), token budget
- **`demo`** *(optional)* — whether to run a Python demo, runtime, timeout, where outputs land, what to do if it fails
- **`approval`** — how the PR is opened and merged (`github_pr` for manual review, `github_pr_auto_merge` for auto-merge on green build)
- **`quality_gates`** — build validation, verbatim-quote requirements, etc.

The full annotated example is `ai-papers.yaml`. Copy it as a starting point.

## How to add a stream

1. Copy `ai-papers.yaml` to `streams/<your-stream-id>.yaml`.
2. Edit `stream.id` and `stream.name`. Set `enabled: false` while you're tuning.
3. Adjust `schedule.cron`, `discovery`, and `generation` for your topic.
4. Open a PR. Merge it. The pipeline picks up the new stream on its next run.
5. Watch the first auto-generated PR to make sure the output is what you wanted.
6. Flip `enabled: true` in a follow-up commit.

## How to pause a stream

Either:

- Set `stream.enabled: false` in the YAML and commit, **or**
- Move the YAML into `streams/_disabled/` and commit. (Cleaner if you want the file out of the way.)

## How to delete a stream

Don't. Move it to `_disabled/`. Disabled streams document what was tried.
