# EXTENDING_PIPELINE.md

How to change the pipeline without breaking it. Read `PIPELINE.md` first
for the high-level map; this file is task-oriented ("I want to do X →
here's what to edit").

---

## Add a new stream

A stream is a YAML file under `streams/`. The pipeline iterates every
file matching `streams/*.yaml` (it skips `streams/_disabled/` and any
file starting with `_`).

1. Copy `streams/ai-papers.yaml` to `streams/<your-stream-id>.yaml`.
2. Change the `stream.id`, `stream.name`, `discovery.source`, and
   `generation.stages` blocks for your domain.
3. Pick a cron in `schedule.cron` (UTC). The workflow itself only ticks
   when its outer schedule fires (currently manual; see PIPELINE §7).
4. Decide whether you want covers (`cover_image.enabled`), demos
   (`demo.enabled`), and which quality gates apply.
5. Test with a dry run: `python scripts/main.py --stream <id> --dry-run`.
6. Then a real run with a tight cost cap: `--cost-ceiling 0.30`.

The first commit should ideally land in a feature branch — once it works
end-to-end, merge to main.

If you want to **disable** a stream temporarily: either set
`stream.enabled: false`, or move the file into `streams/_disabled/`.
Both work; the directory move is preferred for long pauses since it
keeps `streams/` listings clean.

---

## Anatomy of a stage

A stage is one Claude call with one prompt. Stages run in order and
their outputs are stored in a `stage_outputs: dict[str, str]` keyed by
`name`. Later stages can reference earlier outputs by interpolating
`{stage_outputs.<name>}` inside their prompt.

```yaml
- name: per_paper_section
  role: prose                    # prose | demo | metadata
  model: claude-sonnet-4-6        # optional override; defaults to generation.model
  max_tokens: 4096                # optional
  prompt: |
    You are writing one section of a multi-paper post.

    Paper title: {paper.title}
    Abstract:    {paper.abstract}

    Earlier intro for context:
    {stage_outputs.intro_framing}

    Write a self-contained section: intuition first (no jargon),
    then method, then a critique paragraph...
```

Variables available in prompts:

| Variable                       | What it is                                            |
|--------------------------------|-------------------------------------------------------|
| `{paper.title}`, `{paper.abstract}`, etc. | The current paper from discovery.          |
| `{stage_outputs.<stage_name>}` | The full text output of an earlier stage.             |
| `{stream.name}`, `{stream.id}` | The stream config metadata.                            |

Two roles need extra care:

- **`role: demo`** — the stage output is parsed for a `python` code
  fence and that fence is what `run_demo.py` executes. Make the prompt
  insist on a single self-contained code block with no shell commands.
- **`role: metadata`** — the output is *not* concatenated into the body.
  Two recognized stage names are wired into the rest of the system:
  `tag_extraction` (parsed for the post's `tags:` frontmatter, expects a
  short comma-separated list) and `cover_image_brief` (the first line is
  used as the DALL-E visual brief).

---

## Add a new metadata stage

If you want a metadata stage that does something *new* (e.g. extract a
"reading difficulty" score), there are two options:

1. **Cheap path** — name the stage `tag_extraction` and put the score
   in the tag list. Free, no code change.
2. **Proper path** — add the stage in YAML under a new name (e.g.
   `difficulty_score`), then teach `scripts/generate_post.py` to read
   `stage_outputs["difficulty_score"]` and write it into the
   frontmatter dict before the file is rendered.

For (2), the place to edit is the `_assemble_frontmatter()` function in
`generate_post.py`.

---

## Change the cover image style

Edit `STYLE_SUFFIX` at the top of `scripts/generate_cover.py`. This one
constant defines the "magazine look" applied to every cover across
every stream — change it once, the whole stream re-skins.

To override per-stream, add a `cover_image.style_suffix` field in the
stream YAML, then read it in `generate_cover._build_dalle_prompt`.

To turn off covers for one stream: set `cover_image.enabled: false`.
The post still ships, just without a `image:` frontmatter field.

---

## Change quality gates

The three gates live in different files:

| Gate                | File                          | Knob                                             |
|---------------------|-------------------------------|--------------------------------------------------|
| Demo execution      | `scripts/run_demo.py`         | `demo.timeout_seconds` in stream YAML.           |
| Build validation    | `scripts/validate_build.py`   | Always on; to skip, gate the call in `main.py`.  |
| Quote sanity check  | `scripts/generate_post.py`    | `quality_gates.require_quotes` in stream YAML.   |

To add a new gate: write a function that returns `(ok: bool, reason: str)`,
call it from `_run_stream` in `main.py` after generation, and update the
`status` field to `flagged` if it fails.

---

## Change the cost ceiling

Three places, in order of preference:

1. **Per-run override**: `python scripts/main.py --cost-ceiling 2.0`.
   Useful while iterating.
2. **Workflow override**: add a `cost_ceiling` workflow_dispatch input
   in `publish.yml` and pass it to `main.py`. Lets you bump the cap
   from the GitHub UI without a commit.
3. **Default for all runs**: change the `default` of the
   `--cost-ceiling` argument in `main.py`.

Pricing for new models lives in `scripts/cost_meter.py`'s `PRICING`
dict. If you switch models, add the entry there too — otherwise the
meter logs a warning and counts that call as $0, which silently breaks
the ceiling.

---

## Add a new discovery source

The current loader only knows `hf_daily_papers`. To add another:

1. Add a function `_discover_<source>(stream_cfg) -> list[dict]` in
   `scripts/main.py` (or a new `scripts/discovery_<source>.py` if it's
   substantial). Each dict should match the shape consumed by the
   generation prompts (`title`, `abstract`, `url`, `authors`, etc.).
2. Wire it into the `discovery.source` dispatch in `_run_stream`.
3. Document the new schema fields in `streams/_defaults.yaml`'s comments.

Keep discovery deterministic — run it twice with the same inputs and
get the same picks. This makes the dry-run output trustworthy.

---

## Switch from manual to automatic publishing

Currently `.github/workflows/publish.yml` only fires on
`workflow_dispatch`. To resume scheduled publishing:

1. Uncomment the `schedule:` block at the top of the workflow.
2. Confirm the outer cron is *more frequent* than any per-stream cron —
   e.g. `*/30 * * * *` (every 30 min) so a Sunday-10:00-Riyadh stream
   actually catches its window.
3. Push to `main`. The next tick that overlaps a stream's cron fires a
   real run.

If you want to stay manual but pre-fill inputs for one specific stream,
edit the `default:` of the `stream_id` input in the workflow YAML.

---

## Configure auto-merge end-to-end (one-time)

Two parts to this and they have to happen in this order:

1. **Repo-level**: `Settings → General → Pull Requests → Allow
   auto-merge` — checkbox on, save. (Already done.)
2. **Branch protection on `main`**:
   a. `Settings → Branches → branch protection rule for main → Edit`.
   b. Enable "Require status checks to pass before merging".
   c. *After* the next pipeline PR has appeared and Vercel has posted a
      check on it, come back here, search "Vercel" in the required-checks
      box, add it, and save. (GitHub only lists checks that have run at
      least once — that's why this is two-step.)

From the next PR onward, `gh pr merge --auto` (already in the workflow)
will merge cleanly when Vercel goes green.

---

## Common pitfalls

- **Riyadh timezone**: all `schedule.cron` values are evaluated as
  **UTC** by GitHub Actions. The `timezone:` field in stream YAML is
  informational only. 10:00 Riyadh = 07:00 UTC.
- **Frontmatter validation**: pydantic models in `load_streams.py`
  reject unknown keys *strictly* in some places, *loosely* in others
  (`_Looser` base class). When you add a new field and pydantic
  complains, check whether it should subclass `_Looser`.
- **Demo timeout**: the default 60s is generous for numpy/sklearn.
  Anything that imports torch will blow past it. Either bump the
  timeout or strip torch from the demo prompt.
- **Bind-mount sync (Cowork-only)**: when editing files via the file
  tools and then running `python -m py_compile` from a Linux shell,
  the Linux side may see a stale truncated copy. If `py_compile`
  reports a syntax error in code that *looks* fine, write through bash
  with a heredoc to bypass the mount delay.
- **Cost meter silently zeroes new models**: if you switch
  `generation.model` to a model that's not in `cost_meter.PRICING`, the
  meter logs a warning and treats the call as free. Always update
  `PRICING` when you change models.
- **One PR per run**: `Inspect run summary` picks the *first* posted
  stream and ignores the rest. If you run two streams in one workflow
  run, the second one's MDX is on disk but no PR opens for it. Prefer
  separate workflow runs for separate streams.
