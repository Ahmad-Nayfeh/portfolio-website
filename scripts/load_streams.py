"""scripts/load_streams.py

Reads streams/*.yaml, merges each with streams/_defaults.yaml, validates with
pydantic, and returns the streams that are due to run.

A stream is "due" when its cron expression would have fired since the last
run for that stream. Since we don't store last-run state in the repo, we use
a coarser rule that matches the way GitHub Actions schedules behave in
practice: the stream is due if its cron matches the current 30-minute window
(the workflow runs every 30 minutes — see .github/workflows/publish.yml).

Importable: no top-level side effects. The orchestrator (main.py) calls
`load_due_streams()` to drive the pipeline.
"""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
STREAMS_DIR = REPO_ROOT / "streams"


# ---------------------------------------------------------------------------
# Stream config models — kept loose. We validate the shape of what we use,
# not every field. Fields the YAML may carry but Python doesn't read are
# preserved on the model via `model_config = ConfigDict(extra="allow")`.
# ---------------------------------------------------------------------------


class _Looser(BaseModel):
    model_config = ConfigDict(extra="allow")


class StreamMeta(_Looser):
    id: str
    name: str
    enabled: bool = True
    language: str = "en"


class Schedule(_Looser):
    cron: str
    timezone: str = "UTC"


class Discovery(_Looser):
    source: str = "none"
    lookback_days: int = 7
    top_n: int = 10
    filters: dict[str, Any] = Field(default_factory=dict)
    selection: dict[str, Any] = Field(default_factory=dict)
    # Path (relative to repo root, or absolute) to the JSON ledger that
    # tracks which papers have already been featured. Read by discovery to
    # filter candidates and appended-to by main.py after a successful
    # publish. None = no dedup, which is the legacy behaviour.
    dedup_state_file: str | None = None


class Content(_Looser):
    output_path: str = "content/blog"
    slug_pattern: str = "{year}-W{week}-{topic-slug}"
    cover_image: str = "auto"
    tags: list[str] = Field(default_factory=list)


class Generation(_Looser):
    model: str = "claude-sonnet-4-6"
    fallback_model: str = "claude-haiku-4-5"
    use_batch_api: bool = True
    max_output_tokens: int = 64000
    system_prompt: str = ""
    task_instructions: str = ""
    stages: list[dict[str, Any]] = Field(default_factory=list)


class Demo(_Looser):
    enabled: bool = False
    runtime: str = "github_actions_cpu"
    timeout_minutes: int = 15
    output_dir: str = "public/blog-images/{slug}"
    on_failure: str = "strip_demo_section"


class CoverImage(_Looser):
    """DALL-E 3 cover image generation config (used by generate_cover.py).

    Disabled by default so a stream that doesn't want covers (e.g. a future
    text-only stream) just leaves it unset.
    """

    enabled: bool = False
    model: str = "dall-e-3"
    # Landscape blog cover. 1792x1024 = $0.08 standard; square would be cheaper
    # but looks wrong as a 16:9 header. Override per-stream if needed.
    size: str = "1792x1024"
    quality: str = "standard"
    style: str = "natural"
    output_dir: str = "public/blog-images/{slug}"


class Approval(_Looser):
    method: str = "github_pr"
    reviewers: list[str] = Field(default_factory=list)
    branch_pattern: str = "auto/{stream-id}/{slug}"
    pr_title_pattern: str = "[{stream-name}] {post-title}"


class QualityGates(_Looser):
    build_validation: bool = True
    require_verbatim_quotes: int = 0


class StreamConfig(BaseModel):
    """The full, merged config for one stream."""

    model_config = ConfigDict(extra="allow")

    stream: StreamMeta
    schedule: Schedule
    discovery: Discovery = Field(default_factory=Discovery)
    content: Content = Field(default_factory=Content)
    generation: Generation = Field(default_factory=Generation)
    demo: Demo = Field(default_factory=Demo)
    cover_image: CoverImage = Field(default_factory=CoverImage)
    approval: Approval = Field(default_factory=Approval)
    quality_gates: QualityGates = Field(default_factory=QualityGates)


# ---------------------------------------------------------------------------
# Loading + merging
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge — override wins on conflict, lists are replaced."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return data


def load_all_streams() -> list[StreamConfig]:
    """Load every enabled stream YAML, merged with _defaults.yaml."""
    if not STREAMS_DIR.is_dir():
        log.warning("streams/ directory not found at %s", STREAMS_DIR)
        return []

    defaults_path = STREAMS_DIR / "_defaults.yaml"
    defaults = _load_yaml(defaults_path) if defaults_path.exists() else {}

    streams: list[StreamConfig] = []
    for path in sorted(STREAMS_DIR.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        merged = _deep_merge(defaults, _load_yaml(path))
        try:
            cfg = StreamConfig.model_validate(merged)
        except Exception as e:
            log.error("Stream %s failed validation: %s", path.name, e)
            raise
        streams.append(cfg)
        log.info("Loaded stream %s (enabled=%s)", cfg.stream.id, cfg.stream.enabled)

    return streams


# ---------------------------------------------------------------------------
# Cron matching for the 30-minute window
# ---------------------------------------------------------------------------


def _cron_field_matches(field: str, value: int) -> bool:
    """Match a single cron field against a value. Supports `*`, `*/N`, `N`, `N,M`, `N-M`."""
    field = field.strip()
    if field == "*":
        return True
    for part in field.split(","):
        part = part.strip()
        if part.startswith("*/"):
            try:
                step = int(part[2:])
                if step > 0 and value % step == 0:
                    return True
            except ValueError:
                continue
        elif "-" in part:
            try:
                lo, hi = (int(x) for x in part.split("-", 1))
                if lo <= value <= hi:
                    return True
            except ValueError:
                continue
        else:
            try:
                if int(part) == value:
                    return True
            except ValueError:
                continue
    return False


def cron_matches(cron: str, now: dt.datetime) -> bool:
    """Return True if `cron` (5-field UTC) would fire within the current 30-minute window.

    Cron fields: minute hour dom month dow (0=Sunday).
    We round `now.minute` to the nearest 30-minute mark and check whether any
    minute in [mark, mark+29] matches the minute field.
    """
    parts = cron.split()
    if len(parts) != 5:
        return False
    minute_field, hour_field, dom_field, month_field, dow_field = parts

    # Match the non-minute fields against now (UTC).
    if not _cron_field_matches(hour_field, now.hour):
        return False
    if not _cron_field_matches(dom_field, now.day):
        return False
    if not _cron_field_matches(month_field, now.month):
        return False
    if not _cron_field_matches(dow_field, now.weekday() % 7 if dow_field == "*" else (now.isoweekday() % 7)):
        # cron's dow uses 0=Sunday; Python weekday() uses 0=Monday.
        # The above safe-guards `*`, otherwise convert: Sunday -> 0.
        dow_value = (now.isoweekday() % 7)
        if not _cron_field_matches(dow_field, dow_value):
            return False

    # Window: round down to 30-min boundary, check 30 minutes forward.
    window_start = (now.minute // 30) * 30
    for m in range(window_start, window_start + 30):
        if m >= 60:
            break
        if _cron_field_matches(minute_field, m):
            return True
    return False


def load_due_streams(now: dt.datetime | None = None) -> list[StreamConfig]:
    """Return enabled streams whose cron fires in the current 30-min window."""
    if now is None:
        now = dt.datetime.utcnow()
    due: list[StreamConfig] = []
    for cfg in load_all_streams():
        if not cfg.stream.enabled:
            log.info("Skipping %s: disabled", cfg.stream.id)
            continue
        if not cron_matches(cfg.schedule.cron, now):
            log.info(
                "Skipping %s: cron %r not due at %s UTC",
                cfg.stream.id,
                cfg.schedule.cron,
                now.isoformat(timespec="minutes"),
            )
            continue
        log.info("Stream %s is DUE", cfg.stream.id)
        due.append(cfg)
    return due


def get_stream_by_id(stream_id: str) -> StreamConfig | None:
    """Used by manual workflow_dispatch runs targeting a specific stream."""
    for cfg in load_all_streams():
        if cfg.stream.id == stream_id:
            return cfg
    return None
