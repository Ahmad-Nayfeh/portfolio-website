"""scripts/cost_meter.py

Cost accumulator for the publish pipeline.

Tracks cumulative API spend (Claude tokens + DALL-E images) across a single
pipeline run and aborts the run if the cumulative cost exceeds a configurable
ceiling. Behavior at the ceiling: raise `CostCeilingExceeded`. The
orchestrator (`main.py`) catches that and converts it into a fatal exit so
no half-finished post is written.

Architecture: a module-level singleton, initialized once per stream by
`main.py`, queried by call sites in `generate_post.py` and `generate_cover.py`.
We use a singleton (rather than threading a meter through every function
signature) because the call sites are deep in the stack and we don't want
the cost concern leaking into every type signature.

When pricing changes, update the `PRICING` dict below — that's the only
place. As of May 2026 these are the rates:
  - Claude Sonnet 4.6: $3.00/MTok input, $15.00/MTok output (verify on
    anthropic.com/pricing if you've waited a while since this was written)
  - DALL-E 3 1792x1024 standard: $0.08/image
  - DALL-E 3 1024x1024 standard: $0.04/image

Important: these rates are best-effort, not authoritative. If you're
running this years after May 2026, please re-verify before trusting the
numbers — vendors change prices. Over-estimating is safer than
under-estimating since the ceiling is meant to be a guardrail, not a budget.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pricing (USD)
# ---------------------------------------------------------------------------
#
# Token-based models: per-million-token rates.
# Image models: flat per-image rate.
#
# When you add a new model (or vendor changes a rate), edit this dict and
# nothing else. Unknown models fall back to a conservative Sonnet-4 estimate
# with a warning.

PRICING: dict[str, dict[str, float]] = {
    # Anthropic Claude.
    "claude-sonnet-4-6":            {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-6-20250930":   {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-5":            {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-5-20250514":   {"input": 3.00, "output": 15.00},
    "claude-sonnet-4":              {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-20250514":     {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5":             {"input": 1.00, "output":  5.00},
    "claude-haiku-4-5-20251001":    {"input": 1.00, "output":  5.00},
    "claude-opus-4-6":              {"input": 15.0, "output": 75.00},

    # OpenAI DALL-E 3 — flat per image, key is `<model>-<size>-<quality>`.
    "dall-e-3-1792x1024-standard": {"per_image": 0.080},
    "dall-e-3-1024x1024-standard": {"per_image": 0.040},
    "dall-e-3-1792x1024-hd":       {"per_image": 0.120},
    "dall-e-3-1024x1024-hd":       {"per_image": 0.080},
}


def _claude_rates(model: str) -> dict[str, float]:
    """Look up Claude per-MTok rates, falling back to a safe default."""
    rates = PRICING.get(model)
    if rates is None or "input" not in rates:
        log.warning(
            "cost_meter: unknown Claude model %r — defaulting to Sonnet-4 pricing "
            "($3/$15 per MTok). Add it to PRICING in cost_meter.py for accurate "
            "cost tracking.",
            model,
        )
        return {"input": 3.00, "output": 15.00}
    return rates


def _dalle_per_image(model: str, size: str, quality: str) -> float:
    """Look up the per-image DALL-E cost. Returns the flat USD price."""
    key = f"{model}-{size}-{quality}"
    rates = PRICING.get(key)
    if rates is None or "per_image" not in rates:
        log.warning(
            "cost_meter: unknown image model key %r — defaulting to $0.08. "
            "Add it to PRICING in cost_meter.py for accurate tracking.",
            key,
        )
        return 0.08
    return rates["per_image"]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CostCeilingExceeded(Exception):
    """Raised when cumulative cost passes the configured ceiling."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CostEntry:
    label: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    images: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "images": self.images,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class CostMeter:
    ceiling_usd: float = 1.0
    entries: list[CostEntry] = field(default_factory=list)

    @property
    def total_usd(self) -> float:
        return sum(e.cost_usd for e in self.entries)

    def record_claude(
        self,
        *,
        label: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        rates = _claude_rates(model)
        cost = (
            (input_tokens / 1_000_000.0) * rates["input"]
            + (output_tokens / 1_000_000.0) * rates["output"]
        )
        entry = CostEntry(
            label=label,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        self.entries.append(entry)
        log.info(
            "cost_meter: %s on %s — %d in + %d out tokens = $%.4f "
            "(running total $%.4f / ceiling $%.2f)",
            label, model, input_tokens, output_tokens, cost,
            self.total_usd, self.ceiling_usd,
        )
        self._check_ceiling(stage_label=label)

    def record_dalle(
        self,
        *,
        label: str,
        model: str,
        size: str,
        quality: str,
    ) -> None:
        cost = _dalle_per_image(model, size, quality)
        entry = CostEntry(
            label=label,
            model=f"{model} ({size}, {quality})",
            images=1,
            cost_usd=cost,
        )
        self.entries.append(entry)
        log.info(
            "cost_meter: %s on %s/%s/%s — 1 image = $%.4f "
            "(running total $%.4f / ceiling $%.2f)",
            label, model, size, quality, cost,
            self.total_usd, self.ceiling_usd,
        )
        self._check_ceiling(stage_label=label)

    def _check_ceiling(self, *, stage_label: str) -> None:
        if self.total_usd > self.ceiling_usd:
            raise CostCeilingExceeded(
                f"Cost ceiling exceeded at stage '{stage_label}': "
                f"cumulative ${self.total_usd:.4f} > ceiling ${self.ceiling_usd:.2f}. "
                f"Aborting before further API calls."
            )

    def to_summary(self) -> dict:
        return {
            "ceiling_usd": round(self.ceiling_usd, 4),
            "total_usd": round(self.total_usd, 4),
            "entries": [e.to_dict() for e in self.entries],
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


_meter: Optional[CostMeter] = None


def init_meter(ceiling_usd: float = 1.0) -> CostMeter:
    """Initialize a fresh meter for a new stream run.

    Called by the orchestrator at the start of each stream iteration so the
    ceiling is per-blog, not per-workflow.
    """
    global _meter
    _meter = CostMeter(ceiling_usd=ceiling_usd)
    return _meter


def get_meter() -> Optional[CostMeter]:
    """Return the active meter, or None if no run is in progress.

    Call sites are intentionally tolerant of `None` so the pipeline still
    works in tests / dry runs that never call `init_meter`.
    """
    return _meter


def reset_meter() -> None:
    """Clear the singleton. Used between streams so cost doesn't leak."""
    global _meter
    _meter = None
