"""Token usage tracking for LLM API calls."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from kong.config import LLMProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PricingTier:
    """Per-model pricing in USD per 1M tokens.

    Rates sourced from published Anthropic/OpenAI API pricing pages.
    """

    input_rate: float
    output_rate: float
    cache_write_rate: float = 0.0
    cache_read_rate: float = 0.0


_DEFAULT_PRICING = PricingTier(input_rate=3.0, output_rate=15.0)

PRICING_REGISTRY: dict[str, PricingTier] = {
    # Anthropic: cache_write = input * 1.25, cache_read = input * 0.10
    "claude-fable-5": PricingTier(10.0, 50.0, cache_write_rate=12.50, cache_read_rate=1.00),
    "claude-opus-5": PricingTier(5.0, 25.0, cache_write_rate=6.25, cache_read_rate=0.50),
    "claude-opus-4-8": PricingTier(5.0, 25.0, cache_write_rate=6.25, cache_read_rate=0.50),
    "claude-opus-4-7": PricingTier(5.0, 25.0, cache_write_rate=6.25, cache_read_rate=0.50),
    "claude-opus-4-6": PricingTier(5.0, 25.0, cache_write_rate=6.25, cache_read_rate=0.50),
    "claude-sonnet-5": PricingTier(2.0, 10.0, cache_write_rate=2.50, cache_read_rate=0.20),
    "claude-sonnet-4-6": PricingTier(3.0, 15.0, cache_write_rate=3.75, cache_read_rate=0.30),
    "claude-sonnet-4-20250514": PricingTier(3.0, 15.0, cache_write_rate=3.75, cache_read_rate=0.30),
    "claude-haiku-4-5": PricingTier(1.0, 5.0, cache_write_rate=1.25, cache_read_rate=0.10),
    "claude-haiku-4-5-20251001": PricingTier(1.0, 5.0, cache_write_rate=1.25, cache_read_rate=0.10),
    # Z.ai: published rates for GLM-5.3, cached input billed separately.
    # glm-5.3-flash is deliberately absent: its rate is not published next to
    # the flagship one, and a guessed number reads like a measured one.
    "glm-5.3": PricingTier(1.40, 4.40, cache_read_rate=0.26),
    # OpenAI: cached input is 50% of input rate, no cache write billing
    "gpt-4o": PricingTier(2.50, 10.00, cache_read_rate=1.25),
    "gpt-4o-2024-11-20": PricingTier(2.50, 10.00, cache_read_rate=1.25),
    "gpt-4o-mini": PricingTier(0.15, 0.60, cache_read_rate=0.075),
    "gpt-4o-mini-2024-07-18": PricingTier(0.15, 0.60, cache_read_rate=0.075),
    "o1": PricingTier(15.00, 60.00, cache_read_rate=7.50),
    "o3-mini": PricingTier(1.10, 4.40, cache_read_rate=0.55),
}


_ZERO_PRICING = PricingTier(0.0, 0.0, 0.0, 0.0)


_warned_unknown_models: set[str] = set()


def is_known_model(model: str) -> bool:
    """True if *model* has published rates in the registry.

    Costs reported for an unknown model are estimates based on
    ``_DEFAULT_PRICING``, not that model's real rates.
    """
    return model in PRICING_REGISTRY


def get_pricing(model: str, provider: LLMProvider | None = None) -> PricingTier:
    if provider is LLMProvider.CUSTOM:
        return _ZERO_PRICING
    tier = PRICING_REGISTRY.get(model)
    if tier is None:
        # Falling back silently makes a wrong cost look like a measured one.
        if model not in _warned_unknown_models:
            _warned_unknown_models.add(model)
            logger.warning(
                "No published pricing for model %r; reporting costs at the "
                "default rate ($%.2f in / $%.2f out per 1M tokens). The real "
                "cost may differ.",
                model, _DEFAULT_PRICING.input_rate, _DEFAULT_PRICING.output_rate,
            )
        return _DEFAULT_PRICING
    return tier


def register_custom_model(model: str) -> None:
    if model not in PRICING_REGISTRY:
        PRICING_REGISTRY[model] = _ZERO_PRICING


@dataclass
class ModelTokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    calls: int = 0

    def cost_usd(self, model: str) -> float:
        tier = get_pricing(model)
        return (
            (self.input_tokens / 1_000_000) * tier.input_rate
            + (self.output_tokens / 1_000_000) * tier.output_rate
            + (self.cache_creation_tokens / 1_000_000) * tier.cache_write_rate
            + (self.cache_read_tokens / 1_000_000) * tier.cache_read_rate
        )


@dataclass
class TokenUsage:
    by_model: dict[str, ModelTokenUsage] = field(default_factory=dict)

    def _get(self, model: str) -> ModelTokenUsage:
        if model not in self.by_model:
            self.by_model[model] = ModelTokenUsage()
        return self.by_model[model]

    @property
    def input_tokens(self) -> int:
        return sum(m.input_tokens for m in self.by_model.values())

    @property
    def output_tokens(self) -> int:
        return sum(m.output_tokens for m in self.by_model.values())

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def calls(self) -> int:
        return sum(m.calls for m in self.by_model.values())

    @property
    def total_cost_usd(self) -> float:
        return sum(m.cost_usd(model) for model, m in self.by_model.items())
