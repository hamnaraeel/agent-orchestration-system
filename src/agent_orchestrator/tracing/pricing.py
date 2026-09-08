"""Cost estimation from token counts, driven by `settings.model_pricing`.

The bundled default prices (see config.py) are round, illustrative
placeholders -- not verified current vendor pricing. Override
`settings.model_pricing` (or the `MODEL_PRICING` env var, as JSON) with real
rates before treating any cost figure here as accurate.
"""
from __future__ import annotations

from ..config import settings


def estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    pricing = settings.model_pricing.get(model_name, settings.model_pricing["_default"])
    return (
        input_tokens * pricing["input_per_million"] / 1_000_000
        + output_tokens * pricing["output_per_million"] / 1_000_000
    )
