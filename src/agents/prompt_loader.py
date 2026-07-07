from __future__ import annotations

import yaml


def load_prompts(path: str) -> dict[str, dict[str, str]]:
    """Load all prompt variants from a YAML file (see agents/prompts.yaml)."""
    with open(path) as f:
        raw: dict[str, dict[str, str]] = yaml.safe_load(f)
    return raw


def get_prompt(
    prompts: dict[str, dict[str, str]],
    variant: str,
    key: str,
    default_variant: str = "default",
) -> str:
    """Return the template for *key*, falling back to the default variant."""
    variant_prompts = prompts.get(variant, {})
    if key in variant_prompts:
        return variant_prompts[key]
    return prompts[default_variant][key]
