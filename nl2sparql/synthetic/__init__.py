"""Synthetic data generation module for fine-tuning LLMs."""

from .generator import (
    SyntheticPair,
    SyntheticDataGenerator,
    generate_nl_variations,
)

__all__ = [
    "SyntheticPair",
    "SyntheticDataGenerator",
    "generate_nl_variations",
]
