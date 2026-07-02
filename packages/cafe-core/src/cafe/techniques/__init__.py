"""Composed pipelines: register techniques, wire them with an instrumented
context, and let CAFE swap them as factors and attribute per-stage."""

from cafe.techniques.composed import (
    Pipeline,
    composed,
    pipeline,
    stage_report,
    technique_factor,
)
from cafe.techniques.context import Context
from cafe.techniques.registry import REGISTRY, TechniqueSpec, names_for, stages, technique

__all__ = [
    "technique",
    "composed",
    "technique_factor",
    "stage_report",
    "pipeline",
    "Pipeline",
    "Context",
    "REGISTRY",
    "TechniqueSpec",
    "names_for",
    "stages",
]
