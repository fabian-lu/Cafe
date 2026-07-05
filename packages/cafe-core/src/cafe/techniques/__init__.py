"""Composed pipelines: build a :class:`Pipeline` from techniques, wire them with an
instrumented context, and let CAFE swap them as factors and attribute per-stage."""

from cafe.techniques.composed import PipelineView, pipeline, stage_report
from cafe.techniques.context import Context
from cafe.techniques.pipe import Pipeline
from cafe.techniques.registry import TechniqueSpec

__all__ = [
    "Pipeline",
    "PipelineView",
    "pipeline",
    "stage_report",
    "Context",
    "TechniqueSpec",
]
