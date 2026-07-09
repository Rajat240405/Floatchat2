"""Phase 21 regression tests for Variable Registry + Retrieval Planner."""

import pytest

from floatchat.variable_registry.registry import VariableRegistry
from floatchat.retrieval_planner.planner import RetrievalPlanner


def test_variable_registry_classifies_temp_as_core():
    classification = VariableRegistry.classify_variables(["TEMP"])
    assert classification["metadata_index"] == "synthetic"
    assert classification["profile_type"] == "S"


def test_variable_registry_classifies_doxy_as_bgc():
    classification = VariableRegistry.classify_variables(["DOXY"])
    assert classification["metadata_index"] == "bio"
    assert classification["profile_type"] == "B"


def test_variable_registry_mixed_core_bgc_routes_to_synthetic():
    classification = VariableRegistry.classify_variables(["TEMP", "DOXY"])
    assert classification["metadata_index"] == "synthetic"


def test_retrieval_planner_produces_reasoning():
    planner = RetrievalPlanner()
    plan = planner.plan(["TEMP"])
    assert "synthetic" in plan.reasoning.lower()