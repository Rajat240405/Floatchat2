"""Scientific Retrieval Planner – Phase 21.

Decides metadata index, profile type, and retrieval strategy
based on the Variable Registry.
"""

from dataclasses import dataclass
from typing import List, Optional

from floatchat.variable_registry.registry import VariableRegistry


@dataclass
class RetrievalPlan:
    """Result of the planning stage."""

    variables: List[str]
    metadata_index: str          # "bio" or "synthetic"
    profile_type: str            # "B", "S", or "R"
    requires_synthetic: bool
    requires_bio: bool
    reasoning: str


class RetrievalPlanner:
    """Decides the correct retrieval strategy for a set of variables."""

    def __init__(self):
        self.registry = VariableRegistry

    def plan(self, variables: List[str]) -> RetrievalPlan:
        classification = self.registry.classify_variables(variables)

        reasoning = (
            f"Requested: {variables}. "
            f"Core={classification['core']}, BGC={classification['bgc']}. "
            f"Strategy: {classification['strategy']}. "
            f"Index: {classification['metadata_index']}, "
            f"Profile type: {classification['profile_type']}"
        )

        return RetrievalPlan(
            variables=variables,
            metadata_index=classification["metadata_index"],
            profile_type=classification["profile_type"],
            requires_synthetic=classification["strategy"] == "synthetic",
            requires_bio=classification["strategy"] == "bio",
            reasoning=reasoning,
        )