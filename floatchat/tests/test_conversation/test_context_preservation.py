"""Tests for conversation context preservation across general/data queries."""

from floatchat.conversation.memory import InMemoryConversationManager
from floatchat.models import ChatResponse, ParsedIntent


class TestContextPreservation:
    def test_general_query_preserves_data_context(self) -> None:
        mgr = InMemoryConversationManager()

        # First: DATA_QUERY stores variables, region, float, year
        mgr.update_context(
            "sess-1",
            ParsedIntent(
                intent="profile_plot",
                variables=["DOXY"],
                region="arabian_sea",
                float_id="3902490",
                year=2024,
            ),
            ChatResponse(intent="profile_plot", message="ok"),
        )

        # Second: GENERAL_QUERY should NOT erase data context
        mgr.update_context(
            "sess-1",
            ParsedIntent(intent="general_chat"),
            ChatResponse(intent="general_chat", message="Explanation."),
        )

        ctx = mgr.get_context("sess-1")
        assert ctx is not None
        assert ctx.last_variables == ["DOXY"]
        assert ctx.last_region == "arabian_sea"
        assert ctx.last_float_id == "3902490"
        assert ctx.last_year == 2024

    def test_data_query_overrides_explicitly(self) -> None:
        mgr = InMemoryConversationManager()

        mgr.update_context(
            "sess-1",
            ParsedIntent(
                intent="profile_plot",
                variables=["DOXY"],
                region="arabian_sea",
            ),
            ChatResponse(intent="profile_plot", message="ok"),
        )

        # Follow-up explicitly changes variable and region
        mgr.update_context(
            "sess-1",
            ParsedIntent(
                intent="profile_plot",
                variables=["CHLA"],
                region="bay_of_bengal",
            ),
            ChatResponse(intent="profile_plot", message="ok"),
        )

        ctx = mgr.get_context("sess-1")
        assert ctx.last_variables == ["CHLA"]
        assert ctx.last_region == "bay_of_bengal"

    def test_merge_after_general_query_uses_preserved_context(self) -> None:
        mgr = InMemoryConversationManager()

        # Data query
        mgr.update_context(
            "sess-1",
            ParsedIntent(
                intent="profile_plot",
                variables=["DOXY"],
                region="arabian_sea",
                float_id="3902490",
            ),
            ChatResponse(intent="profile_plot", message="ok"),
        )

        # General query (preserves context)
        mgr.update_context(
            "sess-1",
            ParsedIntent(intent="general_chat"),
            ChatResponse(intent="general_chat", message="Explanation."),
        )

        # New data query with only variable — should inherit region and float
        minimal = ParsedIntent(intent="profile_plot", variables=["CHLA"])
        merged = mgr.merge_context("sess-1", minimal)

        assert merged.variables == ["CHLA"]
        assert merged.region == "arabian_sea"
        assert merged.float_id == "3902490"

    def test_conversational_recovery_uses_context(self) -> None:
        mgr = InMemoryConversationManager()

        mgr.update_context(
            "sess-1",
            ParsedIntent(
                intent="profile_plot",
                variables=["DOXY"],
                region="arabian_sea",
            ),
            ChatResponse(intent="profile_plot", message="ok"),
        )

        # Empty follow-up — merge should fill from context
        empty = ParsedIntent(intent="profile_plot")
        merged = mgr.merge_context("sess-1", empty)

        assert merged.variables == ["DOXY"]
        assert merged.region == "arabian_sea"
