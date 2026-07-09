"""Tests for InMemoryConversationManager."""

import pytest

from floatchat.conversation.memory import InMemoryConversationManager
from floatchat.models import ChatResponse, ConversationContext, ParsedIntent


class TestInMemoryConversationManager:
    def test_get_context_missing_returns_none(self) -> None:
        mgr = InMemoryConversationManager()
        assert mgr.get_context("nonexistent") is None

    def test_merge_context_no_session_returns_unchanged(self) -> None:
        mgr = InMemoryConversationManager()
        intent = ParsedIntent(intent="profile_plot", variables=["DOXY"])
        merged = mgr.merge_context(None, intent)
        assert merged.variables == ["DOXY"]

    def test_merge_context_no_existing_context_returns_unchanged(self) -> None:
        mgr = InMemoryConversationManager()
        intent = ParsedIntent(intent="profile_plot", variables=["DOXY"])
        merged = mgr.merge_context("sess-1", intent)
        assert merged.variables == ["DOXY"]

    def test_merge_context_fills_missing_variables(self) -> None:
        mgr = InMemoryConversationManager()
        # Seed context with previous DOXY query
        mgr.update_context(
            "sess-1",
            ParsedIntent(intent="profile_plot", variables=["DOXY"], region="arabian_sea"),
            ChatResponse(intent="profile_plot", message="ok"),
        )

        # New query has no variables — should inherit from context
        intent = ParsedIntent(intent="profile_plot", variables=[])
        merged = mgr.merge_context("sess-1", intent)
        assert merged.variables == ["DOXY"]
        assert merged.region == "arabian_sea"

    def test_merge_context_preserves_explicit_variables(self) -> None:
        mgr = InMemoryConversationManager()
        mgr.update_context(
            "sess-1",
            ParsedIntent(intent="profile_plot", variables=["DOXY"], region="arabian_sea"),
            ChatResponse(intent="profile_plot", message="ok"),
        )

        # New query explicitly asks for CHLA — should NOT be overwritten
        intent = ParsedIntent(intent="profile_plot", variables=["CHLA"])
        merged = mgr.merge_context("sess-1", intent)
        assert merged.variables == ["CHLA"]

    def test_merge_context_fills_missing_region(self) -> None:
        mgr = InMemoryConversationManager()
        mgr.update_context(
            "sess-1",
            ParsedIntent(intent="profile_plot", variables=["DOXY"], region="arabian_sea"),
            ChatResponse(intent="profile_plot", message="ok"),
        )

        intent = ParsedIntent(intent="profile_plot", variables=["CHLA"])
        merged = mgr.merge_context("sess-1", intent)
        assert merged.region == "arabian_sea"

    def test_merge_context_explicit_region_overrides(self) -> None:
        mgr = InMemoryConversationManager()
        mgr.update_context(
            "sess-1",
            ParsedIntent(intent="profile_plot", variables=["DOXY"], region="arabian_sea"),
            ChatResponse(intent="profile_plot", message="ok"),
        )

        intent = ParsedIntent(
            intent="profile_plot", variables=["CHLA"], region="bay_of_bengal"
        )
        merged = mgr.merge_context("sess-1", intent)
        assert merged.region == "bay_of_bengal"

    def test_merge_context_fills_float_id(self) -> None:
        mgr = InMemoryConversationManager()
        mgr.update_context(
            "sess-1",
            ParsedIntent(intent="profile_plot", variables=["DOXY"], float_id="3902490"),
            ChatResponse(intent="profile_plot", message="ok"),
        )

        intent = ParsedIntent(intent="profile_plot", variables=["TEMP"])
        merged = mgr.merge_context("sess-1", intent)
        assert merged.float_id == "3902490"

    def test_merge_context_fills_year(self) -> None:
        mgr = InMemoryConversationManager()
        mgr.update_context(
            "sess-1",
            ParsedIntent(
                intent="profile_plot", variables=["DOXY"], region="arabian_sea", year=2024
            ),
            ChatResponse(intent="profile_plot", message="ok"),
        )

        intent = ParsedIntent(intent="comparison_plot", year=2023)
        merged = mgr.merge_context("sess-1", intent)
        assert merged.year == 2023
        assert merged.variables == ["DOXY"]
        assert merged.region == "arabian_sea"

    def test_merge_context_fills_profile_number(self) -> None:
        mgr = InMemoryConversationManager()
        mgr.update_context(
            "sess-1",
            ParsedIntent(
                intent="profile_plot",
                variables=["DOXY"],
                float_id="3902490",
                profile_number=52,
            ),
            ChatResponse(intent="profile_plot", message="ok"),
        )

        intent = ParsedIntent(intent="profile_plot", variables=["TEMP"])
        merged = mgr.merge_context("sess-1", intent)
        assert merged.profile_number == 52
        assert merged.float_id == "3902490"

    def test_merge_context_respects_max_turns(self) -> None:
        mgr = InMemoryConversationManager(max_turns=2)
        mgr.update_context(
            "sess-1",
            ParsedIntent(intent="profile_plot", variables=["DOXY"]),
            ChatResponse(intent="profile_plot", message="ok"),
        )
        # Turn count is now 1
        mgr.update_context(
            "sess-1",
            ParsedIntent(intent="profile_plot", variables=["CHLA"]),
            ChatResponse(intent="profile_plot", message="ok"),
        )
        # Turn count is now 2 — at the limit

        intent = ParsedIntent(intent="profile_plot", variables=[])
        merged = mgr.merge_context("sess-1", intent)
        # Should still merge because 2 >= 2 is True, so it returns unchanged
        # Wait, the logic is: if turn_count >= max_turns, return unchanged
        # After 2 updates, turn_count = 2. max_turns = 2.
        # 2 >= 2 is True, so context is expired.
        assert merged.variables == []

    def test_update_context_general_query_preserves_data(self) -> None:
        mgr = InMemoryConversationManager()
        # First, a DATA_QUERY
        mgr.update_context(
            "sess-1",
            ParsedIntent(intent="profile_plot", variables=["DOXY"], region="arabian_sea"),
            ChatResponse(intent="profile_plot", message="ok"),
        )

        # Then a GENERAL_QUERY (no variables/region/float_id)
        mgr.update_context(
            "sess-1",
            ParsedIntent(intent="general_chat"),
            ChatResponse(intent="general_chat", message="Explanation here."),
        )

        ctx = mgr.get_context("sess-1")
        assert ctx is not None
        assert ctx.last_variables == ["DOXY"]
        assert ctx.last_region == "arabian_sea"
        assert ctx.turn_count == 2

    def test_clear_context(self) -> None:
        mgr = InMemoryConversationManager()
        mgr.update_context(
            "sess-1",
            ParsedIntent(intent="profile_plot", variables=["DOXY"]),
            ChatResponse(intent="profile_plot", message="ok"),
        )
        assert mgr.get_context("sess-1") is not None

        mgr.clear_context("sess-1")
        assert mgr.get_context("sess-1") is None

    def test_multiple_sessions_isolated(self) -> None:
        mgr = InMemoryConversationManager()
        mgr.update_context(
            "sess-a",
            ParsedIntent(intent="profile_plot", variables=["DOXY"], region="arabian_sea"),
            ChatResponse(intent="profile_plot", message="ok"),
        )
        mgr.update_context(
            "sess-b",
            ParsedIntent(intent="profile_plot", variables=["CHLA"], region="north_atlantic"),
            ChatResponse(intent="profile_plot", message="ok"),
        )

        ctx_a = mgr.get_context("sess-a")
        ctx_b = mgr.get_context("sess-b")

        assert ctx_a is not None
        assert ctx_b is not None
        assert ctx_a.last_variables == ["DOXY"]
        assert ctx_b.last_variables == ["CHLA"]
        assert ctx_a.last_region == "arabian_sea"
        assert ctx_b.last_region == "north_atlantic"
