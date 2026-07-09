"""Tests for QueryEngine."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from floatchat.models import ChatResponse, MetadataRecord, ParsedIntent, SearchCriteria
from floatchat.query_engine.engine import QueryEngine


class TestQueryEngine:
    def _make_engine(self, records=None, df=None, figure=None):
        metadata = MagicMock()
        metadata.search = MagicMock(return_value=records or [])

        repository = MagicMock()
        ncd = MagicMock()
        repository.fetch = MagicMock(return_value=ncd)

        reader = MagicMock()
        reader.read = MagicMock(return_value=df if df is not None else pd.DataFrame())

        viz = MagicMock()
        viz.render = MagicMock(return_value=figure or {"data": [], "layout": {}})

        return QueryEngine(metadata, repository, reader, viz)

    def test_execute_no_records(self) -> None:
        engine = self._make_engine(records=[])
        intent = ParsedIntent(intent="profile_plot", variables=["DOXY"])
        response = engine.execute(intent)

        assert isinstance(response, ChatResponse)
        assert "No Argo profiles matched" in response.message
        assert response.figure is None

    def test_execute_success(self) -> None:
        records = [
            MetadataRecord(
                file="a.nc",
                date=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                latitude=0.0,
                longitude=0.0,
                ocean="I",
                profiler_type="x",
                institution="IF",
                parameters="PRES DOXY",
                parameter_data_mode="R A",
                date_update=datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
            )
        ]
        df = pd.DataFrame(
            {
                "profile_idx": [0],
                "level_idx": [0],
                "PRES": [10.0],
                "DOXY": [200.0],
            }
        )
        engine = self._make_engine(records=records, df=df, figure={"data": [1]})
        intent = ParsedIntent(intent="profile_plot", variables=["DOXY"])
        response = engine.execute(intent)

        assert response.figure is not None
        assert response.data_summary["matched_records"] == 1
        engine.metadata.search.assert_called_once()
        engine.repository.fetch.assert_called_once_with("a.nc")
        engine.reader.read.assert_called_once()
        engine.viz.render.assert_called_once()

    def test_execute_mixed_core_bio_searches_and_reads_separately(self) -> None:
        core_record = MetadataRecord(
            file="coriolis/6903091/profiles/R6903091_001.nc",
            date=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            latitude=10.0,
            longitude=65.0,
            ocean="I",
            profiler_type="x",
            institution="IF",
            parameters="",
            parameter_data_mode="",
            date_update=datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
        )
        bio_record = MetadataRecord(
            file="coriolis/6903091/profiles/BR6903091_001.nc",
            date=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            latitude=10.0,
            longitude=65.0,
            ocean="I",
            profiler_type="x",
            institution="IF",
            parameters="PRES DOXY",
            parameter_data_mode="R A",
            date_update=datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
        )

        metadata = MagicMock()
        metadata.search.side_effect = [[core_record], [bio_record]]

        repository = MagicMock()
        core_ncd = MagicMock()
        bio_ncd = MagicMock()
        repository.fetch.side_effect = [core_ncd, bio_ncd]

        reader = MagicMock()
        reader.read.side_effect = [
            pd.DataFrame({"profile_idx": [0], "level_idx": [0], "PRES": [10.0], "TEMP": [24.0]}),
            pd.DataFrame({"profile_idx": [0], "level_idx": [0], "PRES": [10.0], "DOXY": [205.0]}),
        ]

        viz = MagicMock()
        viz.render = MagicMock(return_value={"data": [1]})

        engine = QueryEngine(metadata, repository, reader, viz)
        intent = ParsedIntent(
            intent="profile_plot",
            variables=["TEMP", "DOXY"],
            region="arabian_sea",
        )
        response = engine.execute(intent)

        assert response.figure is not None
        assert metadata.search.call_count == 2
        assert metadata.search.call_args_list[0].args[0].parameters == ["TEMP"]
        assert metadata.search.call_args_list[1].args[0].parameters == ["DOXY"]
        reader.read.assert_any_call(core_ncd, ["TEMP"])
        reader.read.assert_any_call(bio_ncd, ["DOXY"])
