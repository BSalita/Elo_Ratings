import json
import os
from datetime import date

import polars as pl
import pytest

import ffbridge_report_service as reports
from ffbridge_quality_pipeline import SCHEMA_VERSION


def _players() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": ["1", "2", "3", "4"],
            "player_name": ["One", "Two", "Three", "Four"],
            "scratch_elo": [1400.0, 1300.0, 1200.0, 1100.0],
            "handicap_elo": [1400.0, 1300.0, 1200.0, 1100.0],
            "elo_rating": [1400.0, 1300.0, 1200.0, 1100.0],
            "games_played": [20, 20, 20, 20],
            "provisional_games": [0, 0, 0, 0],
            "avg_scratch_pct": [60.0, 58.0, 56.0, 54.0],
            "avg_handicap_pct": [60.0, 58.0, 56.0, 54.0],
            "avg_iv_bonus": [0.0, 0.0, 0.0, 0.0],
            "avg_percentage": [60.0, 58.0, 56.0, 54.0],
            "stdev_percentage": [1.0, 1.0, 1.0, 1.0],
        }
    )


def _quality(id_column: str, ids: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            id_column: ids,
            "par_suit_rate": [0.9, 0.1, 0.8][: len(ids)],
            "par_contract_rate": [0.9, 0.1, 0.8][: len(ids)],
            "sacrifice_rate": [0.2, 0.8, 0.5][: len(ids)],
            "dd_tricks_diff_avg": [0.9, 0.1, 0.8][: len(ids)],
        }
    )


def _board_quality() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "session_id": ["10", "20"],
            "board_id": ["100", "200"],
            "Board": [1, 1],
            "group_id": ["1", "2"],
            "team_id": ["10", "20"],
            "Date": [date(2026, 1, 1), date(2026, 2, 1)],
            "Pair_Declarer_Direction": ["NS", "EW"],
            "Declarer_Direction": ["N", "E"],
            "Player_ID_N": ["1", "1"],
            "Player_ID_E": ["2", "2"],
            "Player_ID_S": ["3", "3"],
            "Player_ID_W": ["4", "4"],
            "Pair_ID_NS": ["1_3", "1_3"],
            "Pair_ID_EW": ["2_4", "2_4"],
            "Is_Par_Suit": [True, False],
            "Is_Sacrifice": [False, True],
            "Sacrifice_Opportunity": [False, True],
            "Par_Contract_Score_NS": [1, -1],
            "Par_Contract_Score_EW": [-1, 1],
            "DD_Tricks_Diff": [2, -2],
        }
    ).with_columns(pl.col("Date").cast(pl.Date))


def _write_quality_cache(path) -> None:
    _board_quality().write_parquet(path / reports.QUALITY_BOARDS_PATH.name)
    _quality("player_id", ["1"]).write_parquet(
        path / reports.QUALITY_PLAYERS_PATH.name
    )
    _quality("pair_id", ["1_3"]).write_parquet(
        path / reports.QUALITY_PAIRS_PATH.name
    )
    (path / reports.QUALITY_METADATA_PATH.name).write_text(
        json.dumps({"cutoff": "2026-08-01", "schema_version": SCHEMA_VERSION}),
        encoding="utf-8",
    )


def test_player_quality_uses_full_qualifying_population_and_preserves_elo() -> None:
    quality = _quality("player_id", ["1", "2", "3"])
    table, _sql, _anchor = reports.show_top_players(
        _players(),
        top_n=2,
        min_games=10,
        quality_df=quality,
    )

    assert table.get_column("Player_ID").to_list() == ["1", "2"]
    assert table.get_column("Rank").to_list() == [1, 2]
    assert table.get_column("Player_Elo").to_list() == [1400, 1300]
    assert table.get_column("Quality_Rank").to_list() == [1, 3]
    assert table.row(0, named=True)["Par_Suit_Rate_Pct"] == 90.0
    assert table.row(0, named=True)["Par_Contract_Rate_Pct"] == 95.0
    assert table.row(0, named=True)["Sacrifice_Rank"] == 3


def test_public_metric_definitions_describe_role_and_filter_scope() -> None:
    definitions = reports.QUALITY_METRIC_DEFINITIONS

    assert "declarer only" in definitions["DD_Tricks_Diff_Avg"]["attribution"]
    assert "directional DD score" in definitions["Par_Contract_Rate_Pct"]["formula"]
    assert "all declarations" in definitions["Par_Suit_Rate_Pct"]["formula"]
    assert "negative-par declarations" in definitions["Sacrifice_Rate_Pct"]["formula"]
    assert "leaderboard filters" in definitions["filter_scope"]


def test_unmatched_player_quality_values_and_ranks_stay_null() -> None:
    table, _sql, _anchor = reports.show_top_players(
        _players(),
        top_n=4,
        min_games=10,
        quality_df=_quality("player_id", ["1", "2", "3"]),
    )

    unmatched = table.filter(pl.col("Player_ID") == "4").row(0, named=True)
    for column in reports._QUALITY_OUTPUT_COLUMNS:
        assert unmatched[column] is None
    assert table.get_column("Rank").to_list() == [1, 2, 3, 4]


def test_pair_quality_rank_can_differ_from_elo_rank() -> None:
    results = pl.DataFrame(
        {
            "pair_id": ["B", "C", "A"],
            "pair_name": ["Bee", "See", "Aye"],
            "player1_id": ["1", "3", "5"],
            "player2_id": ["2", "4", "6"],
            "scratch_pair_elo": [1400.0, 1300.0, 1200.0],
            "handicap_pair_elo": [1400.0, 1300.0, 1200.0],
            "Club_Scratch_Pct": [None, None, None],
            "Club_Handicap_Pct": [None, None, None],
            "National_Scratch_Pct": [60.0, 58.0, 56.0],
            "National_Handicap_Pct": [60.0, 58.0, 56.0],
            "iv_bonus": [0.0, 0.0, 0.0],
            "score_status": ["official", "official", "official"],
            "date": ["2026-01-03", "2026-01-03", "2026-01-03"],
        }
    )
    quality = pl.DataFrame(
        {
            "pair_id": ["A", "B", "C"],
            "par_suit_rate": [0.9, 0.8, 0.7],
            "par_contract_rate": [0.9, 0.8, 0.7],
            "sacrifice_rate": [0.1, 0.9, 0.5],
            "dd_tricks_diff_avg": [0.9, 0.8, 0.7],
        }
    )

    table, _sql, _anchor = reports.show_top_pairs(
        results,
        top_n=3,
        min_games=1,
        quality_df=quality,
    )

    assert table.get_column("Pair_ID").to_list() == ["B", "C", "A"]
    assert table.get_column("Rank").to_list() == [1, 2, 3]
    assert table.get_column("Quality_Rank").to_list() == [2, 3, 1]
    assert table.filter(pl.col("Pair_ID") == "A").item(0, "Sacrifice_Rank") == 3


def test_missing_quality_cache_is_explicitly_unavailable(tmp_path) -> None:
    players, pairs, status = reports.load_quality_sidecars(tmp_path)

    assert players is None
    assert pairs is None
    assert status == {
        "status": "unavailable",
        "reason": "quality_cache_missing",
        "cutoff": None,
    }


def test_quality_cache_reuses_frames_then_reloads_on_mtime_change(tmp_path) -> None:
    players_path = tmp_path / reports.QUALITY_PLAYERS_PATH.name
    _write_quality_cache(tmp_path)

    players_before, pairs_before, status = reports.load_quality_sidecars(tmp_path)
    players_cached, pairs_cached, _cached_status = reports.load_quality_sidecars(tmp_path)
    assert players_cached is players_before
    assert pairs_cached is pairs_before
    assert status["cutoff"] == "2026-08-01"

    updated = _quality("player_id", ["1"]).with_columns(
        pl.lit(0.42).alias("par_suit_rate")
    )
    previous_mtime = players_path.stat().st_mtime_ns
    updated.write_parquet(players_path)
    os.utime(players_path, ns=(previous_mtime + 1_000_000, previous_mtime + 1_000_000))

    players_after, _pairs_after, _status_after = reports.load_quality_sidecars(tmp_path)
    assert players_after is not players_before
    assert players_after.item(0, "par_suit_rate") == 0.42


def test_incompatible_quality_cache_raises(tmp_path) -> None:
    _board_quality().write_parquet(tmp_path / reports.QUALITY_BOARDS_PATH.name)
    pl.DataFrame(
        {
            "player_id": ["1"],
            "par_suit_rate": ["not numeric"],
            "par_contract_rate": [0.5],
            "sacrifice_rate": [0.5],
            "dd_tricks_diff_avg": [0.0],
        }
    ).write_parquet(tmp_path / reports.QUALITY_PLAYERS_PATH.name)
    _quality("pair_id", ["1"]).write_parquet(
        tmp_path / reports.QUALITY_PAIRS_PATH.name
    )
    (tmp_path / reports.QUALITY_METADATA_PATH.name).write_text(
        json.dumps({"cutoff": "2026-08-01", "schema_version": SCHEMA_VERSION}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="par_suit_rate must be numeric"):
        reports.load_quality_sidecars(tmp_path)


def test_filtered_quality_uses_only_selected_sessions(tmp_path) -> None:
    _write_quality_cache(tmp_path)
    selected = pl.DataFrame({"tournament_id": ["10"]})

    players, pairs, status = reports.load_filtered_quality_sidecars(
        selected, tmp_path
    )

    assert status["filtered_session_count"] == 1
    assert status["filtered_board_rows"] == 1
    assert players is not None
    assert pairs is not None
    declarer = players.filter(pl.col("player_id") == "1").row(0, named=True)
    defender = players.filter(pl.col("player_id") == "2").row(0, named=True)
    assert declarer["dd_tricks_diff_avg"] == 2.0
    assert defender["dd_tricks_diff_avg"] is None
    assert declarer["par_suit_rate"] == 1.0
    assert pairs.filter(pl.col("pair_id") == "1_3").item(
        0, "dd_tricks_diff_avg"
    ) == 2.0


def test_filtered_quality_uses_only_selected_teams_within_session(tmp_path) -> None:
    _write_quality_cache(tmp_path)
    _board_quality().with_columns(
        pl.lit("10").alias("session_id")
    ).write_parquet(tmp_path / reports.QUALITY_BOARDS_PATH.name)
    selected = pl.DataFrame(
        {"tournament_id": ["10"], "team_id": ["10"]}
    )

    players, _pairs, status = reports.load_filtered_quality_sidecars(
        selected, tmp_path
    )

    assert status["filtered_board_rows"] == 1
    assert players is not None
    assert players.filter(pl.col("player_id") == "1").item(
        0, "dd_tricks_diff_avg"
    ) == 2.0


def test_schema_v1_quality_cache_is_rejected(tmp_path) -> None:
    _write_quality_cache(tmp_path)
    (tmp_path / reports.QUALITY_METADATA_PATH.name).write_text(
        json.dumps({"cutoff": "2026-08-01", "schema_version": 1}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version must be 2"):
        reports.load_quality_sidecars(tmp_path)
