import json
import os

import polars as pl
import pytest

import ffbridge_report_service as reports


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
    assert table.row(0, named=True)["Sacrifice_Rank"] == 3


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
    pairs_path = tmp_path / reports.QUALITY_PAIRS_PATH.name
    metadata_path = tmp_path / reports.QUALITY_METADATA_PATH.name
    _quality("player_id", ["1"]).write_parquet(players_path)
    _quality("pair_id", ["1"]).write_parquet(pairs_path)
    metadata_path.write_text(
        json.dumps({"cutoff": "2026-08-01"}),
        encoding="utf-8",
    )

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
        json.dumps({"cutoff": "2026-08-01"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="par_suit_rate must be numeric"):
        reports.load_quality_sidecars(tmp_path)
