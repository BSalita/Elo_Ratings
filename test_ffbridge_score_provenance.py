from __future__ import annotations

import pathlib
import tempfile
from unittest.mock import patch

import elo_ffbridge_classic as classic
from elo_ffbridge_common import ffbridge_scoring_mode, fill_missing_score_ranks
import streamlit_app_ffbridge_elo_ratings as app
from streamlit_app_ffbridge_elo_ratings import process_tournaments_to_elo


def _result(**scores):
    return {
        "team_id": "10",
        "pair_id": "10",
        "player1_id": "101",
        "player2_id": "202",
        "player1_name": "One Player",
        "player2_name": "Two Player",
        "Club_Scratch_Pct": None,
        "Club_Handicap_Pct": None,
        "National_Scratch_Pct": None,
        "National_Handicap_Pct": None,
        "scoring_mode": "scratch",
        "iv_bonus": 0.0,
        "score_source": "national_official",
        "score_status": "official",
        "scratch_score_status": "unresolved",
        "handicap_score_status": "unresolved",
        "score_source_url": None,
        "rank": 1,
        "theoretical_rank": 1,
        "pe": 0,
        "pe_bonus": "0",
        "group_id": "20",
        "club_id": "30",
        "club_code": "30",
        "club_name": "Club",
        "player1_iv": None,
        "player2_iv": None,
        "pair_iv": None,
        **scores,
    }


class _Adapter:
    rows = []

    @classmethod
    def fetch_tournament_results(cls, *_args, **_kwargs):
        return cls.rows, True


def _replay(row, *, use_handicap):
    _Adapter.rows = [row]
    tournaments = [
        {
            "id": "225538",
            "name": "Test",
            "date": "2025-10-07",
            "series_id": 5,
        }
    ]
    with patch(
        "streamlit_app_ffbridge_elo_ratings._standardize_elo_frames",
        side_effect=lambda results, players, _use_handicap: (results, players),
    ):
        results, _players, _ratings, _stats = process_tournaments_to_elo(
            tournaments,
            _Adapter,
            use_handicap=use_handicap,
            show_progress=False,
        )
    return results.row(0, named=True)


def test_schedule_registry_classifies_only_first_tuesday_roy_rene() -> None:
    assert ffbridge_scoring_mode(5, "2025-10-07") == "handicap"
    assert ffbridge_scoring_mode(5, "2025-10-14") == "scratch"
    assert ffbridge_scoring_mode(384, "2025-10-14") == "handicap"
    assert ffbridge_scoring_mode(386, "2025-10-14") == "handicap"


def test_classic_does_not_turn_pe_bonus_into_handicap_percentage() -> None:
    scores = classic._canonical_classic_scores(
        {"percent": 62.5, "PE_bonus": 90},
        series_id=3,
        tournament_date="2025-10-08",
    )
    assert scores["National_Scratch_Pct"] == 62.5
    assert scores["National_Handicap_Pct"] is None


def test_club_rank_is_computed_within_club_and_preserves_ties() -> None:
    rows = [
        {"club_code": "A", "Club_Scratch_Pct": 60.0},
        {"club_code": "A", "Club_Scratch_Pct": 55.0},
        {"club_code": "A", "Club_Scratch_Pct": 55.0},
        {"club_code": "B", "Club_Scratch_Pct": 50.0},
    ]
    fill_missing_score_ranks(rows)
    assert [row["Club_Scratch_Rank"] for row in rows] == [1, 2, 2, 1]


def test_handicap_only_score_updates_only_handicap_elo() -> None:
    row = _replay(
        _result(
            National_Handicap_Pct=82.37,
            scoring_mode="handicap",
            handicap_score_status="official",
        ),
        use_handicap=True,
    )
    assert row["Pct_Used"] == 82.37
    assert row["Score_Source"] == "National_Handicap_Pct"
    assert row["player1_handicap_elo_after"] != row["player1_handicap_elo_before"]
    assert row["player1_scratch_elo_after"] == row["player1_scratch_elo_before"]


def test_scratch_only_score_updates_only_scratch_elo() -> None:
    row = _replay(
        _result(
            National_Scratch_Pct=62.5,
            scratch_score_status="official",
        ),
        use_handicap=False,
    )
    assert row["Pct_Used"] == 62.5
    assert row["Score_Source"] == "National_Scratch_Pct"
    assert row["player1_scratch_elo_after"] != row["player1_scratch_elo_before"]
    assert row["player1_handicap_elo_after"] == row["player1_handicap_elo_before"]


def test_rebuild_persists_result_player_and_pair_parquets() -> None:
    _Adapter.rows = [
        _result(
            National_Scratch_Pct=62.5,
            scratch_score_status="official",
        )
    ]
    tournaments = [
        {
            "id": "100",
            "name": "Test",
            "date": "2025-01-01",
            "series_id": 3,
        }
    ]
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        cache_paths = (
            root / "v7.results.parquet",
            root / "v7.players.parquet",
            root / "v7.meta.json",
        )
        pair_path = root / "v7.pairs.parquet"
        with (
            patch.object(app, "_FFBRIDGE_ELO_CACHE_DIR", root),
            patch.object(app, "_elo_cache_key", return_value="v7"),
            patch.object(app, "_elo_cache_paths", return_value=cache_paths),
            patch.object(app, "_elo_pair_cache_path", return_value=pair_path),
            patch.object(app, "_prune_old_elo_cache"),
        ):
            dataset = app.compute_and_persist_elo_dataset(
                _Adapter,
                tournaments,
                "FFBridge_Lancelot_API",
                fetch_iv=False,
            )

        assert cache_paths[0].exists()
        assert cache_paths[1].exists()
        assert pair_path.exists()
        assert cache_paths[2].exists()
        assert dataset["pairs_df"].row(0, named=True)["pair_id"] == "101_202"
