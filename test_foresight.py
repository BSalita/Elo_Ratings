import math

import duckdb
import numpy as np
import polars as pl
import pytest

import ffbridge_report_service as reports
import foresight as fs
from test_ffbridge_quality_reports import _player_results

ELO_COLUMNS = {"player_pattern": "Elo_R_{pos}", "pair_ns": "Elo_R_NS", "pair_ew": "Elo_R_EW"}


def _lead_population(count: int = 2000, seed: int = 7) -> pl.DataFrame:
    """Honest leaders: skill trend with Elo, small true spread, sampling noise."""
    rng = np.random.default_rng(seed)
    elo = rng.normal(1500.0, 200.0, count)
    leads = rng.integers(fs.MIN_LEADS, 2000, count)
    true_mean = -0.12 + 0.0002 * (elo - 1500.0) + rng.normal(0.0, 0.02, count)
    observed = true_mean + rng.normal(0.0, 1.0, count) / np.sqrt(leads)
    return pl.DataFrame(
        {
            "entity_id": [str(i) for i in range(count)],
            "elo": elo,
            "lead_n": leads,
            "lead_mean": observed,
            "lead_var": np.ones(count),
        }
    )


def test_peer_z_is_calibrated_removes_the_elo_trend_and_flags_an_outlier() -> None:
    population = _lead_population()
    planted = pl.DataFrame(
        {"entity_id": ["x"], "elo": [1500.0], "lead_n": [1000], "lead_mean": [0.25], "lead_var": [1.0]}
    )
    stats = fs.combine_stats({"lead": pl.concat([population, planted])})
    scored = fs.score_foresight(stats).join(stats.select("entity_id", "elo"), on="entity_id")
    honest = scored.filter(pl.col("entity_id") != "x")

    assert 0.9 < honest.get_column("Lead_Foresight_Z").std() < 1.1
    assert abs(honest.select(pl.corr("Lead_Foresight_Z", "elo")).item()) < 0.1
    assert scored.filter(pl.col("entity_id") == "x").item(0, "Lead_Foresight_Z") > 5
    # One component: the combined score is that component.
    assert scored.filter(pl.col("entity_id") == "x").item(0, "Foresight_Z") == pytest.approx(
        scored.filter(pl.col("entity_id") == "x").item(0, "Lead_Foresight_Z")
    )
    assert honest.get_column("Guess_Foresight_Z").null_count() == honest.height
    assert honest.get_column("Home_Edge_Z").null_count() == honest.height


def test_entities_below_the_minimum_sample_get_no_score() -> None:
    population = _lead_population(count=300)
    few = pl.DataFrame(
        {
            "entity_id": ["few"],
            "elo": [1500.0],
            "lead_n": [fs.MIN_LEADS - 1],
            "lead_mean": [3.0],
            "lead_var": [1.0],
        }
    )
    scored = fs.score_foresight(fs.combine_stats({"lead": pl.concat([population, few])}))
    row = scored.filter(pl.col("entity_id") == "few").row(0, named=True)
    assert row["Lead_Foresight_Z"] is None
    assert row["Foresight_Z"] is None


def test_foresight_z_is_the_stouffer_combination_of_available_components() -> None:
    rng = np.random.default_rng(3)
    count = 400
    ids = [str(i) for i in range(count)]
    elo = rng.normal(1500.0, 200.0, count)
    lead = pl.DataFrame(
        {
            "entity_id": ids,
            "elo": elo,
            "lead_n": [200] * count,
            "lead_mean": rng.normal(-0.12, 0.07, count),
            "lead_var": [1.0] * count,
        }
    )
    # Guess slope and its residual variance from sums of squares.
    slope = rng.normal(-0.5, 0.3, count)
    guess = pl.DataFrame(
        {
            "entity_id": ids[: count // 2],
            "elo": elo[: count // 2],
            "guess_n": [102] * (count // 2),
            "guess_sxx": [25.0] * (count // 2),
            "guess_syy": (25.0 * slope[: count // 2] ** 2 + 100.0),
            "guess_sxy": 25.0 * slope[: count // 2],
        }
    )
    scored = fs.score_foresight(fs.combine_stats({"lead": lead, "guess": guess}))
    both = scored.filter(pl.col("Guess_Foresight_Z").is_not_null())
    only_lead = scored.filter(pl.col("Guess_Foresight_Z").is_null())
    assert both.height == count // 2 and only_lead.height == count // 2
    expected = (both.get_column("Lead_Foresight_Z") + both.get_column("Guess_Foresight_Z")) / math.sqrt(2)
    assert both.get_column("Foresight_Z").to_list() == pytest.approx(expected.to_list())
    assert only_lead.get_column("Foresight_Z").to_list() == pytest.approx(
        only_lead.get_column("Lead_Foresight_Z").to_list()
    )


def test_attach_foresight_uses_leaderboard_elo_and_places_columns() -> None:
    population = _lead_population(count=300).drop("elo")
    stats = fs.combine_stats({"lead": population})
    leaderboard = pl.DataFrame(
        {
            "Rank": list(range(1, 301)),
            "Player_ID": list(range(300)),
            "Player_Elo": [2000.0 - i for i in range(300)],
            "Games": [20] * 300,
        }
    )
    table = fs.attach_foresight(
        leaderboard, stats, id_column="Player_ID", elo_column="Player_Elo", after="Player_Elo"
    )
    assert table.columns == ["Rank", "Player_ID", "Player_Elo", *fs.FORESIGHT_COLUMNS, "Games"]
    assert table.height == 300
    assert table.get_column("Lead_Foresight_Z").null_count() == 0
    # Without an Elo source nobody is scored.
    unscored = fs.attach_foresight(leaderboard, stats, id_column="Player_ID")
    assert unscored.get_column("Foresight_Z").null_count() == 300
    with pytest.raises(ValueError, match="already has"):
        fs.attach_foresight(table, stats, id_column="Player_ID")


def _acbl_self() -> pl.DataFrame:
    """Two sessions at club 1, one at club 2. Players 1-2 are NS, 3-4 are EW."""
    rows = []
    for session, club, pct_ns in ((10, 1, 0.7), (11, 1, 0.6), (20, 2, 0.4)):
        for board, declarer, sd, dd, guess, diff in (
            (1, "W", 0.5, 0, 1.0, -1),  # North leads; West declares
            (2, "N", 0.0, 1, 0.0, 0),  # East leads; North declares
            (3, "S", 0.25, 0, 2.0, 0),  # West leads; South declares
        ):
            rows.append(
                {
                    "session_id": session,
                    "Club": club,
                    "Board": board,
                    "Declarer_Direction": declarer,
                    "Declarer_Pair_Direction": "NS" if declarer in "NS" else "EW",
                    "Player_ID_N": 1,
                    "Player_ID_S": 2,
                    "Player_ID_E": 3,
                    "Player_ID_W": 4,
                    "SD_Lead_Cost": sd,
                    "DD_Lead_Cost": dd,
                    "SC_Guess_Declarer": guess,
                    "DD_Tricks_Diff": diff,
                    "Pct_NS": pct_ns,
                    "Elo_E_NS": 0.5,
                    "Elo_E_EW": 0.5,
                    "Elo_R_N": 1600.0,
                    "Elo_R_S": 1500.0,
                    "Elo_R_E": 1400.0,
                    "Elo_R_W": float("nan"),
                    "Elo_R_NS": 1550.0,
                    "Elo_R_EW": 1350.0,
                }
            )
    return pl.DataFrame(rows, schema_overrides={"DD_Lead_Cost": pl.UInt8, "SD_Lead_Cost": pl.Float32})


def _acbl_stats(rating_type: str, columns: set[str] | None = None) -> tuple[dict, dict]:
    frame = _acbl_self()
    con = duckdb.connect()
    try:
        con.register("self", frame)
        stats, status = fs.acbl_foresight_stats(
            con,
            columns=set(frame.columns) if columns is None else columns,
            rating_type=rating_type,
            elo_columns=ELO_COLUMNS,
        )
    finally:
        con.close()
    return {row["entity_id"]: row for row in stats.to_dicts()}, status


def test_acbl_player_statistics_credit_leader_declarer_and_home_club() -> None:
    players, status = _acbl_stats("Players")
    assert status == {"lead": "ok", "guess": "ok", "home": "ok"}
    # Leads: North against West's contract, East against North's, West against South's.
    assert players["1"]["lead_n"] == 3 and players["1"]["lead_mean"] == pytest.approx(0.5)
    assert players["3"]["lead_n"] == 3 and players["3"]["lead_mean"] == pytest.approx(-1.0)
    assert players["4"]["lead_n"] == 3 and players["4"]["lead_mean"] == pytest.approx(0.25)
    assert players["2"]["lead_n"] is None
    # Declarations: West, North and South each declared one board per session.
    assert players["4"]["guess_n"] == 3 and players["2"]["guess_n"] == 3
    assert players["3"]["guess_n"] is None
    # Home club 1 (two sessions), away club 2. NS result = Pct_NS - 0.5.
    assert players["1"]["home_n"] == 2 and players["1"]["away_n"] == 1
    assert players["1"]["home_mean"] == pytest.approx(0.15)
    assert players["1"]["away_mean"] == pytest.approx(-0.10)
    assert players["3"]["home_mean"] == pytest.approx(-0.15)
    assert players["3"]["away_mean"] == pytest.approx(0.10)
    # Elo: finite values only (West's NaN is ignored).
    assert players["1"]["elo"] == pytest.approx(1600.0)
    assert players["4"]["elo"] is None


def test_acbl_pair_statistics_use_the_leaderboard_pair_ids() -> None:
    pairs, _status = _acbl_stats("Pairs")
    assert set(pairs) == {"1-2", "3-4"}
    # NS defended West's contract; EW defended North's and South's.
    assert pairs["1-2"]["lead_n"] == 3 and pairs["1-2"]["lead_mean"] == pytest.approx(0.5)
    assert pairs["3-4"]["lead_n"] == 6 and pairs["3-4"]["lead_mean"] == pytest.approx(-0.375)
    assert pairs["1-2"]["guess_n"] == 6 and pairs["3-4"]["guess_n"] == 3
    assert pairs["1-2"]["home_n"] == 2 and pairs["1-2"]["away_n"] == 1
    assert pairs["1-2"]["elo"] == pytest.approx(1550.0)


def test_acbl_metrics_without_their_columns_are_reported_not_computed() -> None:
    columns = set(_acbl_self().columns) - {"SD_Lead_Cost", "SC_Guess_Declarer", "Club"}
    players, status = _acbl_stats("Players", columns)
    assert players == {}
    assert all(state.startswith("unavailable") for state in status.values())
    assert "ACBL_SD_LEAD_SAMPLES" in status["lead"]
    assert "ACBL_SUITC" in status["guess"]
    assert "acbl_elo_ratings_create.py" in status["home"]


def _ff_quality_boards() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "session_id": ["10", "10", "20"],
            "Declarer_Direction": ["W", "N", "S"],
            "Player_ID_N": ["1", "1", "1"],
            "Player_ID_E": ["2", "2", "2"],
            "Player_ID_S": ["3", "3", "3"],
            "Player_ID_W": ["4", "4", "4"],
            "Pair_ID_NS": ["1_3", "1_3", "1_3"],
            "Pair_ID_EW": ["2_4", "2_4", "2_4"],
            "DD_Lead_Cost": [0, 1, 0],
            "SD_Lead_Cost": [0.5, 0.0, 0.25],
        },
        schema_overrides={"DD_Lead_Cost": pl.UInt8, "SD_Lead_Cost": pl.Float32},
    ).lazy()


def test_ffbridge_leads_go_to_the_leader_and_the_defending_pair() -> None:
    players = {r["entity_id"]: r for r in fs.ffbridge_lead_stats(_ff_quality_boards(), "Players").to_dicts()}
    assert players["1"]["lead_mean"] == pytest.approx(0.5)  # North leads against West
    assert players["2"]["lead_mean"] == pytest.approx(-1.0)  # East leads against North
    assert players["4"]["lead_mean"] == pytest.approx(0.25)  # West leads against South
    assert "3" not in players
    pairs = {r["entity_id"]: r for r in fs.ffbridge_lead_stats(_ff_quality_boards(), "Pairs").to_dicts()}
    assert pairs["1_3"]["lead_n"] == 1 and pairs["2_4"]["lead_n"] == 2


def test_ffbridge_guesses_go_to_the_declarer_and_the_declaring_pair() -> None:
    boards = pl.DataFrame(
        {
            "session_id": ["10"] * 4,
            "Declarer": ["7", "7", "7", "9"],
            "Dummy": ["5", "5", "5", "8"],
            "Tricks": [9, 8, 10, 7],
            "DD_Tricks": [9, 9, 10, 7],
            "SC_Guess_Declarer": [0.0, 1.0, 0.0, None],
        }
    ).lazy()
    players = {r["entity_id"]: r for r in fs.ffbridge_guess_stats(boards, "Players").to_dicts()}
    assert set(players) == {"7"}
    assert players["7"]["guess_n"] == 3
    # x = (0, 1, 0), y = (0, -1, 0): the slope is -1.
    assert players["7"]["guess_sxy"] / players["7"]["guess_sxx"] == pytest.approx(-1.0)
    pairs = fs.ffbridge_guess_stats(boards, "Pairs")
    assert pairs.get_column("entity_id").to_list() == ["5_7"]


def _ff_results() -> pl.DataFrame:
    rows = []
    for club, pct in (("A", 60.0), ("A", 62.0), ("B", 50.0)):
        rows.append(
            {
                "pair_id": "1_2",
                "player1_id": "1",
                "player2_id": "2",
                "club_code": club,
                "National_Scratch_Pct": pct,
                "Club_Scratch_Pct": None,
            }
        )
    return pl.DataFrame(rows, schema_overrides={"Club_Scratch_Pct": pl.Float64})


def test_ffbridge_home_statistics_split_the_most_played_club_from_the_rest() -> None:
    players = {r["entity_id"]: r for r in fs.ffbridge_home_stats(_ff_results(), "Players").to_dicts()}
    assert set(players) == {"1", "2"}
    assert players["1"]["home_n"] == 2 and players["1"]["away_n"] == 1
    assert players["1"]["home_mean"] == pytest.approx(61.0)
    assert players["1"]["away_mean"] == pytest.approx(50.0)
    pairs = fs.ffbridge_home_stats(_ff_results(), "Pairs")
    assert pairs.get_column("entity_id").to_list() == ["1_2"]
    with pytest.raises(ValueError, match="club_code"):
        fs.ffbridge_home_stats(_ff_results().drop("club_code"), "Players")


def test_ffbridge_leaderboard_gains_foresight_columns_only_when_asked() -> None:
    plain, _sql, _anchor = reports.run_top_players_favorite(_player_results(), top_n=4, min_games=10)
    assert not set(fs.FORESIGHT_COLUMNS) & set(plain.columns)

    stats = fs.combine_stats(
        {
            "lead": pl.DataFrame(
                {
                    "entity_id": ["1", "2", "3", "4"],
                    "lead_n": [400, 400, 400, 10],
                    "lead_mean": [0.3, -0.1, -0.12, -0.11],
                    "lead_var": [1.0, 1.0, 1.0, 1.0],
                }
            )
        }
    )
    table, _sql, _anchor = reports.run_top_players_favorite(
        _player_results(), top_n=4, min_games=10, foresight_df=stats
    )
    assert table.columns[-4:] == list(fs.FORESIGHT_COLUMNS)
    assert table.get_column("Player_ID").to_list() == plain.get_column("Player_ID").to_list()
    assert table.get_column("Player_Elo").to_list() == plain.get_column("Player_Elo").to_list()
    by_id = {row["Player_ID"]: row for row in table.to_dicts()}
    assert by_id["1"]["Lead_Foresight_Z"] > by_id["2"]["Lead_Foresight_Z"]
    assert by_id["4"]["Lead_Foresight_Z"] is None  # below MIN_LEADS


def test_ffbridge_foresight_status_reports_missing_board_files(tmp_path) -> None:
    stats, status = fs.ffbridge_foresight_stats(
        results=_ff_results(),
        rating_type="Players",
        quality_boards_path=tmp_path / "missing.parquet",
        club_boards_path=None,
    )
    assert status["lead"].startswith("unavailable")
    assert status["guess"].startswith("unavailable")
    assert status["home"] == "ok"
    assert stats.height == 2
