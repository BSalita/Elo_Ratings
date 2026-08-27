from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from datetime import date
from unittest import mock

import polars as pl

from ffbridge_quality_pipeline import (
    AuditReport,
    NoQualityRowsError,
    SessionAudit,
    audit_historical_cache,
    augment_raw_session,
    build_pair_sidecar,
    build_player_sidecar,
    flatten_team_scores,
    normalize_quality_frame,
    resolve_output_dir,
    stable_pair_id,
    write_quality_artifacts,
)


def _write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _training(path: pathlib.Path, session_ids: list[int]) -> None:
    pl.DataFrame({"session_id": session_ids}, schema={"session_id": pl.Int64}).write_parquet(
        path
    )


def _quality_input() -> pl.DataFrame:
    rows = [
        {
            "session_id": 10,
            "board_id": 100,
            "Board": 1,
            "group_id": 7,
            "team_id": 70,
            "Player_ID_N": 20,
            "Player_ID_E": 40,
            "Player_ID_S": 3,
            "Player_ID_W": 11,
            "Pair_Declarer_Direction": "NS",
            "Declarer_Direction": "N",
            "BidLvl": "4",
            "BidSuit": "S",
            "ParScore_NS": 420,
            "ParScore_EW": -420,
            "DDTricks": 10,
            "DDTricks_Diff": 1,
            "DDScore_1S_N": 80,
            "DDScore_2S_N": 110,
            "DDScore_3S_N": 140,
            "DDScore_4S_N": 420,
            "DDScore_5S_N": -50,
            "DDScore_6S_N": -100,
            "DDScore_7S_N": -150,
        },
        {
            "session_id": 10,
            "board_id": 101,
            "Board": 2,
            "group_id": 7,
            "team_id": 71,
            "Player_ID_N": 9,
            "Player_ID_E": 40,
            "Player_ID_S": 8,
            "Player_ID_W": 11,
            "Pair_Declarer_Direction": "EW",
            "Declarer_Direction": "E",
            "BidLvl": "5",
            "BidSuit": "H",
            "ParScore_NS": 100,
            "ParScore_EW": -100,
            "DDTricks": 10,
            "DDTricks_Diff": -2,
            "DDScore_1H_E": 50,
            "DDScore_2H_E": 50,
            "DDScore_3H_E": -50,
            "DDScore_4H_E": -50,
            "DDScore_5H_E": -100,
            "DDScore_6H_E": -200,
            "DDScore_7H_E": -300,
        },
    ]
    return pl.from_dicts(rows, infer_schema_length=None)


def _dates() -> pl.DataFrame:
    return pl.DataFrame(
        {"session_id": ["10"], "Date": [date(2024, 1, 2)]}
    )


class FFBridgeQualityPipelineTests(unittest.TestCase):
    def test_audit_respects_cutoff_and_compares_raw_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = pathlib.Path(temporary) / "source"
            source.mkdir()
            _training(source / "ffbridge_training_data_df.parquet", [1])
            _write_json(
                source / "competitions" / "sessions" / "1.json",
                {"id": 1, "groupSessions": [{"date": "2025-01-01T12:00:00+01:00"}]},
            )
            _write_json(
                source / "competitions" / "sessions" / "2.json",
                {"id": 2, "groupSessions": [{"date": "2024-12-31T12:00:00+01:00"}]},
            )
            _write_json(
                source / "competitions" / "sessions" / "3.json",
                {"id": 3, "groupSessions": [{"date": "2025-01-02T12:00:00+01:00"}]},
            )
            ranking = [{"team": {"id": 22, "player1": {}, "player2": {}}}]
            _write_json(source / "results" / "sessions" / "2" / "ranking.json", ranking)
            _write_json(
                source / "results" / "teams" / "22" / "session" / "2" / "scores.json",
                [],
            )

            report = audit_historical_cache(source)
            self.assertEqual([session.session_id for session in report.sessions], ["2", "1"])
            self.assertTrue(report.sessions[0].complete)
            self.assertFalse(report.sessions[0].in_training)
            self.assertTrue(report.sessions[1].in_training)
            self.assertEqual(report.to_dict()["summary"]["sessions_through_cutoff"], 2)

    def test_legacy_quality_derivation_and_pair_keys(self) -> None:
        result = normalize_quality_frame(_quality_input(), session_dates=_dates())
        self.assertEqual(result["DD_Tricks_Diff"].to_list(), [1, -2])
        self.assertEqual(result["Is_Par_Suit"].to_list(), [True, False])
        self.assertEqual(result["Is_Par_Contract"].to_list(), [True, True])
        self.assertEqual(result["Is_Sacrifice"].to_list(), [False, True])
        self.assertEqual(result["Pair_ID_NS"].to_list(), ["20_3", "8_9"])
        self.assertEqual(result["Pair_ID_EW"].to_list(), ["11_40", "11_40"])
        self.assertEqual(stable_pair_id("20", "3"), "20_3")
        self.assertEqual(result.schema["Player_ID_N"], pl.String)

    def test_identity_uses_migration_id_and_never_fabricates_alias(self) -> None:
        scores = [
            {
                "id": 500,
                "boardNumber": 1,
                "contract": "1NT",
                "declarer": "N",
                "result": "=",
                "board": {
                    "id": 50,
                    "boardNumber": 1,
                    "deal": None,
                    "frequencies": [
                        {
                            "nsScore": "40",
                            "ewScore": "60",
                            "nsNote": 40,
                            "ewNote": 60,
                            "count": 1,
                        }
                    ],
                    "dds": {"N": {"S": 9, "NT": 8}},
                },
                "lineup": {
                    "northPlayer": {"id": 1, "migrationId": 101},
                    "eastPlayer": {"id": 2},
                    "southPlayer": {"id": 3},
                    "westPlayer": None,
                },
            }
        ]
        frame, unmapped = flatten_team_scores("10", scores, {"2": "202"})
        row = frame.to_dicts()[0]
        self.assertEqual(row["Player_ID_N"], "101")
        self.assertEqual(row["Player_ID_E"], "202")
        self.assertEqual(row["Player_ID_S"], "3")
        self.assertIsNone(row["Player_ID_W"])
        self.assertEqual(row["DD_N_S"], 9)
        self.assertEqual(row["DD_N_N"], 8)
        self.assertEqual(row["board_frequencies"][0]["nsScore"], "")
        self.assertEqual(row["board_frequencies"][0]["ewScore"], "")
        self.assertEqual(unmapped, 1)

    def test_duplicate_endpoint_copies_are_collapsed(self) -> None:
        duplicated = pl.concat([_quality_input().head(1), _quality_input().head(1)])
        normalized = normalize_quality_frame(duplicated, session_dates=_dates())
        self.assertEqual(normalized.height, 1)

    def test_conflicting_duplicate_board_plays_are_rejected(self) -> None:
        original = _quality_input().head(1)
        conflicting = original.with_columns(
            pl.lit(-3, dtype=original.schema["DDTricks_Diff"]).alias(
                "DDTricks_Diff"
            )
        )
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            normalize_quality_frame(
                pl.concat([original, conflicting]),
                session_dates=_dates(),
            )

    def test_identityless_board_rows_are_excluded_before_deduplication(self) -> None:
        original = _quality_input().head(1)
        identityless = original.with_columns(
            *[
                pl.lit(
                    None, dtype=original.schema[f"Player_ID_{seat}"]
                ).alias(f"Player_ID_{seat}")
                for seat in ("N", "E", "S", "W")
            ],
            pl.lit(-3, dtype=original.schema["DDTricks_Diff"]).alias(
                "DDTricks_Diff"
            ),
        )

        normalized = normalize_quality_frame(
            pl.concat([original, identityless]),
            session_dates=_dates(),
        )

        self.assertEqual(normalized.height, 1)
        self.assertEqual(normalized["DD_Tricks_Diff"][0], original["DDTricks_Diff"][0])

    def test_session_without_mapped_player_identity_is_unsupported(self) -> None:
        original = _quality_input().head(1)
        identityless = original.with_columns(
            *[
                pl.lit(
                    None, dtype=original.schema[f"Player_ID_{seat}"]
                ).alias(f"Player_ID_{seat}")
                for seat in ("N", "E", "S", "W")
            ]
        )
        with self.assertRaisesRegex(NoQualityRowsError, "mapped player identity"):
            normalize_quality_frame(identityless, session_dates=_dates())

    def test_session_without_deal_and_contract_is_explicitly_unsupported(self) -> None:
        raw = pl.DataFrame(
            {
                "PBN": [None],
                "Contract": [None],
            },
            schema={"PBN": pl.String, "Contract": pl.String},
        )
        with self.assertRaisesRegex(NoQualityRowsError, "no board rows"):
            augment_raw_session(raw)

        with self.assertRaisesRegex(NoQualityRowsError, "contain no board rows"):
            flatten_team_scores("10", [], {})

    def test_player_and_pair_aggregates_rank_high_values_first(self) -> None:
        quality = normalize_quality_frame(_quality_input(), session_dates=_dates())
        players = build_player_sidecar(quality)
        pairs = build_pair_sidecar(quality)
        player_20 = players.filter(pl.col("player_id") == "20").row(0, named=True)
        player_9 = players.filter(pl.col("player_id") == "9").row(0, named=True)
        self.assertEqual(player_20["dd_tricks_diff_avg"], 1.0)
        self.assertEqual(player_9["dd_tricks_diff_avg"], -2.0)
        self.assertLess(
            player_20["DD_Tricks_Diff_Rank"], player_9["DD_Tricks_Diff_Rank"]
        )
        repeated_pair = pairs.filter(pl.col("pair_id") == "11_40").row(0, named=True)
        self.assertEqual(repeated_pair["Board_Rows"], 2)
        self.assertAlmostEqual(repeated_pair["dd_tricks_diff_avg"], -0.5)

    def test_atomic_metadata_is_written_last_and_describes_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            quality = normalize_quality_frame(_quality_input(), session_dates=_dates())
            report = AuditReport(
                source_dir=str(root / "source"),
                cutoff="2025-01-01",
                training_session_count=1,
                cached_session_count=1,
                sessions=(
                    SessionAudit(
                        session_id="10",
                        session_date="2024-01-02",
                        metadata_path="10.json",
                        in_training=True,
                        ranking_present=False,
                        expected_team_ids=(),
                        present_team_ids=(),
                        missing_team_ids=(),
                    ),
                ),
            )
            metadata = write_quality_artifacts(
                quality,
                root / "out",
                cutoff=date(2025, 1, 1),
                source_dir=root / "source",
                audit=report,
            )
            out = root / "out"
            parsed = json.loads((out / "ffbridge_quality_metadata.json").read_text())
            self.assertEqual(parsed, metadata)
            self.assertEqual(parsed["board_rows"], 2)
            self.assertTrue(all((out / name).is_file() for name in parsed["files"].values()))
            self.assertFalse(list(out.glob(".*.tmp")))

    def test_output_dir_requires_explicit_path_or_environment(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "--output-dir is required"):
                resolve_output_dir(None)
        self.assertEqual(resolve_output_dir(pathlib.Path("chosen")), pathlib.Path("chosen"))


if __name__ == "__main__":
    unittest.main()

