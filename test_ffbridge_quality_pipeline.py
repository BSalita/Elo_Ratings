from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from datetime import date
from unittest import mock

import polars as pl
import requests

from ffbridge_quality_pipeline import (
    AuditReport,
    LancelotDDMismatchError,
    NoQualityRowsError,
    QUALITY_BOARD_COLUMNS,
    SessionAudit,
    _apply_lancelot_dd_audit,
    _attach_board_ev_columns,
    _attach_sd_ev_from_unique_deals,
    _audit_lancelot_dd_sample,
    _ddss_columns_from_table,
    _stale_dd_par_deals,
    _fragment_schema_is_current,
    _has_embedded_dd_table,
    _import_mlbridge,
    _official_hrs_cache,
    _select_lancelot_dd_audit_pbns,
    attach_embedded_dd_metrics,
    audit_historical_cache,
    augment_raw_session,
    build_pair_sidecar,
    build_player_sidecar,
    discover_session_metadata,
    fetch_missing_artifacts,
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
            "ParContract": ["4SN"],
            "ParScore_NS": 400,
            "ParScore_EW": -400,
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
            "ParContract": ["4SE"],
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
    def test_discover_session_metadata_writes_only_requested_dates(self) -> None:
        class FakeFFBridge:
            LANCELOT_TO_MIGRATION = {27: 386}

            @staticmethod
            def get_simultaneous_sessions_page(
                _series_id: int, *, page: int, **_kwargs: object
            ) -> dict[str, object]:
                pages = {
                    1: {
                        "items": [
                            {"id": 10, "date": "2025-12-31T00:00:00+01:00"},
                            {"id": 11, "date": "2026-01-02T00:00:00+01:00"},
                        ],
                        "pagination": {"has_next_page": True},
                    },
                    2: {
                        "items": [
                            {"id": 12, "date": "2026-02-01T00:00:00+01:00"},
                        ],
                        "pagination": {"has_next_page": False},
                    },
                }
                return pages[page]

        with tempfile.TemporaryDirectory() as temporary:
            source = pathlib.Path(temporary)
            with mock.patch(
                "ffbridge_quality_pipeline._import_ffbridge_lib",
                return_value=FakeFFBridge,
            ):
                writes = discover_session_metadata(
                    source,
                    start_date=date(2026, 1, 1),
                    cutoff=date(2026, 1, 31),
                    delay=0,
                )
            self.assertEqual(writes, 1)
            path = source / "competitions" / "sessions" / "11.json"
            self.assertTrue(path.is_file())
            self.assertEqual(json.loads(path.read_text())["series_id"], 386)

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

            report = audit_historical_cache(source, cutoff=date(2025, 1, 1))
            self.assertEqual([session.session_id for session in report.sessions], ["2", "1"])
            self.assertTrue(report.sessions[0].complete)
            self.assertFalse(report.sessions[0].in_training)
            self.assertTrue(report.sessions[1].in_training)
            self.assertEqual(report.to_dict()["summary"]["sessions_through_cutoff"], 2)

    def test_audit_requires_one_team_endpoint_per_table(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = pathlib.Path(temporary)
            _training(source / "ffbridge_training_data_df.parquet", [])
            _write_json(
                source / "competitions" / "sessions" / "1.json",
                {"id": 1, "date": "2026-01-02"},
            )
            _write_json(
                source / "results" / "sessions" / "1" / "ranking.json",
                [
                    {
                        "orientation": "NS",
                        "team": {"id": 10, "orientation": "NS"},
                    },
                    {
                        "orientation": "EW",
                        "team": {"id": 20, "orientation": "EW"},
                    },
                ],
            )

            report = audit_historical_cache(source, cutoff=date(2026, 1, 2))
            self.assertEqual(report.sessions[0].expected_team_ids, ("10",))

    def test_legacy_quality_derivation_and_pair_keys(self) -> None:
        result = normalize_quality_frame(_quality_input(), session_dates=_dates())
        self.assertEqual(result["DD_Tricks_Diff"].to_list(), [1, -2])
        self.assertEqual(result["Is_Par_Suit"].to_list(), [True, False])
        self.assertEqual(result["Par_Contract_Score_NS"].to_list(), [1, 1])
        self.assertEqual(result["Par_Contract_Score_EW"].to_list(), [-1, 1])
        self.assertEqual(result["Is_Sacrifice"].to_list(), [False, True])
        self.assertEqual(result["Sacrifice_Opportunity"].to_list(), [False, True])
        self.assertEqual(
            result["Pair_Declarer_Direction"].to_list(), ["NS", "EW"]
        )
        self.assertEqual(result["Declarer_Direction"].to_list(), ["N", "E"])
        self.assertEqual(result["Pair_ID_NS"].to_list(), ["20_3", "8_9"])
        self.assertEqual(result["Pair_ID_EW"].to_list(), ["11_40", "11_40"])
        self.assertEqual(stable_pair_id("20", "3"), "20_3")
        self.assertEqual(result.schema["Player_ID_N"], pl.String)

    def test_augmented_struct_par_contracts_supply_par_strains(self) -> None:
        frame = _quality_input().head(1).drop("ParContract").with_columns(
            pl.Series(
                "ParContracts",
                [
                    [
                        {
                            "Level": "4",
                            "Strain": "S",
                            "Doubled": "",
                            "Pair_Direction": "NS",
                            "Result": 0,
                        }
                    ]
                ],
            )
        )

        result = normalize_quality_frame(frame, session_dates=_dates())

        self.assertTrue(result["Is_Par_Suit"][0])

    def test_stale_v1_fragments_are_rejected_until_regenerated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            current = pathlib.Path(temporary) / "current.parquet"
            stale = pathlib.Path(temporary) / "stale.parquet"
            normalize_quality_frame(_quality_input(), session_dates=_dates()).write_parquet(
                current
            )
            pl.DataFrame(
                {
                    "session_id": ["1"],
                    "Is_Par_Contract": [True],
                    "DD_Tricks_Diff": [1],
                }
            ).write_parquet(stale)

            self.assertTrue(_fragment_schema_is_current(current))
            self.assertFalse(_fragment_schema_is_current(stale))
            self.assertEqual(
                list(normalize_quality_frame(_quality_input(), session_dates=_dates()).columns),
                list(QUALITY_BOARD_COLUMNS),
            )

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
            augment_raw_session(raw, max_sd_adds=0, dd_audit_rate=0)

        with self.assertRaisesRegex(NoQualityRowsError, "contain no board rows"):
            flatten_team_scores("10", [], {})

    def test_embedded_dd_table_derives_quality_columns_without_allaugmentations(
        self,
    ) -> None:
        raw = pl.DataFrame(
            {
                "PBN": ["N:QJ93.J5.AKT.J732 K.AQT873.873.A65 T762.K92.QJ64.K9 A854.64.952.QT84"],
                "Contract": ["4SN"],
                "Board": [1],
                "BidLvl": [4],
                "BidSuit": ["S"],
                "Declarer_Direction": ["N"],
                "Result": [0],
                "Dealer": ["N"],
                "Vul": ["None"],
                "DD_N_C": [6],
                "DD_N_D": [8],
                "DD_N_H": [5],
                "DD_N_N": [6],
                "DD_N_S": [9],
                "DD_E_C": [7],
                "DD_E_D": [5],
                "DD_E_H": [7],
                "DD_E_N": [7],
                "DD_E_S": [4],
                "DD_S_C": [6],
                "DD_S_D": [8],
                "DD_S_H": [5],
                "DD_S_N": [6],
                "DD_S_S": [9],
                "DD_W_C": [7],
                "DD_W_D": [5],
                "DD_W_H": [7],
                "DD_W_N": [7],
                "DD_W_S": [4],
            }
        )
        self.assertTrue(_has_embedded_dd_table(raw))
        with mock.patch(
            "ffbridge_quality_pipeline._full_mlbridge_augment",
            side_effect=AssertionError("AllAugmentations must not run"),
        ):
            out = augment_raw_session(raw, max_sd_adds=0, dd_audit_rate=0)
        self.assertEqual(out["DD_Tricks"][0], 9)
        self.assertEqual(out["Tricks"][0], 10)
        self.assertEqual(out["DDTricks_Diff"][0], 1)
        self.assertIn("ParScore_NS", out.columns)
        self.assertIn("DDScore_4S_N", out.columns)
        self.assertEqual(out["HCP_N"][0], 12)
        quality = normalize_quality_frame(
            out.with_columns(
                pl.lit("10").alias("session_id"),
                pl.lit("100").alias("board_id"),
                pl.lit("7").alias("group_id"),
                pl.lit("70").alias("team_id"),
                *[
                    pl.lit(str(value)).alias(f"Player_ID_{seat}")
                    for seat, value in zip(("N", "E", "S", "W"), (20, 40, 3, 11))
                ],
            ),
            session_dates=_dates(),
        )
        self.assertEqual(quality.height, 1)
        self.assertTrue(set(QUALITY_BOARD_COLUMNS).issubset(quality.columns))

    def test_attach_embedded_dd_metrics_sets_declarer_score(self) -> None:
        raw = pl.DataFrame(
            {
                "PBN": ["N:AKQJT9.AK.AK.AKQ J.QJT98.QJT9.JT9 876.765.8765.876 5432.432.432.5432"],
                "BidLvl": [6],
                "BidSuit": ["S"],
                "Declarer_Direction": ["N"],
                "Result": [0],
                "Dealer": ["N"],
                "Vul": ["None"],
                "DD_N_C": [13],
                "DD_N_D": [13],
                "DD_N_H": [13],
                "DD_N_N": [13],
                "DD_N_S": [13],
                "DD_E_C": [0],
                "DD_E_D": [0],
                "DD_E_H": [0],
                "DD_E_N": [0],
                "DD_E_S": [0],
                "DD_S_C": [13],
                "DD_S_D": [13],
                "DD_S_H": [13],
                "DD_S_N": [13],
                "DD_S_S": [13],
                "DD_W_C": [0],
                "DD_W_D": [0],
                "DD_W_H": [0],
                "DD_W_N": [0],
                "DD_W_S": [0],
            }
        )
        out = attach_embedded_dd_metrics(raw)
        self.assertEqual(out["DD_Tricks"][0], 13)
        self.assertGreater(out["ParScore_NS"][0], 0)

    def test_lancelot_dd_audit_sample_is_ten_percent(self) -> None:
        pbns = [f"deal-{index:02d}" for index in range(10)]
        self.assertEqual(len(_select_lancelot_dd_audit_pbns(pbns, 0.1)), 1)
        self.assertEqual(_select_lancelot_dd_audit_pbns(pbns, 0), [])
        self.assertEqual(_select_lancelot_dd_audit_pbns(pbns, 0.1), _select_lancelot_dd_audit_pbns(pbns, 0.1))

    def test_lancelot_dd_audit_accepts_matching_ddss_table(self) -> None:
        from endplay.types import Deal

        pbn = "N:QJ93.J5.AKT.J732 K.AQT873.873.A65 T762.K92.QJ64.K9 A854.64.952.QT84"
        _ff_lib, augment_lib = _import_mlbridge()
        table = augment_lib.solve_dd_for_deals([Deal(pbn)])[0]
        columns = _ddss_columns_from_table(table)
        frame = pl.DataFrame({"PBN": [pbn], **{name: [value] for name, value in columns.items()}})
        self.assertEqual(_audit_lancelot_dd_sample(frame, rate=1.0), 1)
        corrected, force_pbns = _apply_lancelot_dd_audit(frame, rate=1.0)
        self.assertEqual(int(corrected["DD_N_S"][0]), columns["DD_N_S"])
        self.assertEqual(force_pbns, set())

    def test_lancelot_dd_audit_rejects_mismatch(self) -> None:
        from endplay.types import Deal

        pbn = "N:QJ93.J5.AKT.J732 K.AQT873.873.A65 T762.K92.QJ64.K9 A854.64.952.QT84"
        _ff_lib, augment_lib = _import_mlbridge()
        table = augment_lib.solve_dd_for_deals([Deal(pbn)])[0]
        columns = _ddss_columns_from_table(table)
        columns["DD_N_S"] = (columns["DD_N_S"] + 1) % 14
        frame = pl.DataFrame({"PBN": [pbn], **{name: [value] for name, value in columns.items()}})
        with self.assertRaisesRegex(LancelotDDMismatchError, "DD_N_S"):
            _audit_lancelot_dd_sample(frame, rate=1.0)
        corrected, force_pbns = _apply_lancelot_dd_audit(frame, rate=1.0)
        self.assertEqual(
            int(corrected["DD_N_S"][0]), _ddss_columns_from_table(table)["DD_N_S"]
        )
        self.assertEqual(force_pbns, {pbn})

    def test_stale_dd_par_deals_detects_cache_disagreement(self) -> None:
        row = {
            "PBN": ["N:AK..."],
            "Dealer": ["N"],
            "Vul": ["None"],
            **{f"DD_{seat}_{suit}": [7] for seat in "NESW" for suit in "CDHSN"},
        }
        unique = pl.DataFrame(row)
        cache = pl.DataFrame({**row, "DD_N_S": [8], "ParScore": [100]})
        stale = _stale_dd_par_deals(unique, cache)
        self.assertEqual(stale.height, 1)
        same = _stale_dd_par_deals(unique, unique.with_columns(pl.lit(100).alias("ParScore")))
        self.assertEqual(same.height, 0)

    def test_board_ev_columns_use_unique_deal_summaries(self) -> None:
        frame = pl.DataFrame(
            {
                "session_id": ["10", "10"],
                "Board": [1, 1],
                "PBN": ["N:AK.AK.AK.AKQJT9 ...", "N:AK.AK.AK.AKQJT9 ..."],
                "Pair_Declarer_Direction": ["NS", "EW"],
                "Declarer_Direction": ["N", "E"],
                "BidLvl": [4, 2],
                "BidSuit": ["S", "H"],
                "Vul_NS": [False, False],
                "Vul_EW": [False, False],
                "Score_NS": [420, -110],
                "Score_EW": [-420, 110],
                "EV_NS_N_S_4_NV": [400.0, 400.0],
                "EV_EW_E_H_2_NV": [110.0, 110.0],
                "EV_NS_NV_Max": [450.0, 450.0],
                "EV_NS_V_Max": [500.0, 500.0],
                "EV_EW_NV_Max": [140.0, 140.0],
                "EV_EW_V_Max": [200.0, 200.0],
            }
        )
        out = _attach_board_ev_columns(frame)
        self.assertAlmostEqual(out["EV_Score_Declarer"][0], 400.0)
        self.assertAlmostEqual(out["EV_Score_Declarer"][1], 110.0)
        self.assertAlmostEqual(out["EV_Max_Declarer"][0], 450.0)
        self.assertAlmostEqual(out["EV_Max_Declarer"][1], 140.0)
        self.assertIn("MP_EV_Pct_Declarer", out.columns)
        self.assertIn("MP_EV_Max_Pct_Declarer", out.columns)
        self.assertGreaterEqual(out["MP_EV_Pct_Declarer"][0], 0.0)
        self.assertLessEqual(out["MP_EV_Pct_Declarer"][0], 1.0)

    def test_sd_cache_hit_skips_hand_record_augmenter(self) -> None:
        pbn = "N:QJ93.J5.AKT.J732 K.AQT873.873.A65 T762.K92.QJ64.K9 A854.64.952.QT84"
        frame = pl.DataFrame(
            {
                "PBN": [pbn, pbn],
                "Dealer": ["N", "N"],
                "Vul": ["None", "None"],
                "Board": [1, 1],
                "Pair_Declarer_Direction": ["NS", "EW"],
            }
        )
        cache_row: dict[str, object] = {"PBN": pbn, "Probs_Trials": 10}
        for pair in ("NS", "EW"):
            for declarer in pair:
                for strain in "SHDCN":
                    for taken in range(14):
                        cache_row[f"Probs_{pair}_{declarer}_{strain}_{taken}"] = 1.0 / 14
        cache = pl.DataFrame([cache_row])
        _ff_lib, augment_lib = _import_mlbridge()
        with (
            mock.patch.object(
                augment_lib,
                "AllHandRecordAugmentations",
                side_effect=AssertionError("must skip AllHandRecordAugmentations"),
            ),
            mock.patch.object(
                augment_lib,
                "identify_best_contracts_by_ev",
                side_effect=AssertionError("must skip identify_best_contracts_by_ev"),
            ),
        ):
            out, cache_out = _attach_sd_ev_from_unique_deals(
                frame,
                hrs_cache_df=cache,
                cache_file_path=None,
                sd_productions=10,
                max_sd_adds=None,
            )
        self.assertIn("EV_NS_N_S_4_NV", out.columns)
        self.assertIn("EV_NS_NV_Max", out.columns)
        self.assertEqual(out.height, 2)
        self.assertIsNotNone(out["EV_NS_N_S_4_NV"][0])
        self.assertNotIn("EV_NS_NV_Max", cache_out.columns)
        self.assertTrue(any(column.startswith("Probs_") for column in cache_out.columns))

    def test_official_hrs_cache_keeps_dd_par_probs_not_ev(self) -> None:
        pbn = "N:QJ93.J5.AKT.J732 K.AQT873.873.A65 T762.K92.QJ64.K9 A854.64.952.QT84"
        cache = pl.DataFrame(
            {
                "PBN": [pbn],
                "Dealer": ["N"],
                "Vul": ["None"],
                "DD_N_C": [6],
                "Probs_Trials": [10],
                "EV_NS_NV_Max": [450.0],
                "EV_NS_N_S_4_NV": [400.0],
            }
        )
        out = _official_hrs_cache(cache)
        self.assertIn("PBN", out.columns)
        self.assertIn("DD_N_C", out.columns)
        self.assertIn("ParScore", out.columns)
        self.assertIn("Probs_Trials", out.columns)
        self.assertNotIn("EV_NS_NV_Max", out.columns)
        self.assertNotIn("EV_NS_N_S_4_NV", out.columns)
        self.assertEqual(out["DD_N_C"][0], 6)
        self.assertEqual(out["Probs_Trials"][0], 10)

    def test_unique_deal_cache_stores_embedded_dd_and_par(self) -> None:
        pbn = "N:QJ93.J5.AKT.J732 K.AQT873.873.A65 T762.K92.QJ64.K9 A854.64.952.QT84"
        dd = {
            "DD_N_C": 6,
            "DD_N_D": 8,
            "DD_N_H": 5,
            "DD_N_N": 6,
            "DD_N_S": 9,
            "DD_E_C": 7,
            "DD_E_D": 5,
            "DD_E_H": 7,
            "DD_E_N": 7,
            "DD_E_S": 4,
            "DD_S_C": 6,
            "DD_S_D": 8,
            "DD_S_H": 5,
            "DD_S_N": 6,
            "DD_S_S": 9,
            "DD_W_C": 7,
            "DD_W_D": 5,
            "DD_W_H": 7,
            "DD_W_N": 7,
            "DD_W_S": 4,
        }
        frame = pl.DataFrame(
            {
                "PBN": [pbn],
                "Dealer": ["N"],
                "Vul": ["None"],
                "Board": [1],
                **{name: [value] for name, value in dd.items()},
            }
        )
        cache_row: dict[str, object] = {"PBN": pbn, "Probs_Trials": 10}
        for pair in ("NS", "EW"):
            for declarer in pair:
                for strain in "SHDCN":
                    for taken in range(14):
                        cache_row[f"Probs_{pair}_{declarer}_{strain}_{taken}"] = 1.0 / 14
        cache = pl.DataFrame([cache_row])
        _ff_lib, augment_lib = _import_mlbridge()
        with mock.patch.object(
            augment_lib,
            "estimate_sd_trick_distributions_for_df",
            side_effect=AssertionError("must not re-solve SD when Probs exist"),
        ):
            _out, cache_out = _attach_sd_ev_from_unique_deals(
                frame,
                hrs_cache_df=cache,
                cache_file_path=None,
                sd_productions=10,
                max_sd_adds=None,
            )
        assert cache_out is not None
        self.assertEqual(cache_out["DD_N_S"][0], 9)
        self.assertIsNotNone(cache_out["ParScore"][0])
        self.assertNotIn("EV_NS_NV_Max", cache_out.columns)

    def test_player_and_pair_aggregates_rank_high_values_first(self) -> None:
        quality = normalize_quality_frame(_quality_input(), session_dates=_dates())
        players = build_player_sidecar(quality)
        pairs = build_pair_sidecar(quality)
        player_20 = players.filter(pl.col("player_id") == "20").row(0, named=True)
        player_40 = players.filter(pl.col("player_id") == "40").row(0, named=True)
        self.assertEqual(player_20["dd_tricks_diff_avg"], 1.0)
        self.assertIsNone(
            players.filter(pl.col("player_id") == "9").item(
                0, "dd_tricks_diff_avg"
            )
        )
        self.assertEqual(player_40["dd_tricks_diff_avg"], -2.0)
        self.assertLess(
            player_20["DD_Tricks_Diff_Rank"],
            player_40["DD_Tricks_Diff_Rank"],
        )
        self.assertEqual(player_20["par_suit_rate"], 1.0)
        self.assertEqual(player_40["par_suit_rate"], 0.0)
        self.assertEqual(player_40["sacrifice_rate"], 1.0)
        repeated_pair = pairs.filter(pl.col("pair_id") == "11_40").row(0, named=True)
        self.assertEqual(repeated_pair["Board_Rows"], 2)
        self.assertEqual(repeated_pair["dd_tricks_diff_avg"], -2.0)
        self.assertEqual(repeated_pair["par_contract_rate"], 0.0)
        self.assertEqual(repeated_pair["par_suit_rate"], 0.0)
        self.assertEqual(repeated_pair["sacrifice_rate"], 1.0)

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
                unmapped_seat_count=99,
            )
            out = root / "out"
            parsed = json.loads((out / "ffbridge_quality_metadata.json").read_text())
            self.assertEqual(parsed, metadata)
            self.assertEqual(parsed["schema_version"], 2)
            self.assertIn("Sacrifice_Rate_Pct", parsed["metric_definitions"])
            self.assertEqual(parsed["raw_unmapped_seat_count"], 99)
            self.assertGreaterEqual(parsed["raw_unmapped_seat_count"], parsed["unmapped_seat_count"])
            self.assertEqual(parsed["board_rows"], 2)
            self.assertTrue(all((out / name).is_file() for name in parsed["files"].values()))
            self.assertFalse(list(out.glob(".*.tmp")))

    def test_fetch_missing_skips_read_timeout_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            report = AuditReport(
                source_dir=str(root),
                cutoff="2026-09-09",
                training_session_count=0,
                cached_session_count=1,
                sessions=(
                    SessionAudit(
                        session_id="99",
                        session_date="2026-01-02",
                        metadata_path="99.json",
                        in_training=False,
                        ranking_present=True,
                        expected_team_ids=("1", "2"),
                        present_team_ids=(),
                        missing_team_ids=("1", "2"),
                    ),
                ),
            )
            fake = mock.Mock()

            def scores(team_id: int, session_id: int, **kwargs: object) -> list[dict[str, int]]:
                if int(team_id) == 1:
                    raise requests.exceptions.ReadTimeout("timed out")
                return [{"board": 1}]

            fake.get_team_session_scores.side_effect = scores
            with mock.patch(
                "ffbridge_quality_pipeline._import_ffbridge_lib", return_value=fake
            ):
                writes = fetch_missing_artifacts(
                    report, timeout=0.01, max_attempts=1, workers=1, delay=0
                )
            self.assertEqual(writes, 1)
            written = root / "results" / "teams" / "2" / "session" / "99" / "scores.json"
            self.assertTrue(written.is_file())
            self.assertFalse(
                (root / "results" / "teams" / "1" / "session" / "99" / "scores.json").is_file()
            )

    def test_output_dir_requires_explicit_path_or_environment(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "--output-dir is required"):
                resolve_output_dir(None)
        self.assertEqual(resolve_output_dir(pathlib.Path("chosen")), pathlib.Path("chosen"))


if __name__ == "__main__":
    unittest.main()

