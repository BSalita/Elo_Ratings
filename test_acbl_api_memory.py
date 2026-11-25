"""The ACBL API's compact frame and its Arrow export give DuckDB the same answers.

See ACBL_API_MEMORY.md: every String column (and String fields inside
ParContracts) is stored Categorical, and DuckDB reads a cached Arrow export
instead of converting the Polars frame on every request.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest
from unittest.mock import patch

import duckdb
import polars as pl

import acbl_api_server as server

PAR = pl.List(
    pl.Struct(
        {
            "Level": pl.String,
            "Strain": pl.String,
            "Doubled": pl.String,
            "Pair_Direction": pl.String,
            "Result": pl.Int16,
        }
    )
)

QUERY = """
SELECT session_id,
       Declarer_Direction = 'N' AS is_n,
       strata_bucket = 'not-a-bucket' AS no_bucket,
       CASE WHEN BidSuit IS NOT NULL AND ParContracts IS NOT NULL
             AND list_contains(list_transform(ParContracts, lambda p: p.Strain), BidSuit)
            THEN 1 ELSE 0 END AS par_suit,
       Player_Name_N || ' - ' || Player_Name_S AS pair_name,
       ParContracts
FROM _full ORDER BY session_id
"""


def _board_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": ["2024-01-01 00:00:00", "2024-02-01 00:00:00", "2024-03-01 00:00:00"],
            "session_id": [1, 2, 3],
            "Player_Name_N": ["Ann", "Bob", "Ann"],
            "Player_Name_S": ["Cy", "Di", None],
            "BidSuit": ["S", "H", None],
            "Declarer_Direction": ["N", "E", "S"],
            "strata_bucket": ["open", "open", "0-299"],
            "Pct_NS": [0.5, 0.6, 0.4],
            "ParContracts": [
                [{"Level": "4", "Strain": "S", "Doubled": "", "Pair_Direction": "NS", "Result": 0}],
                [
                    {"Level": "3", "Strain": "N", "Doubled": "X", "Pair_Direction": "EW", "Result": -1},
                    {"Level": "4", "Strain": "H", "Doubled": "", "Pair_Direction": "EW", "Result": 0},
                ],
                None,
            ],
        },
        schema_overrides={"session_id": pl.Int64, "ParContracts": PAR},
    )


class CompactFrameTests(unittest.TestCase):
    def test_every_string_becomes_categorical(self) -> None:
        lf = _board_frame().lazy().with_columns(pl.col("Date").str.strptime(pl.Datetime))
        schema = lf.with_columns(server._dtype_shrink_exprs(dict(lf.collect_schema()))).collect_schema()
        self.assertEqual(schema["Date"], pl.Datetime("us"))
        self.assertEqual(schema["session_id"], pl.Int32)
        for col in ("Player_Name_N", "BidSuit", "Declarer_Direction", "strata_bucket"):
            self.assertEqual(schema[col], pl.Categorical, col)
        fields = {f.name: f.dtype for f in schema["ParContracts"].inner.fields}
        self.assertEqual(fields["Strain"], pl.Categorical)
        self.assertEqual(fields["Result"], pl.Int16)

    def test_arrow_export_answers_like_the_original_frame(self) -> None:
        original = _board_frame().with_columns(pl.col("Date").str.strptime(pl.Datetime))
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "acbl_club_elo_ratings.parquet"
            _board_frame().write_parquet(path)
            with patch.object(server, "DATA_ROOT", pathlib.Path(tmp)):
                server._SCHEMA_CACHE.clear()
                server._FRAME_CACHE.clear()
                server._ARROW_CACHE.clear()
                try:
                    server._load_full_frame("club")
                    table = server._arrow_table_for(str(path))
                finally:
                    server._SCHEMA_CACHE.clear()
                    server._FRAME_CACHE.clear()
                    server._ARROW_CACHE.clear()
        new_con = duckdb.connect()
        new_con.register("_full", table)
        old_con = duckdb.connect()
        old_con.register("_full", original)
        new = new_con.sql(QUERY).pl()
        old = old_con.sql(QUERY).pl()
        self.assertEqual(new.schema["pair_name"], pl.String)
        self.assertTrue(
            new.cast({"session_id": pl.Int64}).equals(old),
            f"\n{new}\n{old}",
        )


if __name__ == "__main__":
    unittest.main()
