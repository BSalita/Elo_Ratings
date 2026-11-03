import polars as pl
import pytest

import compact_ffbridge_hierarchical_domains as compact


def test_fragments_with_reordered_columns_are_stacked(tmp_path):
    first = tmp_path / "a.parquet"
    second = tmp_path / "b.parquet"
    pl.DataFrame({"x": [1], "y": ["a"]}).write_parquet(first)
    pl.DataFrame({"y": ["b"], "x": [2]}).write_parquet(second)
    frame = compact._concat_fragments([first, second])
    assert frame.columns == ["x", "y"]
    assert frame["x"].to_list() == [1, 2]
    assert frame["y"].to_list() == ["a", "b"]


def test_fragments_with_different_columns_are_rejected(tmp_path):
    first = tmp_path / "a.parquet"
    second = tmp_path / "b.parquet"
    pl.DataFrame({"x": [1], "y": ["a"]}).write_parquet(first)
    pl.DataFrame({"x": [2], "z": ["b"]}).write_parquet(second)
    with pytest.raises(ValueError, match="extra=\\['z'\\] missing=\\['y'\\]"):
        compact._concat_fragments([first, second])
