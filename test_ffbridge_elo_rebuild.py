from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import elo_ffbridge_lancelot as lancelot
import streamlit_app_ffbridge_elo_ratings as app


def _session(session_id: str, days_ago: int) -> dict:
    day = datetime.now().date() - timedelta(days=days_ago)
    return {"id": session_id, "date": f"{day.isoformat()}T00:00:00+02:00"}


class EloRebuildNeededTests(unittest.TestCase):
    def test_retries_recent_empty_session_even_if_marked_processed(self):
        tournaments = [_session("304741", 1), _session("304731", 13)]
        needed = app._elo_ids_still_needed(
            eligible_ids={"304741", "304731"},
            processed_ids={"304741", "304731"},
            missing_ids={"304741"},
            all_tournaments=tournaments,
        )
        self.assertEqual(needed, {"304741"})

    def test_does_not_retry_old_empty_session(self):
        tournaments = [_session("100001", 40)]
        needed = app._elo_ids_still_needed(
            eligible_ids={"100001"},
            processed_ids=set(),
            missing_ids={"100001"},
            all_tournaments=tournaments,
        )
        self.assertEqual(needed, set())

    def test_new_list_session_is_needed(self):
        tournaments = [_session("304741", 0), _session("304731", 13)]
        needed = app._elo_ids_still_needed(
            eligible_ids={"304741", "304731"},
            processed_ids={"304731"},
            missing_ids=set(),
            all_tournaments=tournaments,
        )
        self.assertEqual(needed, {"304741"})


class SessionListCacheTests(unittest.TestCase):
    def test_uses_stale_list_when_live_fetch_is_empty(self):
        stale = [{"id": 304741, "date": "2026-09-14T00:00:00+02:00"}]

        def fake_load(_cache_dir, _name, max_age_hours=None, series_id=None):
            del series_id
            if max_age_hours is None:
                return [dict(stale[0])]
            return None

        with patch.object(lancelot, "load_from_disk_cache", side_effect=fake_load), \
             patch.object(lancelot, "lancelot_get", return_value=None), \
             patch.object(lancelot, "save_to_disk_cache"):
            sessions = lancelot._fetch_sessions_for_series(
                62, 868, force_refresh=True
            )
        self.assertEqual([session["id"] for session in sessions], [304741])
        self.assertEqual(sessions[0]["series_id"], 868)
