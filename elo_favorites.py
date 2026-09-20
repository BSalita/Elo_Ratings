"""Load and run Elo favorites JSON (postmortem-style SQL + brace macros).

Python prepares DuckDB table ``self`` and a meta dict. Ranking recipes live in
``default.acbl.favorites.json`` / ``default.ffbridge.favorites.json``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

_SCRIPT_DIR = Path(__file__).resolve().parent

MACRO_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

ACBL_MACRO_KEYS = frozenset({
    "Top_N",
    "Min_Sessions",
    "Prior_Sessions",
    "Prior_Anchor",
    "Min_Skill_Z",
    "Rating_Method",
    "Rating_Type",
    "Elo_Suffix",
    "Elo_Col_N",
    "Elo_Col_S",
    "Elo_Col_E",
    "Elo_Col_W",
    "Elo_Col_NS",
    "Elo_Col_EW",
})

FFBRIDGE_MACRO_KEYS = frozenset({
    "Top_N",
    "Min_Games",
    "Prior_Sessions",
    "Rating_Type",
    "Elo_Kind",
    "Elo_Col_Name",
    "Title_Col_Name",
    "Pct_Col",
    "Pair_Elo_Col",
    "Pct_Expr",
})

ORG_MACRO_KEYS = {
    "acbl": ACBL_MACRO_KEYS,
    "ffbridge": FFBRIDGE_MACRO_KEYS,
}

_ORG_FILENAMES = {
    "acbl": "default.acbl.favorites.json",
    "ffbridge": "default.ffbridge.favorites.json",
}

_ORG_ENV = {
    "acbl": "ACBL_ELO_FAVORITES",
    "ffbridge": "FFBRIDGE_ELO_FAVORITES",
}


def process_sql_macros(sql: str, meta: dict[str, Any]) -> str:
    """Replace ``{Key}`` tokens from ``meta``. Missing or None values are left as-is."""
    for key, value in meta.items():
        if value is None:
            continue
        sql = sql.replace("{" + str(key) + "}", str(value))
    return sql


def normalize_from_self(sql: str) -> str:
    """Ensure the statement reads DuckDB table ``self``, matching postmortem."""
    query = sql.strip().rstrip(";")
    if 'from "self"' not in query.lower():
        query = query.replace("FROM self", 'FROM "self"')
        query = query.replace("from self", 'FROM "self"')
    if 'from "self"' not in query.lower():
        query = 'FROM "self" ' + query
    return query


def extract_macros(text: str) -> set[str]:
    return set(MACRO_RE.findall(text or ""))


def favorites_path(organization: str) -> Path:
    org = organization.lower().strip()
    if org not in _ORG_FILENAMES:
        raise ValueError(f"Unknown Elo favorites organization {organization!r}")
    env_file = os.environ.get(_ORG_ENV[org], "").strip()
    candidates = [Path(env_file)] if env_file else []
    candidates.append(_SCRIPT_DIR / _ORG_FILENAMES[org])
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"{_ORG_FILENAMES[org]} missing for {org}. "
        f"Tried: {', '.join(str(path) for path in candidates)}"
    )


def load_favorites(organization: str, *, path: Path | None = None) -> dict[str, Any]:
    source = path or favorites_path(organization)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source} is not a JSON object")
    return payload


def vetted_prompts(favorites: dict[str, Any]) -> dict[str, Any]:
    select_boxes = favorites.get("SelectBoxes") or {}
    vetted = select_boxes.get("Vetted_Prompts") if isinstance(select_boxes, dict) else None
    if not isinstance(vetted, dict) or not vetted:
        raise ValueError("favorites.SelectBoxes.Vetted_Prompts is missing")
    return vetted


def vetted_prompt_sql(favorites: dict[str, Any], prompt_id: str) -> str:
    entry = vetted_prompts(favorites).get(prompt_id)
    if not isinstance(entry, dict):
        raise KeyError(f"Unknown favorite id {prompt_id!r}")
    statements = []
    for item in entry.get("prompts") or []:
        if not isinstance(item, dict):
            continue
        sql = str(item.get("sql") or "").strip()
        prompt = str(item.get("prompt") or "").strip()
        if not sql or prompt.startswith("/"):
            continue
        statements.append(sql)
    if not statements:
        raise KeyError(f"Favorite {prompt_id!r} has no executable SQL")
    if len(statements) > 1:
        raise ValueError(f"Favorite {prompt_id!r} has multiple SQL statements")
    return statements[0]


def button_prompt_ids(
    favorites: dict[str, Any],
    button_id: str,
    meta: dict[str, Any],
) -> list[str]:
    buttons = favorites.get("Buttons") or {}
    if not isinstance(buttons, dict):
        raise ValueError("favorites.Buttons must be an object")
    button = buttons.get(button_id)
    if not isinstance(button, dict):
        raise KeyError(f"Unknown favorites button {button_id!r}")
    ids: list[str] = []
    for ref in button.get("prompts") or []:
        if not isinstance(ref, str) or not ref.startswith("@") or len(ref) < 2:
            raise ValueError(f"Button {button_id!r} prompt must be @id, got {ref!r}")
        ids.append(process_sql_macros(ref[1:], meta))
    if not ids:
        raise ValueError(f"Button {button_id!r} has no prompts")
    return ids


def flatten_favorites(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return vetted prompts that have executable SQL (MCP / GET /favorites)."""
    buttons = payload.get("Buttons") or {}
    id_to_buttons: dict[str, list[str]] = {}
    if isinstance(buttons, dict):
        for button_id, button in buttons.items():
            if not isinstance(button, dict):
                continue
            for ref in button.get("prompts") or []:
                if isinstance(ref, str) and ref.startswith("@") and len(ref) > 1:
                    id_to_buttons.setdefault(ref[1:], []).append(str(button_id))

    favorites: list[dict[str, Any]] = []
    for fav_id, entry in vetted_prompts(payload).items():
        if not isinstance(entry, dict):
            continue
        statements: list[dict[str, str]] = []
        for item in entry.get("prompts") or []:
            if not isinstance(item, dict):
                continue
            sql = str(item.get("sql") or "").strip()
            prompt = str(item.get("prompt") or "").strip()
            if not sql or prompt.startswith("/"):
                continue
            statements.append({"prompt": prompt, "sql": sql})
        if not statements:
            continue
        favorites.append(
            {
                "id": str(fav_id),
                "title": str(entry.get("title") or fav_id),
                "help": str(entry.get("help") or ""),
                "buttons": id_to_buttons.get(str(fav_id), []),
                "statements": statements,
            }
        )
    return favorites


def lint_favorites(organization: str, favorites: dict[str, Any] | None = None) -> list[str]:
    """Return human-readable lint errors. Empty list means the file is consistent."""
    org = organization.lower().strip()
    payload = favorites if favorites is not None else load_favorites(org)
    allowed = ORG_MACRO_KEYS[org]
    errors: list[str] = []
    vetted = vetted_prompts(payload)
    buttons = payload.get("Buttons") or {}
    if not isinstance(buttons, dict) or not buttons:
        errors.append("Buttons is missing")
        return errors
    for button_id, button in buttons.items():
        if not isinstance(button, dict):
            errors.append(f"Button {button_id!r} is not an object")
            continue
        for ref in button.get("prompts") or []:
            if not isinstance(ref, str) or not ref.startswith("@"):
                errors.append(f"Button {button_id!r} has a non-@ prompt {ref!r}")
                continue
            raw_id = ref[1:]
            macros = extract_macros(raw_id)
            unknown = macros - allowed
            if unknown:
                errors.append(f"Button {button_id!r} uses unknown macros {sorted(unknown)}")
            if not macros and raw_id not in vetted:
                errors.append(f"Button {button_id!r} references missing {raw_id!r}")
    for fav_id, entry in vetted.items():
        if not isinstance(entry, dict):
            errors.append(f"Vetted prompt {fav_id!r} is not an object")
            continue
        for item in entry.get("prompts") or []:
            if not isinstance(item, dict):
                continue
            sql = str(item.get("sql") or "")
            unknown = extract_macros(sql) - allowed
            if unknown:
                errors.append(f"{fav_id} SQL uses unknown macros {sorted(unknown)}")
    return errors


def run_sql(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    meta: dict[str, Any],
) -> tuple[pl.DataFrame, str]:
    substituted = normalize_from_self(process_sql_macros(sql, meta))
    leftover = extract_macros(substituted)
    if leftover:
        raise ValueError(f"Unsubstituted favorites macros: {sorted(leftover)}")
    return con.execute(substituted).pl(), substituted


def run_favorite(
    con: duckdb.DuckDBPyConnection,
    favorites: dict[str, Any],
    prompt_id: str,
    meta: dict[str, Any],
) -> tuple[pl.DataFrame, str]:
    return run_sql(con, vetted_prompt_sql(favorites, prompt_id), meta)


def run_favorite_on_df(
    df: pl.DataFrame,
    organization: str,
    prompt_id: str,
    meta: dict[str, Any],
) -> tuple[pl.DataFrame, str]:
    favorites = load_favorites(organization)
    con = duckdb.connect(config={"enable_external_access": "false"})
    try:
        con.register("self", df)
        return run_favorite(con, favorites, prompt_id, meta)
    finally:
        con.close()
