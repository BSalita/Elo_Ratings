# Elo_Ratings

Unofficial bridge-game Elo ratings. This is a flat Python repo containing **two independent products** plus shared libraries. There is no build step (pure Python), no test suite, and no linter config committed.

- **ACBL product** (client/server): `acbl_api_server.py` (FastAPI) + `streamlit_app_acbl_elo_ratings.py` (Streamlit UI that runs in API-only mode).
- **FFBridge product** (standalone): `streamlit_app_ffbridge_elo_ratings.py` (Streamlit UI that fetches live data from the public FFBridge Lancelot API).
- Shared code: `elo_common.py`, `elo_ffbridge_*.py`, `streamlitlib/`.

Deployment is Railway (Nixpacks); the `railway-*.toml` files hold the canonical start commands.

## Cursor Cloud specific instructions

Dependencies are Python-only (`requirements.txt`). The Cloud update script creates a `.venv` and installs into it, so **activate the venv first**: `source .venv/bin/activate`.

Notes for running/testing the services:

- **Streamlit is pinned to `1.53.1` on purpose** (see comment in `requirements.txt`): `streamlit >= 1.56.0` breaks `streamlit-aggrid` (grids render blank). Do not bump it.
- **ACBL API server** (`acbl_api_server.py`): dev run `uvicorn acbl_api_server:app --reload --port 8000`. `/health` works with no data. The `/acbl/report` and `/acbl/detail` endpoints read Parquet files from `./data/` (`acbl_club_elo_ratings.parquet`, `acbl_tournament_elo_ratings.parquet`) — or from Cloudflare R2 when `R2_BUCKET` is set. **These Parquet files are NOT in the repo** (the real ones are ~15 GB club / ~4 GB tournament). Without a data source those two endpoints return HTTP 500 (`FileNotFoundError`). To exercise them locally, supply a `./data/` parquet with the expected schema (seats N/E/S/W: `Player_ID_*`, `Player_Name_*`, `MasterPoints_*`, `Elo_R_*`, plus `Date, session_id, Pct_NS, Round, Board, DD_Tricks_Diff, Is_Par_Suit, Is_Par_Contract, Is_Sacrifice, Pair_Number_NS/EW`) or configure R2. The `report` query filters players by `min_sessions` (default 10 distinct `session_id`s).
- **ACBL Streamlit app** (`streamlit_app_acbl_elo_ratings.py`): requires `ACBL_API_BASE_URL` to be set (e.g. `http://localhost:8000`); it is API-only and reads no parquet itself. Start the API server first. Run: `ACBL_API_BASE_URL=http://localhost:8000 streamlit run streamlit_app_acbl_elo_ratings.py --server.port 8501`.
- **FFBridge Streamlit app** (`streamlit_app_ffbridge_elo_ratings.py`): fully standalone; needs outbound internet to `api-lancelot.ffbridge.fr` (public, no auth) and optionally `api.ffbridge.fr` (Classic API, needs `FFBRIDGE_BEARER_TOKEN`). Run: `streamlit run streamlit_app_ffbridge_elo_ratings.py --server.port 8502`. It caches fetched tournament data under `FFBRIDGE_CACHE_DIR` (code default is the relative `data/ffbridge`, i.e. `./data/ffbridge`; note `.env.example` sets the absolute `/data/ffbridge`, which is a Railway volume mount and is not writable locally — leave it unset or point it at a writable dir).
- `psycopg[binary]` and `DATABASE_URL`/`POSTGRES_*` env vars exist in `.env.example` but no code currently connects to Postgres; treat it as dormant/optional.
- Copy `.env.example` to `.env` if you need to set tokens/R2 creds (`python-dotenv` is loaded by the apps).
