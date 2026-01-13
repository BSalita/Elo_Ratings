# FFBridge Lancelot API Documentation

**Base URL:** `https://api-lancelot.ffbridge.fr`  
**Authentication:** Most endpoints are public. User-specific endpoints require separate auth (not the same token as api.ffbridge.fr).  
**Version:** 2.0.8  
**Used by:** `elo_ffbridge_lancelot.py` adapter

> **Note:** The Lancelot API appears to be an older/legacy system but contains more detailed historical data and board-level information including full deal records in PBN format. It provides 620+ sessions for Rondes de France alone vs ~100 in the Classic API.

---

## 1. Public Endpoints (No Authentication Required)

### `GET /public/version`
**Description:** Get current API version.

**Response:**
```json
{ "version": "2.0.8" }
```

---

### `GET /seasons/current`
**Description:** Get current bridge season information.

**Response:**
| Field | Description |
|-------|-------------|
| `id` | Season ID (e.g., 37) |
| `label` | Season name (e.g., "2025/2026") |
| `startDate` | Season start date |
| `endDate` | Season end date |
| `mandatoryLicenseStartDate` | When license required |
| `renewLicenseEndDate` | License renewal deadline |

---

### `GET /seasons/search`
**Description:** Search historical seasons.

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `maxSeason` | `current` or season ID |
| `maxPerPage` | Results per page |

**Response:** Paginated list of seasons with dates and IDs.

---

## 2. Competition Search APIs

### `GET /results/search/`
**Description:** Search for competitions/tournaments by type and season.

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `searchCompetitionType` | `clubSimultaneous`, `club`, etc. |
| `searchSeason` | `current` or season ID |
| `currentPage` | Page number |

**Example:** `/results/search/?searchCompetitionType=clubSimultaneous&searchSeason=current&currentPage=1`

**Response:**
| Field | Description |
|-------|-------------|
| `items[]` | Array of series/competitions |
| `items[].id` | Lancelot series ID |
| `items[].label` | Series name |
| `items[].migrationId` | Maps to api.ffbridge.fr series ID |
| `pagination` | Pagination info |

**Lancelot ID to Migration ID Mapping:**
| Lancelot ID | Migration ID | Name |
|-------------|--------------|------|
| 1 | 3 | Rondes de France |
| 2 | 4 | Trophées du Voyage |
| 3 | 5 | Roy René |
| 17 | 140 | Amour du Bridge |
| 25 | 384 | Simultanet |
| 27 | 386 | Simultané Octopus |
| 47 | 604 | Atout Simultané |
| 62 | 868 | Festival des Simultanés |

---

## 3. Simultaneous Competition APIs

### `GET /competitions/simultaneous/{lancelot_id}`
**Description:** Get series metadata.

**Response:**
```json
{
  "id": 1,
  "label": "Rondes de France",
  "migrationId": 3
}
```

---

### `GET /competitions/simultaneous/{lancelot_id}/sessions`
**Description:** Get all sessions (tournament dates) for a series.

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `currentPage` | Page number |
| `maxPerPage` | Results per page (default 80) |

**Example:** `/competitions/simultaneous/1/sessions?currentPage=1&maxPerPage=80`

**Response:**
| Field | Description |
|-------|-------------|
| `items[].id` | Session ID |
| `items[].label` | Session name (e.g., "Rondes de France 2026-01-06 Après-midi") |
| `items[].date` | Session date (ISO 8601) |
| `items[].moment` | Time of day ("A" = Après-midi/Afternoon) |
| `pagination` | Total: 620 sessions for Rondes de France |

---

## 4. Session & Ranking APIs

### `GET /results/sessions/{session_id}/simultaneousIds`
**Description:** Get all clubs (simultaneous IDs) that participated in a session.

**Response (array):**
| Field | Description |
|-------|-------------|
| `id` | Lancelot club ID (internal) |
| `ffbCode` | FFBridge club code (**use this for mapping**) |
| `label` | Club name (human-readable) |

**Club Code Mapping Notes:**
- The `simultaneousId` in ranking entries corresponds to `id` or `ffbCode` in this response
- Strip leading zeros from codes for consistent matching: `"05802079"` → `"5802079"`
- Build a mapping dict from multiple sessions for complete coverage
- Club names are only available through this endpoint, not in ranking data

---

### `GET /results/sessions/{session_id}/ranking`
**Description:** Get full ranking for a session (all clubs combined).

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `simultaneousId` | (Optional) Filter to specific club |
| `paginate` | `true` for paginated response |

**Response (array or paginated):**
| Field | Description |
|-------|-------------|
| `rank` | Position in ranking (with handicap) |
| `sessionScore` | Percentage score (**appears to be handicap-adjusted**) |
| `totalScore` | Final score (may include carryover) |
| `pe` | Performance points |
| `peBonus` | Bonus points (numeric, e.g., 10.0) |
| `orientation` | NS or EW |
| `section` | Section (A, B, etc.) |
| `tableNumber` | Table number |
| `simultaneousId` | Club code (**maps to ffbCode via `/simultaneousIds` endpoint**) |
| `handicapPercentage` | Handicap (if applicable) |
| `rankWithoutHandicap` | Non-handicapped rank (club rank) |
| `team.id` | Team ID |
| `team.label` | Pair names |
| `team.player1`, `team.player2` | Player details |

**Score Calculation Notes:**
- `sessionScore`/`totalScore` behaves like the handicap-adjusted score in practice
- To derive club percentage: `club_pct = sessionScore - (peBonus / 10.0)`
- `rank` is the handicapped national ranking
- `rankWithoutHandicap` is the club-level ranking (without handicap)
- Club codes in `simultaneousId` may have leading zeros that need stripping for consistent matching

**Handicap (IV) Formula:**

Based on empirical analysis comparing scratch vs handicapped scores from bridgeinter.net:

```
Handicap% = Scratch% + Bonus(Pair_IV)
```

Where:
- **Pair_IV** = sum of both players' IV (Indice de Valeur) ratings
- **Bonus** = an integer from 0 to ~15% based on pair strength
- **Strongest pairs** (IV sum ≥ 60): Bonus = 0%
- **Weaker pairs** (lower IV): Bonus increases proportionally

**Observed bonus values:** 0, 3, 4, 6, 7, 8, 9, 10, 11, 14 (percentage points)

**Examples from Simultané Octopus (Jan 12, 2026):**
| Players | Scratch | Handicap | Bonus | IV Level |
|---------|---------|----------|-------|----------|
| MARGAIL/TEISSIER | 66.10% | 66.10% | +0% | Very strong |
| LAUMOND/DAS | 62.41% | 65.41% | +3% | Strong |
| AMAR/SUISSA | 57.30% | 66.30% | +9% | Medium |
| L'HIRONDEL/VAYSSE | 59.67% | 73.67% | +14% | Weaker |

**Key insight:** Both scratch and handicapped percentages are calculated from the **same national field** (not local vs national). The only difference is the IV-based bonus applied to weaker pairs.

**Player fields:**
| Field | Description |
|-------|-------------|
| `id` | Lancelot person ID |
| `migrationId` | Maps to api.ffbridge.fr person ID |
| `ffbId` | License number |
| `firstName`, `lastName` | Name |
| `gender` | "M" or "F" |

---

## 5. Team APIs

### `GET /results/teams/{team_id}`
**Description:** Get team details and all rankings.

**Response:**
| Field | Description |
|-------|-------------|
| `player1` - `player8` | Player details (pairs use 1-2) |
| `orientation` | NS or EW |
| `homeGames[]` | Array of game IDs |
| `awayGames[]` | Array of game IDs |
| `rankings[]` | All session rankings for this team |
| `rankings[].rank` | Position |
| `rankings[].pe`, `rankings[].peBonus` | Points |
| `rankings[].round.session` | Session info |

---

### `GET /results/teams/{team_id}/session/{session_id}/scores`
**Description:** Get all board results for a team in a session.

**Response (array):** Full deal information for each board played.

| Field | Description |
|-------|-------------|
| `board.id` | Board ID |
| `board.boardNumber` | Board number (1-N) |
| `board.deal` | **PBN format deal** (e.g., "N:5.AJ64.9652.J852 KQ96.KQT.AJ743.7...") |
| `board.frequencies[]` | All results achieved on this board |
| `contract`, `declarer`, `lead` | Contract details |
| `ewNote`, `nsNote` | Matchpoint percentages |
| `ewScore`, `nsScore` | Bridge scores |

**Frequency object:**
| Field | Description |
|-------|-------------|
| `nsScore`, `ewScore` | Bridge score |
| `nsNote`, `ewNote` | Matchpoint percentage |
| `count` | Number of tables with this result |

---

## 6. Board/Deal APIs

### `GET /results/scores/board/{board_id}`
**Description:** Get all results for a specific board across all tables.

**Response (array):** Every result achieved on this board.
| Field | Description |
|-------|-------------|
| `boardNumber` | Board number |
| `contract`, `declarer`, `lead` | Contract info |
| `ewNote`, `nsNote` | Matchpoint percentages |
| `ewScore`, `nsScore` | Bridge scores |
| `lineup` | Full table information |
| `lineup.northPlayer`, `lineup.southPlayer` | NS pair |
| `lineup.eastPlayer`, `lineup.westPlayer` | EW pair |
| `lineup.segment.game.homeTeam`, `lineup.segment.game.awayTeam` | Full team info |

---

## 7. Group & Session Detail APIs

### `GET /competitions/groups/{group_id}`
**Description:** Get group/competition details.

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `context[]` | Add `result_status` and `result_data` for full info |

**Example:** `/competitions/groups/16713?context[]=result_status&context[]=result_data`

**Response includes:**
| Field | Description |
|-------|-------------|
| `phase.stade.organization` | Club/organization info |
| `phase.stade.season` | Season info |
| `phase.stade.competitionDivision` | Division details |
| `butler` | Butler scoring (true/false) |
| `displayLineup` | Show lineups |
| `location` | "ftf" (face-to-face) or online |

---

### `GET /competitions/groups/{group_id}/groupSessions`
**Description:** Get all sessions for a group.

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `context[]` | Add `result_status` for result availability |

**Response (array):**
| Field | Description |
|-------|-------------|
| `session.id` | Session ID |
| `session.label` | Session name |
| `session.simultaneous` | Series info |
| `session.rounds[]` | Round info with `hasResult` flag |
| `date` | Session date |
| `moment` | Time of day |

---

### `GET /competitions/sessions/{session_id}`
**Description:** Get session details with all participating clubs.

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `context[]` | Add `result_status` and `result_data` |

**Response includes:**
| Field | Description |
|-------|-------------|
| `format` | "pair" or "team" |
| `boardScoringType` | "percent", "imp", etc. |
| `groupSessions[]` | All club groups in session |
| `groupSessions[].group.phase.stade.organization` | Club info |

---

## 8. Division APIs

### `GET /competitions/divisions/searchList`
**Description:** Get all competition divisions.

**Response (array):**
| Field | Description |
|-------|-------------|
| `id` | Division ID |
| `label` | Division name (Expert, Performance, etc.) |
| `migrationId` | Legacy ID |

---

## 9. Authenticated Endpoints (Require Lancelot Auth)

These endpoints require Lancelot-specific authentication (different from api.ffbridge.fr):

| Endpoint | Description |
|----------|-------------|
| `GET /users/whoami` | Current user info |
| `GET /persons/me` | Current person profile |
| `GET /users/me/easi-token` | Get EASI token |
| `POST /persons/check` | Validate person |
| `GET /results/search/me` | Personal results history |
| `GET /results/sessions/{id}/ranking/{person_id}` | Personal ranking |

---

## Key Differences: Lancelot vs Main FFBridge API

| Feature | api.ffbridge.fr | api-lancelot.ffbridge.fr |
|---------|-----------------|--------------------------|
| Authentication | Bearer token (shared) | Separate auth system |
| Deal Data | Contract/result only | Full PBN deals + frequencies |
| Historical Data | Current season focus | All historical sessions |
| Session Count | ~100 per series | 620+ for Rondes de France |
| Board Details | Summary | Every score at every table |
| Player IDs | `person_id` | `id` (Lancelot) + `migrationId` (maps to FFBridge) |

---

## URL Templates

```python
urls = {
    # Search & Browse
    "results_search": "https://api-lancelot.ffbridge.fr/results/search/?searchCompetitionType={type}&searchSeason={season}&currentPage={page}",
    "simultaneous_series": "https://api-lancelot.ffbridge.fr/competitions/simultaneous/{lancelot_id}",
    "simultaneous_sessions": "https://api-lancelot.ffbridge.fr/competitions/simultaneous/{lancelot_id}/sessions?currentPage={page}&maxPerPage={limit}",
    
    # Rankings
    "session_ranking": "https://api-lancelot.ffbridge.fr/results/sessions/{session_id}/ranking",
    "session_ranking_club": "https://api-lancelot.ffbridge.fr/results/sessions/{session_id}/ranking?simultaneousId={club_code}",
    "session_clubs": "https://api-lancelot.ffbridge.fr/results/sessions/{session_id}/simultaneousIds",
    
    # Teams & Scores
    "team_details": "https://api-lancelot.ffbridge.fr/results/teams/{team_id}",
    "team_scores": "https://api-lancelot.ffbridge.fr/results/teams/{team_id}/session/{session_id}/scores",
    
    # Boards
    "board_results": "https://api-lancelot.ffbridge.fr/results/scores/board/{board_id}",
    
    # Groups & Sessions
    "group_details": "https://api-lancelot.ffbridge.fr/competitions/groups/{group_id}?context[]=result_status&context[]=result_data",
    "group_sessions": "https://api-lancelot.ffbridge.fr/competitions/groups/{group_id}/groupSessions?context[]=result_status",
    "session_details": "https://api-lancelot.ffbridge.fr/competitions/sessions/{session_id}?context[]=result_status&context[]=result_data",
}
```

---

## Example IDs for Testing

| Entity | ID | Description |
|--------|-----|-------------|
| Series (Lancelot) | 1 | Rondes de France |
| Series (Migration) | 3 | Rondes de France |
| Session | 247653 | Rondes de France 2026-01-06 |
| Team | 12273863 | Top pair in session 247653 |
| Board | 7404537 | Board 1 in session 247653 |
| Group | 16713 | Lagardere Racing Bridge club group |
| Club Code | 5000107 | Lagardere Racing Bridge |

---

## Workflow: Finding a Player's Board Results

1. **Find sessions for a series:**
   ```
   GET /competitions/simultaneous/1/sessions?currentPage=1&maxPerPage=80
   → Get session_id (e.g., 247653)
   ```

2. **Get ranking for session:**
   ```
   GET /results/sessions/247653/ranking
   → Find team by player name, get team_id (e.g., 12273863)
   ```

3. **Get team's board scores:**
   ```
   GET /results/teams/12273863/session/247653/scores
   → Returns all boards with PBN deals and frequencies
   ```

4. **Get specific board details:**
   ```
   GET /results/scores/board/7404537
   → Returns every result achieved on this board
   ```

---

## Usage in Elo Ratings App

The Lancelot API is used through the `elo_ffbridge_lancelot.py` adapter module.

**Key functions:**
```python
import elo_ffbridge_lancelot as api

# No authentication required
# Fetch all sessions (tournaments)
sessions = api.fetch_tournament_list(series_id="all")

# Fetch results for a specific session
results, was_cached = api.fetch_tournament_results(
    session_id="247653",
    tournament_date="2026-01-06",
    series_id=3
)

# Build club name mapping
unique_codes = ["5000107", "5802079"]
mapping = api.build_club_name_mapping(unique_codes, sessions)
```

**Cache Location:** `data/ffbridge/lancelot_cache/`

**Benefits over Classic API:**
- No authentication required (public access)
- More historical data (620+ sessions for Rondes de France)
- Full PBN deal data available for board-level analysis
