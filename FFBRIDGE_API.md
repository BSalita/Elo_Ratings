# FFBridge API Documentation

**Base URL:** `https://api.ffbridge.fr/api/v1`  
**Authentication:** Bearer token in header: `Authorization: Bearer {token}`

---

## 1. User/Member APIs

### `GET /users/my/infos`
**Description:** Get current authenticated user's profile and settings.

**Response includes:**
| Field | Description |
|-------|-------------|
| `id` | User ID |
| `username` | License number |
| `person.id` | Person ID (use for other APIs) |
| `person.firstname`, `person.lastname` | Name |
| `person.license_number` | FFBridge license number |
| `person.email` | Email address |
| `person.max_iv` | Maximum IV rating achieved |
| `person.bbo_pseudo` | BBO username |
| `person.funbridge_pseudo` | Funbridge username |
| `iv.iv`, `iv.code`, `iv.label` | Current IV rating |
| `organization.organization_id` | Home club ID |
| `organization.last_club` | Home club name |

---

### `GET /members/{person_id}`
**Description:** Get public member profile by person ID.

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `person_id` | The person's ID (e.g., 597539) |

**Response includes:**
| Field | Description |
|-------|-------------|
| `id` | Person ID |
| `firstname`, `lastname` | Name |
| `license_number` | FFBridge license number |
| `iv.iv`, `iv.label`, `iv.code` | Current IV rating |
| `licence.organization_id` | Home club ID |
| `licence.organization_name` | Home club name |
| `licence.organization_code` | Club code |
| `licence.committee_id`, `licence.committee_name` | Regional committee |
| `licence.season`, `licence.status` | License status |
| `address` | Location info |
| `bbo_pseudo`, `funbridge_pseudo` | Online platform usernames |

---

## 2. Results APIs

### `GET /licensee-results/results/person/{person_id}`
**Description:** Get all tournament results for a specific person.

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `person_id` | The person's ID |
| `date` | Filter: `all`, or date range |
| `place` | 0 = all places |
| `type` | 0 = all types |

**Example:** `/licensee-results/results/person/597539?date=all&place=0&type=0`

**Response (array):**
| Field | Description |
|-------|-------------|
| `type` | Tournament type (e.g., "simultane") |
| `tournament_id` | Tournament ID |
| `tournament_name`, `title` | Tournament name |
| `date` | Tournament date |
| `moment_label` | Session (e.g., "Après-midi") |
| `organization_id`, `organization_name` | Club where played |
| `rank` | Final ranking |
| `result` | Percentage score |
| `pe`, `pe_bonus` | Performance points |
| `simultaneous_type_id` | Series ID |

---

### `GET /licensee-results/results/organization/{org_id}`
**Description:** Get all tournaments played at a specific club/organization.

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `org_id` | Organization/club ID |
| `date` | Filter: `all`, or date range |
| `person_organization_id` | Filter by club |
| `place` | 0 = all places |
| `type` | 0 = all types |

**Example:** `/licensee-results/results/organization/1634?date=all&person_organization_id=1634&place=0&type=0`

**Response (array):** List of tournaments with dates, names, organization info.

---

### `GET /licensee-results/results/person/{person_id}?person_organization_id={org_id}`
**Description:** Get a person's results filtered to a specific club only.

**Example:** `/licensee-results/results/person/597539?date=all&person_organization_id=1634&place=0&type=0`

---

## 3. Tournament APIs

### `GET /simultaneous/{series_id}/tournaments`
**Description:** Get list of tournaments for a simultaneous series.

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `series_id` | Series ID (3, 4, 5, 140, 384, 386, 604, 868) |

**Series IDs:**
| ID | Name |
|----|------|
| 3 | Rondes de France |
| 4 | Trophées du Voyage |
| 5 | Roy René |
| 140 | Armour du Bridge |
| 384 | Simultané |
| 386 | Simultané Octopus |
| 604 | Atout Simultané |
| 868 | Festival des Simultanés |

---

### `GET /simultaneous-tournaments/{tournament_id}`
**Description:** Full tournament details with all teams/results.

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `tournament_id` | Tournament ID (e.g., 34271) |
| `organization_id` | (Optional) Filter to specific club's pairs only |

**Response includes:**
| Field | Description |
|-------|-------------|
| `id` | Tournament ID |
| `name` | Tournament name |
| `date` | Tournament date |
| `deal_count` | Number of boards |
| `nb_total_pairs` | Total pairs in tournament |
| `is_imp` | IMP scoring (true/false) |
| `teams[]` | Array of team results |

**Team object:**
| Field | Description |
|-------|-------------|
| `id` | Team ID (use for deal APIs) |
| `ranking` | Final rank |
| `theoretical_ranking` | Handicapped rank |
| `percent` | Percentage score |
| `PE`, `PE_bonus` | Performance points |
| `orientation` | NS or EO |
| `section_name`, `table_number` | Table assignment |
| `organization.id`, `organization.name`, `organization.code` | Club info |
| `players[]` | Array of player details |

**Player object:**
| Field | Description |
|-------|-------------|
| `id` | Person ID |
| `firstname`, `lastname` | Name |
| `license_number` | FFBridge license |
| `gender` | M./Mme |

---

## 4. Deal/Hand APIs

### `GET /simultaneous-tournaments/{tournament_id}/teams/{team_id}/dealsNumber`
**Description:** Get number of deals in tournament for a team.

**Response:**
```json
{ "nb_deals": 32 }
```

---

### `GET /simultaneous-tournaments/{tournament_id}/teams/{team_id}/roadsheets`
**Description:** Get complete roadsheet with all rounds and deals played by a team.

**Response includes:**
| Field | Description |
|-------|-------------|
| `roadsheets[]` | Array of rounds |
| `roadsheets[].teams.players[]` | Player names |
| `roadsheets[].teams.opponents[]` | Opponent names |
| `roadsheets[].deals[]` | Array of deals in this round |

**Deal object in roadsheet:**
| Field | Description |
|-------|-------------|
| `dealNumber` | Board number |
| `teamNote` | Team's matchpoint score |
| `opponentsNote` | Opponents' matchpoint score |
| `teamScore` | Team's bridge score (if declarer) |
| `opponentsScore` | Opponents' score (if declarer) |
| `teamOrientation` | NS or EO |
| `teamAvgNote`, `opponentsAvgNote` | Percentage scores |
| `contract` | Contract (e.g., "3SA") |
| `declarant` | N/S/E/O |
| `first_card` | Opening lead (e.g., "4T") |
| `result` | Result (e.g., "+1", "=", "-2") |

---

### `GET /simultaneous-tournaments/{tournament_id}/teams/{team_id}/deals/{deal_number}`
**Description:** Get detailed results for a specific deal including all frequencies.

**Response includes:**
| Field | Description |
|-------|-------------|
| `tournament` | Tournament context info |
| `teams.players` | Player details with IDs |
| `teams.opponents` | Opponent details with IDs |
| `frequencies[]` | All results achieved on this board |

**Frequency object:**
| Field | Description |
|-------|-------------|
| `dealNumber` | Board number |
| `scoreNS`, `scoreEO` | Bridge scores |
| `noteNS`, `noteEO` | Matchpoint percentages |
| `contract`, `declarant`, `first_card`, `result` | Contract details |
| `organizations[]` | Clubs that achieved this result |

---

### `GET /simultaneous/{tournament_id}/deals/{deal_number}/descriptions?organization_id={org_id}`
**Description:** Get all results for a deal at a specific club.

**Response (array):**
| Field | Description |
|-------|-------------|
| `team_ns_id`, `team_eo_id` | Team identifiers |
| `section` | Section name |
| `contract`, `declarant`, `first_card`, `result` | Contract details |
| `score_ns`, `score_eo` | Bridge scores |
| `note_ns`, `note_eo` | Matchpoint percentages |

---

## 5. Club APIs

### `GET /clubs`
**Description:** Get list of all active FFBridge clubs.

**Response (array):**
| Field | Description |
|-------|-------------|
| `id` | Club/organization ID |
| `name` | Club name |
| `code` | Club code |
| `is_active` | Active status |

---

## Example IDs for Testing

| Entity | ID | Description |
|--------|-----|-------------|
| Person | 597539 | Robert Salita |
| Organization | 1634 | Bridge Club Levallois Perret |
| Tournament | 34271 | Roy René 2026-01-06 |
| Team | 5091417 | Specific pair entry in tournament |

---

## Card Notation

| French | English |
|--------|---------|
| T | ♣ Clubs (Trèfle) |
| C | ♥ Hearts (Cœur) |
| K | ♦ Diamonds (Carreau) |
| P | ♠ Spades (Pique) |
| SA | NT (Sans Atout / No Trump) |

| Card | Meaning |
|------|---------|
| A | Ace |
| R | King (Roi) |
| D | Queen (Dame) |
| V | Jack (Valet) |
| 2-10 | Number cards |
