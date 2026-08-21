# Campus ANPR System — Database Schema (PostgreSQL)

## vehicles
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| plate_number | VARCHAR UNIQUE | normalized (uppercase, no spaces) |
| owner_name | VARCHAR | |
| owner_contact | VARCHAR | phone/email |
| owner_type | ENUM | student / faculty / staff / visitor |
| make | VARCHAR | |
| model | VARCHAR | |
| color | VARCHAR | |
| status | ENUM | verified / unverified / banned |
| ban_reason | TEXT | nullable |
| banned_at | TIMESTAMP | nullable |
| assigned_gate_id | UUID FK → gates.id | nullable |
| assigned_parking_area | VARCHAR | nullable |
| created_at | TIMESTAMP | |
| updated_at | TIMESTAMP | |

## gates
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| name | VARCHAR | e.g. "Gate A" |
| parking_area | VARCHAR | associated parking zone |
| camera_uri | VARCHAR | RTSP/stream source |
| inbound_reference_vector | JSONB | one-time calibrated `{dx, dy}` representing "known inbound direction" onscreen for this camera's mounting angle; used for the displacement-vector direction signal |
| camera_angle_deg | FLOAT | approximate mounting angle off head-on, nullable; informational, for QA when reviewing low-confidence direction events |
| active | BOOLEAN | |
| created_at | TIMESTAMP | |

## detection_events
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| plate_number_raw | VARCHAR | as OCR read it |
| plate_number_matched | VARCHAR | nullable, normalized match |
| vehicle_id | UUID FK → vehicles.id | nullable (no match = unrecognized) |
| gate_id | UUID FK → gates.id | |
| direction | ENUM | entry / exit |
| direction_confidence | ENUM | high / low — low means the two direction signals (scale-trend, displacement-vector) disagreed and the higher-confidence one was used; useful for QA/reviewing gate camera angle |
| ocr_confidence | FLOAT | |
| match_confidence | FLOAT | nullable, fuzzy-match score |
| track_id | VARCHAR | internal CV tracker ID |
| evidence_image_path | VARCHAR | full frame |
| evidence_plate_crop_path | VARCHAR | cropped plate |
| alert_triggered | BOOLEAN | true if banned-vehicle match |
| detected_at | TIMESTAMP | |
| created_at | TIMESTAMP | |

Indexes: `(plate_number_matched)`, `(gate_id, detected_at)`, `(direction, detected_at)` — history search will filter on these heavily.

## occupancy_state (derived/materialized, or computed on read)
Option A — computed live: "currently inside" = vehicles with a matched entry event not yet followed by a matched exit event.
Option B — maintained table for performance at scale:
| Column | Type | Notes |
|---|---|---|
| vehicle_id | UUID FK | nullable if unmatched plate |
| plate_number | VARCHAR | |
| gate_id | UUID FK | gate of entry |
| entered_at | TIMESTAMP | |

Recommendation: start with Option A (simpler, always consistent); move to Option B only if query performance becomes an issue at scale.

## users (admin/auth)
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| username | VARCHAR UNIQUE | |
| password_hash | VARCHAR | bcrypt/argon2 |
| role | ENUM | super_admin / gate_operator / viewer |
| created_at | TIMESTAMP | |
| last_login | TIMESTAMP | nullable |

## Notes
- Use `plate_number_matched` (not raw) for all joins/searches to absorb OCR noise via fuzzy matching done at write time.
- Consider a `plate_aliases` table later if you find OCR consistently misreads certain plates the same way (e.g., 0/O confusion) — not needed for v1.
