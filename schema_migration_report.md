# nflreadpy Migration Assessment — `ingest_nfl_data.py`

**Date:** 2026-09-08, updated 2026-09-10 · **Status: migration applied, plus pressure re-sourced to PFR.** Nothing has been written to Snowflake — run the ingest to rebuild the tables.

Tested with `nflreadpy` 0.1.5 / `polars` 1.44.1 against `nfl_data_py` 0.3.3 / `pandas` 2.2.3 in
throwaway virtualenvs. All comparisons were done on local frames and CSVs.

---

## Verdict

**Not a drop-in swap, but close.** Four of the six data pulls are byte-identical and need only a
function rename. One (play-by-play) needs a new explicit join. One (depth charts) is already broken
in production today and needs rework regardless of which library you pick.

Two pre-existing bugs surfaced during the investigation — see [Pre-existing breakage](#pre-existing-breakage).

**Effort: 8–12 hours.** Breakdown in [Effort estimate](#effort-estimate).

---

## Function mapping

| `nfl_data_py` call | `nflreadpy` equivalent | Schema delta | Verdict |
|---|---|---|---|
| `import_pbp_data([yr])` | `load_pbp(seasons=[yr])` **+ `load_participation(seasons=[yr])` join** | nflreadpy `load_pbp` has **372 cols vs 397**. The 26 missing ones are the `pbp_participation` release that `nfl_data_py` silently merged for you. **6 of them are in `keep_cols` and 4 are used by `feature_engineering.sql`.** | ⚠️ **Needs explicit join** |
| `import_ngs_data('passing', SEASONS)` | `load_nextgen_stats(seasons=..., stat_type='passing')` | 29 cols both. Zero column diffs, zero dtype diffs, zero value mismatches. | ✅ Rename only |
| `import_ngs_data('rushing', SEASONS)` | `load_nextgen_stats(seasons=..., stat_type='rushing')` | 22 cols both. Identical. | ✅ Rename only |
| `import_weekly_pfr('pass', SEASONS)` | `load_pfr_advstats(seasons=..., stat_type='pass', summary_level='week')` | 24 cols both. Identical. | ✅ Rename only |
| `import_weekly_pfr('rush', SEASONS)` | `load_pfr_advstats(seasons=..., stat_type='rush', summary_level='week')` | 16 cols both. Identical. | ✅ Rename only |
| `import_rosters(SEASONS)` | `load_rosters_weekly(seasons=...)` (or `load_rosters` for season-level) | `player_id`→`gsis_id`, `player_name`→`full_name`, `age` dropped (derive from `birth_date`). Values otherwise identical across all 34 shared cols. | ⚠️ **Rename + 2 col maps** — and see below, *this call does not currently work at all* |
| `import_depth_charts(SEASONS)` | `load_depth_charts(seasons=...)` | **Two different schemas depending on season.** 2018–2024 return the legacy 15-col weekly schema (with `season`/`week`/`club_code`/`full_name`/`position`/`depth_team`). **2025+ return a new 12-col timestamped schema** with none of those. Both libraries behave identically. | 🔴 **Already broken for 2025+** |

### Verification method

For each pair I pulled the same slice from both libraries and compared row-for-row after sorting on
natural keys — not just column lists:

- **PBP, 2025 week 1** (2,738 rows × all 39 `keep_cols`): **zero value mismatches**, identical
  non-null counts per column.
- **Full 2018–2025 rehearsal**: 389,358 rows, matching the 389K in `README.md`. Full-season
  aggregates (`sack`, `qb_hit`, `epa`, `was_pressure`, `defenders_in_box`, `time_to_throw`) match
  `nfl_data_py` **exactly** for both 2018 and 2025.
- **NGS / PFR / rosters**: zero value mismatches on all shared columns.

---

## The play-by-play join

`nfl_data_py.import_pbp_data` quietly merges a second nflverse release into every call
(`nfl_data_py/__init__.py:148-158`):

```python
raw.merge(partic, how='left',
          left_on=['play_id','game_id'],
          right_on=['play_id','nflverse_game_id'])
```

`nflreadpy` keeps these separate and makes you do it. The equivalent, verified to reproduce
`nfl_data_py` exactly:

```python
pbp  = nr.load_pbp(seasons=[yr])
part = nr.load_participation(seasons=[yr])
# polars will NOT implicitly coerce join keys — 2018 has pbp play_id=f64 vs participation play_id=i32
pbp  = pbp.with_columns(pl.col("play_id").cast(pl.Float64),  pl.col("game_id").cast(pl.String))
part = part.with_columns(pl.col("play_id").cast(pl.Float64), pl.col("nflverse_game_id").cast(pl.String))
j = pbp.join(part, left_on=["play_id","game_id"],
                   right_on=["play_id","nflverse_game_id"], how="left")
```

Columns this recovers that the SQL depends on — with `feature_engineering.sql` usage counts:

| Column | SQL uses | Feeds |
|---|---|---|
| `was_pressure` | 6 | `PRESSURES_ALLOWED`, `PRESSURE_RATE` (L38–39), `WAS_PRESSURE_FLAG` (L129) |
| `defenders_in_box` | 5 | `AVG_DEFENDERS_IN_BOX` (L85) |
| `time_to_throw` | 3 | `AVG_TIME_TO_THROW` (L40) |
| `number_of_pass_rushers` | 1 | `NUM_PASS_RUSHERS` (L19) |

These are **not** vestigial — fill rates are 92.6% for 2023–2025. Skipping the join silently zeroes
out pressure rate rather than erroring.

`offense_formation` and `offense_personnel` also come from participation but are unused by the SQL.

### Three polars gotchas found by actually running it

1. **Join keys are not implicitly coerced.** The 2018 join raises
   `SchemaError: datatypes of join keys don't match - play_id: f64 on left does not match play_id: i32 on right`.
   pandas coerced silently. Needs explicit casts.
2. **`was_pressure` is `Boolean` in polars, `float32` in pandas.** Because it has nulls,
   `.to_pandas()` yields **`object`** dtype, not `bool`. The existing `clean_columns()` then hits its
   `object` branch and **stringifies it to `'True'`/`'False'`/`'nan'`**, writing VARCHAR to Snowflake.
   `feature_engineering.sql:18` does `TRY_CAST(WAS_PRESSURE AS FLOAT)`, which returns NULL for
   `'False'` — so `PRESSURE_RATE` would silently become 0 with no error anywhere. Fix: cast Boolean →
   `Int8` in polars *before* `.to_pandas()`. (Interestingly `feature_engineering.sql:129-131` already
   has a VARCHAR-tolerant `WAS_PRESSURE` branch, so someone has been bitten by this before.)
3. **Project columns before the join, not after.** Joining the full 372-col pbp frame to the 26-col
   participation frame OOM'd the process (`memory allocation of 192 bytes failed`). Selecting
   `keep_cols` first fixed it and the whole 8-season run then finished in **21 seconds**.

---

## Pandas conversion

`nflreadpy` returns **polars** DataFrames, confirmed. `write_pandas` and `clean_columns()` both need
pandas, so `.to_pandas()` is required at the boundary (needs `pyarrow`, which
`snowflake-connector-python` already pulls in).

Dtype mapping observed on the real frames:

| polars | → pandas | Note |
|---|---|---|
| `String` | `str` / `object` | fine |
| `Int32` (nullable, e.g. `defenders_in_box`) | `float64` + NaN | same as `nfl_data_py` — fine |
| `Float64` | `float64` | `nfl_data_py` downcast to `float32`; payload is ~2× larger |
| **`Boolean` (nullable, e.g. `was_pressure`)** | **`object`** | 🔴 **breaks `clean_columns()`** — see gotcha 2 |
| `Date` (e.g. `birth_date`) | `datetime64` | cosmetic; zero value mismatches after normalising |

`clean_columns()` needs one added line: handle `pl.Boolean` before conversion, or add
`pd.BooleanDtype`/`object`-of-bools to the int8 branch.

---

## 2026 season availability

**This is the headline risk.**

Sept 9 is right — `load_schedules(2026)` puts the first regular-season game on **2026-09-09**. But
nflverse's *library-side* season clock disagrees with its own schedule: `get_current_season()` uses
the Thursday after Labor Day, which for 2026 is **Sept 10**. Today it returns **2025**.

That leaves a **one-day gap on Sept 9** where a week-1 game will have been played but
`load_pbp([2026])` still refuses to fetch it. So don't gate ingestion on `get_current_season()`
alone — see [the fix](#one-structural-problem-for-live-use).

Measured behavior right now:

| Call | 2026 result today |
|---|---|
| `load_pbp(seasons=[2026])` | 🔴 `ValueError: Season must be between 1999 and 2025` |
| `load_participation(seasons=[2026])` | 🔴 `ValueError: Season must be between 2016 and 2025` |
| `load_nextgen_stats(seasons=[2026])` | 🔴 `ValueError: Season must be between 2016 and 2025` |
| `load_pfr_advstats(seasons=[2026])` | 🔴 `ValueError: Season must be between 2018 and 2025` |
| `load_rosters_weekly(seasons=[2026])` | 🔴 `ValueError: Season must be between 2002 and 2025` |
| `load_depth_charts(seasons=[2026])` | ✅ **505,422 rows**, `dt` through `2026-09-08T11:56:57Z` |
| `load_schedules(seasons=[2026])` | ✅ 272 rows (schedule published in advance) |

Depth charts and rosters use a **different rollover** — `get_current_season(roster=True)`, which flips
on March 15 — so `get_current_season(roster=True)` already returns 2026. That is why depth charts
work and pbp does not.

### How it behaves for in-progress / incomplete weeks

Three distinct failure modes, and **all of them raise — none return partial or empty frames**:

1. **Before Sept 10:** client-side `ValueError` from season validation. No network call is made.
2. **On/after Sept 10, before the first games finish:** validation passes, then the fetch 404s —
   `ConnectionError: Failed to download .../play_by_play_2026.parquet: 404 Client Error`. Verified by
   requesting that exact URL.
3. **Mid-season:** the per-season parquet is rebuilt as games complete, so it contains only finished
   games. Partial weeks come back as fewer rows, not as errors or nulls.

**No manual refresh call is needed for a batch job.** Default `cache_mode` is `MEMORY`
(per-process), so every script run fetches fresh. Only if you set `NFLREADPY_CACHE=filesystem` does
the 24h `cache_duration` apply, and then `nr.clear_cache()` / `nr.clear_cache("pbp_2026")` is
available. Configurable via `NFLREADPY_CACHE`, `NFLREADPY_CACHE_DIR`, `NFLREADPY_CACHE_DURATION`,
`NFLREADPY_TIMEOUT` (default 30s), `NFLREADPY_VERBOSE`.

### One structural problem for live use

**Season validation is up-front and all-or-nothing.** `load_nextgen_stats(seasons=[2025, 2026])`
raises and returns *nothing* — it does not return 2025 and skip 2026:

```
load_pbp([2025,2026])            ValueError: Season must be between 1999 and 2025
load_nextgen_stats([2025,2026])  ValueError: Season must be between 2016 and 2025
load_pfr_advstats([2025,2026])   ValueError: Season must be between 2018 and 2025
```

`load_ngs()` and `load_pfr()` currently pass the whole `SEASONS` list in one call, so the moment
`SEASONS` includes 2026 those loads return zero rows for *every* season. `load_pbp()` already loops
per-year and would degrade gracefully.

**Recommendation:** derive `SEASONS` from `nr.get_current_season()` rather than hardcoding
`range(2018, 2026)`, and loop per-season with per-season `try/except` for NGS and PFR too.

---

## Pre-existing breakage

Two things found that are broken *today*, independent of this migration.

**1. `load_rosters()` has never worked.** `ingest_nfl_data.py:85` calls `nfl.import_rosters(SEASONS)`.
That function **does not exist** in `nfl_data_py` 0.3.3 — only `import_seasonal_rosters` and
`import_weekly_rosters` do:

```
AttributeError: module 'nfl_data_py' has no attribute 'import_rosters'
```

The bare `except Exception` at line 87 swallows it, prints `Rosters failed:`, and continues. So
`RAW_ROSTERS` has never been populated. Consistent with it having zero references anywhere downstream.

**2. `RAW_DEPTH_CHARTS` stopped matching the app's query at the 2025 season.** nflverse rebuilt the
`depth_charts` release on a new source **starting with 2025**. Seasons 2018–2024 still return the
legacy 15-col schema, so the app's query works for those. From 2025 on, **both** libraries return:

```
dt, team, player_name, espn_id, gsis_id, pos_grp_id, pos_grp, pos_id, pos_name, pos_abb, pos_slot, pos_rank
```

`streamlit_app.py:80-84` selects `FULL_NAME, POSITION, DEPTH_TEAM, WEEK` and filters on `CLUB_CODE`
and `SEASON`. **None of those six columns exist in the 2025+ feed.** This is not a migration
regression — `nfl_data_py` returns the identical new schema — but it means the depth-chart panel goes
blank for 2025 and 2026 while still working for 2018–2024.

Measured boundary:

| Seasons | Schema | Rows/season | Grain |
|---|---|---|---|
| 2018–2024 | legacy, 15 cols | ~37,000 | one chart per team per week |
| 2025–2026 | new, 12 cols | ~505,000–554,000 | timestamped, many snapshots per day |

The new feed is also **~15× more granular**, so a naive load inflates the table from ~37k to ~554k
rows per season. Deduplicating to the latest snapshot per team per week restores the legacy grain
(2025 → 40,599 rows).

The good news: the new data is richer and live. 2026 depth charts are current as of today
(`dt` max `2026-09-08T11:56:57Z`), 32 teams, 82,169 OL rows, with clean `pos_abb` values
(`LT`/`LG`/`C`/`RG`/`RT`) and `pos_rank` = depth (1 = starter) — better than the legacy
`position`/`depth_position` split. Mapping:

| Old column | New source |
|---|---|
| `full_name` | `player_name` |
| `position` | `pos_abb` (`LT`/`LG`/`C`/`RG`/`RT` — all already in the app's `IN` list) |
| `depth_team` | `pos_rank` |
| `club_code` | `team` |
| `season` | derive at ingest (or from `dt`) |
| `week` | derive from `dt` against `load_schedules()` — no week column exists |

---

## Unused ingestion

Worth flagging before you spend time on it: **four of the seven tables this script builds are read by
nothing.** Reference counts across `feature_engineering.sql`, `streamlit_app.py`, `run_inference.py`,
and `train_models.py`:

| Table | Downstream refs |
|---|---|
| `RAW_PBP` | 5 |
| `RAW_DEPTH_CHARTS` | 2 (currently broken) |
| `RAW_NGS_PASSING` | **0** |
| `RAW_NGS_RUSHING` | **0** |
| `RAW_PFR_PASS` | **0** |
| `RAW_PFR_RUSH` | **0** |
| `RAW_ROSTERS` | **0** (and never populated) |

The NGS and PFR pulls are also the four that are perfect drop-in renames. So the *real* migration
surface is PBP + depth charts. Deleting or deferring the unused loads would cut the work roughly in
half — your call, not mine.

---

## Effort estimate

| Task | Hours |
|---|---|
| Rename NGS ×2 and PFR ×2 calls; add per-season loop + `try/except` | 0.5 |
| PBP: participation join, join-key casts, column projection, `.to_pandas()` | 2.0 |
| `clean_columns()`: handle `pl.Boolean` → `Int8` before conversion | 0.5 |
| Rosters: fix the never-working call, map `gsis_id`/`full_name` | 0.5 |
| Dynamic `SEASONS` from `get_current_season()` + 2026 pre-season guards | 1.5 |
| Validate rebuilt tables against current Snowflake row counts / feature values | 2.0 |
| **Migration subtotal** | **7.0** |
| Depth charts: normalise the legacy and 2025+ schemas into one table | 2.0–3.0 |
| **Total** | **9–10** |

Call it **8–12 hours** with contingency. The depth-chart half is separable and is arguably a bug fix
rather than migration work.

**Revised after building the draft:** the depth-chart work turned out cheaper than estimated in one
respect and dearer in another. Cheaper, because re-deriving the legacy column names as aliases at
ingest means **`streamlit_app.py` needs no change at all**. Dearer, because there are two source
schemas to normalise, not one. Net effect is roughly a wash.

---

## Red flags

1. **🔴 2026 data does not exist yet.** The season starts Sept 10, 2026 (two days out), not Sept 9.
   Nothing can be validated against real 2026 pbp until the first games complete. Budget a
   verification pass in week 1.
2. **🔴 Silent zeroing of `PRESSURE_RATE`** if the `Boolean` → pandas `object` conversion is missed.
   No exception is raised anywhere in the chain — `TRY_CAST('False' AS FLOAT)` just returns NULL.
   This is the single most dangerous item on the list.
3. **🔴 `load_pbp` alone silently drops 4 SQL-critical columns.** A naive `import_pbp_data` →
   `load_pbp` rename compiles, runs, writes 389K rows, and produces a table missing pressure rate,
   time to throw, and defenders in box. It looks like a success.
4. **🟠 All-or-nothing season validation** will zero out the NGS and PFR loads entirely the moment
   `SEASONS` includes an unreleased season.
5. **🟠 New hard dependency on the `pbp_participation` release.** It is now a separate fetch and join
   rather than something bundled. Its fill rate jumped from ~39% to ~93% at 2023, indicating a source
   change, and it has never yet existed for 2026 — it may lag pbp early in the season. This is the
   most fragile link in the pipeline for live use.
6. **🟠 `nfl_data_py` is effectively uninstallable on modern Python.** It pins `pandas<2.0` and
   `numpy<2.0`; pandas 1.5.3 has no cp312+ wheel, so a clean install on Python 3.12 fails building
   from source (`ModuleNotFoundError: No module named 'pkg_resources'`). I had to force
   `--no-deps` with pandas 2.2.3 just to run this comparison. Independent of maintenance status,
   this alone justifies the migration.
7. **🟠 `pyproject.toml` declares neither library.** `nfl_data_py` is imported but never listed as a
   dependency, so the ingestion env is undeclared. Add `nflreadpy` explicitly while you are in there.
8. **🟡 Rosetta/polars note (local only).** `/usr/local/bin/python3.12` on this arm64 Mac is an
   x86_64 build, and polars warns `Missing required CPU features ... will likely result in a crash`.
   Fixed with `pip install 'polars[rtcompat]'`. Not an issue for Snowflake-side execution, but it will
   bite anyone running the ingest locally on Apple Silicon under Rosetta.
9. **✅ No licensing change.** `nflreadpy` MIT, `nfl_data_py` MIT, `polars` MIT. Same nflverse
   releases underneath, same data terms.

---

## Update 2026-09-10 — the season rolled over, and participation did not

The rollover happened, `get_current_season()` now returns **2026**, and live data is flowing. Measured
this morning:

| Call | 2026 status |
|---|---|
| `load_pbp` | ✅ **166 rows** — the Sept 9 opener. Partial weeks return partial data, no error |
| `load_rosters_weekly` | ✅ 2,963 rows |
| `load_depth_charts` | ✅ 509,781 rows |
| `load_nextgen_stats` | ✅ 4 rows |
| `load_pfr_advstats` | ⏳ `ConnectionError` 404 — not published yet, will appear |
| `load_participation` | 🔴 **`ValueError: Season must be between 2016 and 2025`** |

### Participation is retroactive, not delayed

This is worse than the lag I flagged as a watch item, and it is not a bug. nflreadpy caps
participation deliberately (`load_participation.py:31`):

```python
# participation only available on a historical basis from FTN
max_season = get_current_season(roster=True) - 1
```

So participation is **never** available for an in-progress season. 2026 data backfills around
**March 2027**, when the roster year rolls over. That makes these six columns unavailable for the
entire live season:

`offense_formation`, `offense_personnel`, `defenders_in_box`, `number_of_pass_rushers`,
`time_to_throw`, `was_pressure`

**This is not a migration regression.** `nfl_data_py` fetched the same release inside a
`try: ... except HTTPError: pass`, so it would have silently skipped the merge and left the columns
absent entirely. The data does not exist in-season either way. But it does mean the project's
live-2026 goal has a real hole in it that predates this work.

### What that costs, precisely

`PRESSURE_RATE` carries **25% of `PASS_BLOCK_SCORE`** (`feature_engineering.sql:245`), and
`PASS_BLOCK_SCORE` is 60% of `COMPOSITE_OL_SCORE` — so pressure is **~15% of the composite**.

The percentile windows have **no `PARTITION BY SEASON`** (L223-238), so they rank every game in the
table against every other. With `PRESSURE_RATE` null for 2026, Snowflake sorts those rows last under
`ORDER BY ... ASC`, `1 - PERCENT_RANK()` lands near **0**, and every 2026 game silently forfeits its
full 25% pass-block component. 2026 teams would not merely be unscored — they would be ranked
**bottom of the all-time table**, and nothing in the pipeline would raise.

### Loader fix applied

`load_pbp` now fetches participation in its own `try/except` (`_join_participation`). Previously the
`ValueError` propagated and **skipped 2026 play-by-play entirely** — throwing away sacks, QB hits, EPA
and stuff rate, which *are* live, over six columns that are not. It now writes them as typed nulls,
holding one stable 39-column schema so the per-season appends cannot collide, and prints:

```
2026: participation unavailable (ValueError) — writing 6 columns as NULL: ...
2026: >>> PRESSURE_RATE, AVG_TIME_TO_THROW and AVG_DEFENDERS_IN_BOX will be NULL for 2026 <<<
```

Verified: 389,524 rows across 2018–2026, 39 columns throughout, 2026's 166 rows carrying core
metrics with participation null.

### What is available live as a substitute

| Missing column | Live substitute | Status |
|---|---|---|
| `time_to_throw` | `load_nextgen_stats(stat_type='passing').avg_time_to_throw` | ✅ Available now, player-week grain — needs aggregating to team-week. **This is already ingested into the unused `RAW_NGS_PASSING` table.** |
| `number_of_pass_rushers` | `load_ftn_charting().n_pass_rushers` | ⏳ `ftn_charting_2026` 404s today; should appear in-season |
| `defenders_in_box` | `load_ftn_charting().n_defense_box` | ⏳ same |
| `was_pressure` | **No equivalent.** FTN's `is_qb_out_of_pocket` / `is_qb_fault_sack` measure different things | 🔴 No live source |

Pressure itself has no live substitute. `SACK_RATE` and `QB_HIT_RATE` are live and already carry 30%
and 20% of the pass-block score, so the observable part of pass protection is not lost — but
`PRESSURE_RATE` specifically cannot be computed until the backfill.

**This needs a scoring decision before 2026 games are scored — see the open question at the end.**

---

## Where this leaves the 2026 season

The pipeline is now validated against known-good 2025 data ahead of kickoff, which was the point of
doing it this week. From here:

Ingestion is working against live 2026 data as of Sept 10. What remains is not an ingestion problem.

- **Now:** pbp, rosters, depth charts and NGS all carry 2026. PFR will appear once published; the
  loader skips it cleanly until then.
- **All season:** participation stays unavailable, so `PRESSURE_RATE`, `AVG_TIME_TO_THROW` and
  `AVG_DEFENDERS_IN_BOX` are null for 2026.
- **~March 2027:** participation backfills and a re-run fills those columns retroactively.

### Pressure data: what else exists

Asked and measured. Full inventory of pressure-adjacent sources, and how each behaves in-season:

| Source | Field | In-season? | Coverage |
|---|---|---|---|
| `pbp_participation` | `was_pressure` | 🔴 **Never** — capped at roster year − 1 | 2016–2025 |
| **`pfr_advstats('pass')`** | **`times_pressured`**, `times_hurried`, `times_hit`, `times_sacked`, `times_blitzed` | ✅ **Yes** — bounded on `get_current_season()` | 2018+ |
| `pfr_advstats('def')` | `def_pressures`, `def_times_hurried`, `def_times_hitqb` | ✅ Yes | 2018+ |
| `nextgen_stats('passing')` | `avg_time_to_throw` | ✅ Yes — live for 2026 today | 2016+ |
| `ftn_charting` | `n_pass_rushers`, `n_defense_box`, `is_qb_out_of_pocket` | ⏳ 2026 file 404s today | 2022+ |

**`RAW_PFR_PASS` already contains this.** It is one of the four tables nothing currently reads, it
covers 2018+ (exactly `START_SEASON`), and unlike participation nflreadpy permits it for the current
season — the 2026 404 is just "not built yet", not "not allowed".

`times_pressured` is exactly `times_hurried + times_hit + times_sacked` (verified identity).

#### How well it substitutes

Measured against participation's `was_pressure` for all of 2025, 570 team-games, 100% matched:

| Grain | Source | Pearson | Spearman |
|---|---|---:|---:|
| **Team-game** | PFR `times_pressured` | **0.567** | **0.562** |
| Team-game | PFR `def_pressures` | 0.574 | 0.562 |
| Team-game | PFR `times_hurried + times_hit` | 0.396 | 0.367 |
| **Team-season** | PFR `times_pressured` | **0.838** | **0.844** |
| Team-season | PFR `def_pressures` | 0.840 | 0.810 |

PFR also runs systematically lower: mean pressure rate 0.227 against participation's 0.292, mean
absolute difference 0.090.

**The honest read:** PFR pressure is a good *season-level* proxy and a weak *game-level* one — and
`TEAM_OL_SCORES` is scored per `GAME_ID`. Top-10 worst-pressure-rate teams overlap 7 of 10 on the
season. The defensive side offers no advantage and covers 3 fewer team-games.

#### The ML path is also affected

`train_models.py:26-38` and `run_inference.py:11-23` both list `PRESSURE_RATE`,
`AVG_TIME_TO_THROW`, `AVG_DEFENDERS_IN_BOX`, `THIRD_LONG_PRESSURE_RATE` and
`ROLLING_5_PRESSURE_RATE` in `FEATURE_COLS`. `train_models.py:52` then does
`dropna(subset=FEATURE_COLS)`, so **every 2026 row would be dropped from training**, and inference
would run on NaN-filled inputs. Whatever is chosen has to cover the ML path, not just the SQL.

### Open question — scoring with no pressure data

`feature_engineering.sql` is untouched and still weights `PRESSURE_PCTL` at 25% of pass block, so
**scoring 2026 today would rank every 2026 game at the bottom of the all-time table.** Three ways to
handle it:

1. **Re-source `PRESSURE_RATE` from PFR for *all* seasons (2018+).** One definition everywhere, live
   every season, no nulls, no re-weighting, and the ML path keeps working untouched. Cost: historical
   `PRESSURE_RATE` changes, so historical scores shift and the v4 models need retraining.
   *Recommended* — it is the only option that leaves the pipeline with no missing-data special case.
2. **Re-weight when pressure is null** — redistribute the 0.25 across `SACK_RATE`, `QB_HIT_RATE`,
   `PASS_EPA` and `PASS_SUCCESS`. History untouched, but 2026 pass-block scores are built from four
   components rather than five, and `FEATURE_COLS`/`dropna` still needs handling for the ML path.
3. **Gate 2026 out of scoring** until the ~March 2027 backfill. Honest, but gives up the live-season
   goal.

Do **not** use PFR only for 2026 and participation for history — that puts a definitional break
exactly at the boundary you most want to compare across, with a 0.065 level shift in the metric.

Either option 1 or 2 should also source `AVG_TIME_TO_THROW` from the already-ingested
`RAW_NGS_PASSING`, which carries it live. These are modelling changes to `feature_engineering.sql`
and the model feature set, so they are your call — say which and I will implement it.

---

## Implementation

The migration is applied. `ingest_nfl_data.py` now runs on nflreadpy. The original is recoverable at
`git show HEAD:ingest_nfl_data.py`.

### Files changed

| File | Change |
|---|---|
| `ingest_nfl_data.py` | Rewritten on nflreadpy/polars |
| `pyproject.toml` | Added `[project.optional-dependencies] ingest` — kept out of the main deps because `snowflake.yml` ships this file to Streamlit-in-Snowflake |
| `README.md`, `ONE_PAGER.md`, `streamlit_app.py` | Doc/architecture references updated (the `streamlit_app.py` edits are documentation strings only — **no query logic changed**) |
| `feature_engineering.sql` | **Unchanged.** Verified identical feature output — see below |
| Snowflake tables | **Untouched.** Run the ingest to rebuild |

Install and run:

```bash
pip install -e '.[ingest]'
python ingest_nfl_data.py --dry-run   # exercises everything, writes CSVs, no Snowflake
python ingest_nfl_data.py             # writes to Snowflake
```

`--dry-run` never imports the Snowflake connector and writes CSVs to `./dryrun_out/` instead:

```bash
python ingest_nfl_data_nflreadpy.py --dry-run
```

### What it changes

| Area | Change |
|---|---|
| PBP | `load_pbp` + `load_participation` left join, with explicit join-key casts and column projection before the join |
| `to_pandas()` | New helper; casts polars `Boolean` → `Int8` before conversion so `was_pressure` cannot be stringified |
| `clean_columns()` | Bool branch moved **ahead** of the object branch, and now catches pandas `boolean` dtype too |
| NGS / PFR / rosters | Renamed to `load_nextgen_stats` / `load_pfr_advstats` / `load_rosters_weekly`, each looped per-season so one unreleased season cannot zero out the table |
| Rosters | Fixes the never-working `import_rosters` call; adds `player_id` / `player_name` / `age` aliases |
| Depth charts | Normalises the legacy and 2025+ schemas into one table, re-deriving the legacy column names so `streamlit_app.py` keeps working; dedupes the new feed to latest snapshot per team per week |
| `SEASONS` | Now derived from `nr.get_current_season(roster=True)` instead of a hardcoded `range(2018, 2026)` |

### Season bound

`target_seasons()` runs to the **roster year** (2026 today) rather than the game season (2025), and
every loader skips seasons that raise. That sidesteps the Sept 9/10 gap entirely: 2026 is requested
from now on, fails harmlessly until nflverse publishes it, and starts flowing the moment it appears —
with no code change and no scheduled edit.

### Verification

Dry run over 2018–2026, output compared against `nfl_data_py` totals for the same span:

| Metric | `nfl_data_py` | Draft | |
|---|---:|---:|---|
| rows | 389,358 | 389,358 | ✅ |
| `was_pressure` | 48,254.00 | 48,254.00 | ✅ |
| `sack` | 10,722.00 | 10,722.00 | ✅ |
| `qb_hit` | 24,044.00 | 24,044.00 | ✅ |
| `epa` | −1,377.70 | −1,377.70 | ✅ |
| `defenders_in_box` | 1,837,705.00 | 1,837,705.00 | ✅ |
| `time_to_throw` | 419,903.13 | 419,903.13 | ✅ |

Row count also matches the 389K quoted in `README.md`.

**Feature-level parity.** Raw-column parity is necessary but not sufficient, so I also replayed
`feature_engineering.sql`'s two main feature tables against both the old and new pipeline output for
2025 — 570 feature rows each, all metrics identical:

| `TEAM_PASS_BLOCKING_FEATURES` | | `TEAM_RUN_BLOCKING_FEATURES` | |
|---|---|---|---|
| `DROPBACKS` | ✅ | `RUSH_ATTEMPTS` | ✅ |
| `SACKS_ALLOWED` / `SACK_RATE` | ✅ | `AVG_RUSH_YARDS` | ✅ |
| `QB_HITS_ALLOWED` / `QB_HIT_RATE` | ✅ | `RUSH_EPA_PER_PLAY` | ✅ |
| `PRESSURES_ALLOWED` / `PRESSURE_RATE` | ✅ | `STUFF_RATE` | ✅ |
| `AVG_TIME_TO_THROW` | ✅ | `EXPLOSIVE_RUN_RATE` | ✅ |
| `PASS_EPA_PER_PLAY` | ✅ | `SHORT_GAIN_RATE` | ✅ |
| `AVG_PASS_RUSHERS_FACED` | ✅ | `AVG_DEFENDERS_IN_BOX` | ✅ |
| | | `TOTAL_RUSH_YARDS` | ✅ |

This is what confirms `feature_engineering.sql` needs no change. Note its `POSTEAM != 'nan'` guard
(L24) is still required — `clean_columns()` still stringifies object columns, so null teams still
arrive as the literal `'nan'`.

Depth charts, with the app's exact `streamlit_app.py:80-84` predicate replayed against the output:

| Season | Schema | Rows written | App query returns (KC, OL) |
|---|---|---|---|
| 2024 | legacy | 37,312 | 336 rows |
| 2025 | 2025+ | 40,599 (from 554,215 raw) | 246 rows |
| 2026 | 2025+ | 2,177 (week 1 only) | 13 rows |

2026 returns only week 1 because the season has not started — correct, not a defect.

2026 currently skips cleanly on every other loader:

```
Fetching 2026...
  2026 failed: ValueError: Season must be between 1999 and 2025
```

### Decisions I made, flagged for you

1. **The four unused tables.** `RAW_NGS_*`, `RAW_PFR_*` and `RAW_ROSTERS` have zero downstream
   readers. The draft keeps loading all of them, unchanged in scope. Dropping them is a one-line
   edit to `main()` if you want it.
3. **Depth-chart week semantics.** The draft assigns each snapshot to the *next* scheduled week,
   which matches what a depth chart is for. If you would rather date them to the most recently
   completed week, that is a one-word change (`strategy="forward"` → `"backward"`).
3. **`RAW_DEPTH_CHARTS` gains four columns** (`pos_grp`, `pos_slot`, `dt`, `schema_version`) and the
   legacy `formation` / `elias_id` / `football_name` / `first_name` / `last_name` are dropped from
   the normalised output. Nothing reads them, but the table is recreated with `overwrite=True`, so
   confirm you are happy losing them.
4. **Legacy `week` has 234 nulls/season** (`game_type='SBBYE'`, Super Bowl bye). Source-side and
   pre-existing — the script passes them through rather than inventing a value.

### Before you run it against Snowflake

Every loader uses `overwrite=True`, so the run recreates each table. Two things worth knowing:

- `RAW_DEPTH_CHARTS` is recreated with the new 16-column normalised schema. `DEPTH_TEAM` stays
  VARCHAR (`pos_rank` is cast to String) so `streamlit_app.py` is unaffected.
- `RAW_ROSTERS` gets created for the first time, since the old call never worked.

The `--dry-run` output in `./dryrun_out/` is a faithful preview of exactly what will be written.

---

## Update 2026-09-10 (2) — pressure re-sourced to PFR

Implemented option 1: `PRESSURE_RATE` now comes from PFR for **all** seasons, and
`AVG_TIME_TO_THROW` from NGS, so neither depends on retroactive participation data.

### Changes to `feature_engineering.sql`

`TEAM_PASS_BLOCKING_FEATURES` gains two CTEs and no longer reads `WAS_PRESSURE` or
`TIME_TO_THROW` from `RAW_PBP`:

- `pfr_pressure` — `SUM(TIMES_PRESSURED)` per team-game from `RAW_PFR_PASS`
  (`= TIMES_HURRIED + TIMES_HIT + TIMES_SACKED`), divided by pbp `DROPBACKS`.
- `ngs_time_to_throw` — attempts-weighted `AVG_TIME_TO_THROW` per team-week from
  `RAW_NGS_PASSING`, excluding `WEEK = 0` season-aggregate rows.

Output column names and order are unchanged, so `TEAM_OL_FEATURES` and everything below it needed no
edit.

### Two team-abbreviation traps, caught before they shipped

Both would have produced silent NULLs, not errors:

| Source | Uses | Play-by-play uses | Blast radius |
|---|---|---|---|
| `RAW_PFR_PASS` | `OAK` (2018–19) | `LV` for every season | Raiders pressure NULL for 2 seasons |
| `RAW_NGS_PASSING` | `LAR` | `LA` | **Rams time-to-throw NULL in every season** |

Both are remapped in the CTEs. Verified coverage after remapping:

| Season | Team-games | `PRESSURE_RATE` | `AVG_TIME_TO_THROW` |
|---|---:|---:|---:|
| 2018–2025 | 4,454 | **100%** | 98–99% |
| 2026 | 2 | 0% (PFR not yet published) | 100% |

### Re-weight guard for the publication lag

PFR trails play-by-play by a few days, so early each season `PRESSURE_RATE` is NULL. Because the
percentile windows are **not** partitioned by season, a NULL lands near the bottom of the all-time
distribution and costs the game its whole 25% pressure component. Measured on the live 2026 opener,
`COMPOSITE_OL_SCORE` came out at **27.8** against a historical average near 50.

The `scored` CTE now redistributes that weight across the four components that are present
(`/ 0.75`). Validated by recomputing 2018–2025 both ways:

| | Mean absolute error vs true score |
|---|---:|
| **Re-weighted** | **1.08 points** |
| Naive NULL (previous behaviour) | 12.56 points |

This is not a 2026 workaround — the lag recurs every opening week, so the guard is permanent.

### Verified end to end

The whole of `feature_engineering.sql` was executed against the dry-run CSVs in DuckDB, with
`TEAM_OL_CLUTCH_FEATURES` stubbed. All 9 statements run, `ML_TRAINING_DATA` builds, and scores land
where they should:

| Season | Games | Avg composite | Min | Max |
|---|---:|---:|---:|---:|
| 2018–2025 | 4,454 | 49.4 – 53.8 | 5.8 | 96.9 |
| 2026 | 2 | 33.8 | 29.0 | 38.6 |

2026's two games sit inside the historical single-game range; two games is not a sample.

### Two things I did not change, and why

1. **`TEAM_OL_CLUTCH_FEATURES` has no DDL anywhere in the repo.** It is read by
   `feature_engineering.sql:209` and four `streamlit_app.py` queries, but nothing creates it — it
   exists only in Snowflake, from something outside version control. Running the pipeline against a
   clean schema would fail. I did not invent a definition, because guessing would silently produce
   wrong clutch features. **Worth recovering the real DDL and committing it.**
2. **`AVG_DEFENDERS_IN_BOX` and `THIRD_LONG_PRESSURE_RATE` are still participation-derived**, so they
   remain unavailable live. Neither feeds `COMPOSITE_OL_SCORE` — both are ML-only features — but both
   are in `FEATURE_COLS`, so `train_models.py:52`'s `dropna` will still exclude 2026 from training.
   `THIRD_LONG_PRESSURE_RATE` is worse than NULL: `WAS_PRESSURE_FLAG`'s `ELSE 0` renders it a
   confident **0.0**, i.e. "no pressure on 3rd and long". `ftn_charting.n_defense_box` could replace
   the first once nflverse publishes the 2026 file. Both are modelling calls, and the v4 models need
   retraining regardless now that historical `PRESSURE_RATE` has changed definition.

---

## Update 2026-09-10 (3) — clutch DDL reconstructed, ML features resolved

### `AVG_DEFENDERS_IN_BOX` and `AVG_PASS_RUSHERS_FACED` — fixed

nflreadpy's own source comment ("participation only available on a historical basis from FTN") turned
out to be the key: **participation is FTN charting, republished retroactively.** FTN itself is
permitted for the current season. Measured on 2025 plays:

| participation | FTN | Correlation | Exact match |
|---|---|---:|---:|
| `defenders_in_box` | `n_defense_box` | 0.9996 | **99.8%** |
| `number_of_pass_rushers` | `n_pass_rushers` | 0.9999 | **100.0%** |

So this is a backfill of a source from itself, not a definitional change — none of the era-mixing
objection that applied to PFR pressure.

- `ingest_nfl_data.py` gains `load_ftn_charting()` → new `RAW_FTN_CHARTING` table (2022+, 185,215
  rows for 2022–2025).
- `feature_engineering.sql` gains a `PBP_CHARTED` view that COALESCEs participation first, FTN
  second. `TEAM_PASS_BLOCKING_FEATURES` and `TEAM_RUN_BLOCKING_FEATURES` now read the view.
  Coverage: **100% for 2018–2025.**

FTN only reaches back to 2022, hence participation-first ordering. FTN's 2026 file is not published
yet (404), so those columns fill in once it appears — same short lag as PFR, not a season-long gap.

### `THIRD_LONG_PRESSURE_RATE` — was silently wrong, now honest

`WAS_PRESSURE_FLAG`'s `ELSE 0` was converting "pressure was never charted" into a confident "no
pressure". Two consequences, both now fixed by returning NULL and excluding uncharted plays from the
denominator:

| Season | Old avg | New avg | Change |
|---|---:|---:|---:|
| 2018 | 0.3678 | 0.4187 | **+0.0510** |
| 2019 | 0.3442 | 0.3846 | +0.0404 |
| 2020 | 0.3538 | 0.3941 | +0.0403 |
| 2021 | 0.3462 | 0.3862 | +0.0400 |
| 2022 | 0.3352 | 0.3789 | +0.0437 |
| 2023–2025 | — | — | **0.0000** |
| 2026 | 0.0000 | NULL | honest |

2018–2022 were understated by 4–5 points because participation charted only ~39% of plays then.
2023–2025 are unchanged at ~93% coverage. **The v4 training data carried this bias**, which is
another reason retraining is required.

There is no live substitute: pressure at play level exists only in participation, and FTN carries no
pressure field. This feature will be NULL for in-progress seasons by design.

### `TEAM_OL_CLUTCH_FEATURES` — reconstructed, not recovered

The original DDL is **not in the repository and never was**. The initial commit already only
LEFT JOINed it, and `SKILL.md:80` records it as "(created separately)". Recovering the true
definition would mean reading it out of Snowflake, which I could not do — the attempt to read the
connection credentials was correctly blocked.

So `clutch_features.sql` is a **reconstruction from documented behaviour**, built from
`streamlit_app.py:478/517/523`, `streamlit_app.py:252-255` and `SKILL.md:12-13`. It writes to
`TEAM_OL_CLUTCH_FEATURES_RECONSTRUCTED`, so it **cannot overwrite the live table**, and ships with a
validation query and promotion instructions.

It produces all nine metrics the app queries, including `OL_LOW_LEVERAGE_SUCCESS` — which
`feature_engineering.sql` never selected, so it would have been missed by reading that file alone:

| Season | Resp. rate | OL success | Clean pocket | Clutch index | Protection | Run blocking |
|---|---:|---:|---:|---:|---:|---:|
| 2018 | 0.932 | 0.451 | 0.688 | −0.0019 | +0.0577 | +0.0186 |
| 2023 | 0.970 | 0.419 | 0.715 | −0.0102 | +0.0683 | +0.0245 |
| 2025 | 0.967 | 0.431 | 0.708 | −0.0109 | +0.0623 | +0.0201 |

Signs and magnitudes match the documented semantics: protection and run-blocking clutch are positive
(more pressure and more stuffs in high leverage, as expected), and the clutch index sits near zero as
a difference metric should. `OL_RESPONSIBILITY_RATE` steps up at 2023 exactly where participation
coverage jumps from 39% to 93% — independent evidence the classification is keying off the right
thing.

**Two caveats you must resolve before promoting it:**

1. **The high-leverage definition is ambiguous in the docs.** `streamlit_app.py:478` says "3rd/4th
   down, close game, 2nd half"; `SKILL.md:13` says "4th quarter, 3rd down, one-score game". Read as a
   strict AND, high-leverage plays would be a handful per game and the index pure noise, so this
   implements the broader reading: any 3rd/4th down, OR any second-half play in a one-score game
   (42.3% of plays). If validation shows a high correlation with a constant offset, this is the knob.
2. **`qtr`, `game_half` and `game_seconds_remaining` were not being ingested at all**, so the old
   keep list could not have identified a half or quarter. They are now in `PBP_KEEP` (RAW_PBP is 42
   columns). This is further evidence the original table was built from a fuller pbp pull outside
   this repo.

Four of its columns — `OL_RESPONSIBILITY_RATE`, `OL_SUCCESS_RATE`, `OL_CLEAN_POCKET_RATE`,
`OL_PROTECTION_CLUTCH` — need play-level pressure and so are unavailable live. The other five work
in-season.

### Verified end to end

`clutch_features.sql` then `feature_engineering.sql` executed in DuckDB against dry-run extracts with
the reconstruction joined in place of a stub — 1 + 10 statements, all clean. 2018–2025 composite
averages hold at 49.4–53.8.

**One pre-existing behaviour worth knowing:** `TEAM_OL_FEATURES` wraps every clutch column in
`COALESCE(..., 0)` (L246-253), so a missing clutch value reads as a real 0 rather than NULL. For 2026
that makes `OL_CLEAN_POCKET_RATE` display as 0.000 rather than blank. Not introduced here, but it
interacts badly with the live-season gaps.
