"""
NFL data ingestion into Snowflake, built on nflreadpy.

Migrated from nfl_data_py, which is deprecated and pins pandas<2.0 (no wheel
for Python 3.12+). nflreadpy is the maintained nflverse successor and reads the
same underlying releases; see schema_migration_report.md for the full
comparison and verification.

nflreadpy returns polars DataFrames, so everything is polars until the write
boundary, where to_pandas() hands off to write_pandas().

Run `python ingest_nfl_data.py --dry-run` to exercise the whole data path with
zero Snowflake involvement; it writes CSVs to ./dryrun_out/ instead.

Requires: nflreadpy, polars, pandas, pyarrow (pyarrow comes in with
snowflake-connector-python). On Apple Silicon under Rosetta, install
'polars[rtcompat]' or polars will warn and may crash.
"""
import argparse
import os

import nflreadpy as nr
import polars as pl

CONN_NAME = os.getenv("SNOWFLAKE_CONNECTION_NAME") or "martydemo"
DB = "NFL_ANALYTICS"
SCHEMA = "OL_SCORING"
START_SEASON = 2018

# Columns pulled from play-by-play. PBP_PARTICIPATION_COLS live in the separate
# pbp_participation release, which nfl_data_py used to merge in silently.
PBP_KEEP = [
    'game_id','play_id','season','week','posteam','defteam','play_type',
    'yards_gained','epa','success','qb_hit','sack','rush_attempt','pass_attempt',
    'interception','fumble','penalty_team','penalty_type','down','ydstogo',
    'yardline_100','shotgun','no_huddle','qb_scramble','run_location','run_gap',
    'passer_player_name','rusher_player_name','score_differential',
    'half_seconds_remaining','wp','home_team','away_team',
    'offense_formation','offense_personnel','defenders_in_box',
    'number_of_pass_rushers','time_to_throw','was_pressure',
]

# The pbp_participation half of PBP_KEEP, with the dtypes the join produces.
#
# nflverse publishes participation RETROACTIVELY -- nflreadpy caps it at
# get_current_season(roster=True) - 1 with the comment "participation only
# available on a historical basis from FTN". So for the whole of an in-progress
# season these six columns are simply unavailable, and 2026 data backfills
# around March 2027 when the roster year rolls over.
#
# They are still written as typed nulls so RAW_PBP keeps one stable 39-column
# schema. Without that, the per-season appends below would collide: the first
# season creates a 39-column table and a later season would try to append 33.
PBP_PARTICIPATION_COLS = {
    'offense_formation': pl.String,
    'offense_personnel': pl.String,
    'defenders_in_box': pl.Int32,
    'number_of_pass_rushers': pl.Int32,
    'time_to_throw': pl.Float64,
    'was_pressure': pl.Boolean,
}


def target_seasons():
    """START_SEASON..current, bounded by the roster year.

    nflreadpy has two season clocks: get_current_season() flips on the Thursday
    after Labor Day (2026-09-10), while roster=True flips on March 15 and so
    already returns 2026. We bound on the roster year deliberately, so a newly
    published season is picked up with no code change. Seasons whose files do
    not exist yet raise, and every loader below skips them per-season.
    """
    return list(range(START_SEASON, nr.get_current_season(roster=True) + 1))


# --------------------------------------------------------------------------
# Snowflake plumbing (lazily imported so --dry-run needs no connector)
# --------------------------------------------------------------------------

def get_conn():
    import snowflake.connector
    conn = snowflake.connector.connect(connection_name=CONN_NAME)
    conn.cursor().execute(f"USE DATABASE {DB}")
    conn.cursor().execute(f"USE SCHEMA {SCHEMA}")
    conn.cursor().execute("USE WAREHOUSE COMPUTE_WH")
    return conn


def to_pandas(df: pl.DataFrame):
    """polars -> pandas, defusing the Boolean trap.

    A nullable polars Boolean becomes pandas *object* dtype, which
    clean_columns() would then stringify to 'True'/'False'/'nan' and write as
    VARCHAR. feature_engineering.sql:18 does TRY_CAST(WAS_PRESSURE AS FLOAT),
    which returns NULL for 'False' -- so PRESSURE_RATE would silently go to
    zero with no error anywhere. Cast to Int8 first.
    """
    bools = [c for c in df.columns if df.schema[c] == pl.Boolean]
    if bools:
        df = df.with_columns([pl.col(c).cast(pl.Int8) for c in bools])
    return df.to_pandas()


def clean_columns(df):
    df.columns = [c.upper().replace(" ", "_").replace(".", "_") for c in df.columns]
    # Handle real bools before the object pass, so they never get stringified.
    for col in df.select_dtypes(include=["bool", "boolean"]).columns:
        df[col] = df[col].astype("int8")
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.slice(0, 16000)
    for col in df.select_dtypes(include=["float16"]).columns:
        df[col] = df[col].astype("float32")
    return df


def safe_write(conn, df, table_name, overwrite=True, dry_run=False):
    if df.empty:
        print(f"  Skipping {table_name} — empty DataFrame")
        return
    df = clean_columns(df)
    if dry_run:
        os.makedirs("dryrun_out", exist_ok=True)
        path = f"dryrun_out/{table_name}.csv"
        df.to_csv(path, mode="w" if overwrite else "a",
                  header=overwrite, index=False)
        print(f"  [dry-run] {len(df)} rows, {len(df.columns)} cols -> {path}")
        return
    from snowflake.connector.pandas_tools import write_pandas
    print(f"  Writing {len(df)} rows to {table_name} ({len(df.columns)} cols)...")
    write_pandas(conn, df, table_name, auto_create_table=True, overwrite=overwrite)
    print(f"  {table_name} done.")


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------

def _join_participation(pbp, yr):
    """Left-join the pbp_participation release, degrading to typed nulls.

    Participation is fetched in its own try/except on purpose. It is absent for
    any in-progress season (see PBP_PARTICIPATION_COLS), and if that failure
    were allowed to propagate we would skip the season's play-by-play
    altogether -- losing sacks, QB hits, EPA and stuff rate, all of which *are*
    available live, over columns that are not.
    """
    try:
        part = nr.load_participation(seasons=[yr])
    except Exception as e:
        print(f"  {yr}: participation unavailable ({type(e).__name__}) — "
              f"writing {len(PBP_PARTICIPATION_COLS)} columns as NULL: "
              f"{', '.join(PBP_PARTICIPATION_COLS)}")
        print(f"  {yr}: >>> PRESSURE_RATE, AVG_TIME_TO_THROW and "
              f"AVG_DEFENDERS_IN_BOX will be NULL for {yr} <<<")
        return pbp.with_columns(
            [pl.lit(None, dtype=t).alias(c) for c, t in PBP_PARTICIPATION_COLS.items()
             if c not in pbp.columns])

    right = [c for c in PBP_KEEP if c in part.columns and c not in pbp.columns]
    # polars will not implicitly coerce join keys the way pandas did.
    # 2018 has pbp play_id=f64 against participation play_id=i32.
    part = part.select(["play_id", "nflverse_game_id"] + right).with_columns(
        pl.col("play_id").cast(pl.Float64),
        pl.col("nflverse_game_id").cast(pl.String))
    joined = pbp.join(part, left_on=["play_id", "game_id"],
                      right_on=["play_id", "nflverse_game_id"],
                      how="left").drop("nflverse_game_id", strict=False)
    # Backfill anything participation itself did not carry, to hold the schema.
    return joined.with_columns(
        [pl.lit(None, dtype=t).alias(c) for c, t in PBP_PARTICIPATION_COLS.items()
         if c not in joined.columns])


def load_pbp(conn, dry_run=False):
    print("Loading play-by-play data...")
    first = True
    for yr in target_seasons():
        print(f"  Fetching {yr}...")
        try:
            pbp = nr.load_pbp(seasons=[yr])
            if pbp.height == 0:
                print(f"  {yr}: no data, skipping")
                continue

            # Project before joining. Carrying all 372 pbp columns through the
            # join OOMs the process; projecting first runs 8 seasons in ~21s.
            left = [c for c in PBP_KEEP if c in pbp.columns]
            pbp = pbp.select(left).with_columns(
                pl.col("play_id").cast(pl.Float64), pl.col("game_id").cast(pl.String))

            df = _join_participation(pbp, yr)

            missing = [c for c in PBP_KEEP if c not in df.columns]
            if missing:
                print(f"  {yr}: WARNING missing {missing}")

            safe_write(conn, to_pandas(df), "RAW_PBP", overwrite=first, dry_run=dry_run)
            first = False
        except Exception as e:
            print(f"  {yr} failed: {type(e).__name__}: {e}")


def _load_per_season(label, fetch):
    """Fetch season by season, skip the ones that are not published yet.

    nflreadpy validates the whole seasons list up front and raises, so passing
    [2025, 2026] returns nothing at all rather than 2025. Looping keeps a
    single unreleased season from zeroing out the entire table.
    """
    frames = []
    for yr in target_seasons():
        try:
            df = fetch(yr)
            if df.height:
                frames.append(df)
                print(f"  {yr}: {df.height} rows")
            else:
                print(f"  {yr}: no data")
        except Exception as e:
            print(f"  {yr} skipped: {type(e).__name__}: {e}")
    if not frames:
        print(f"  {label}: nothing fetched")
        return None
    return frames[0] if len(frames) == 1 else pl.concat(frames, how="diagonal_relaxed")


def load_ngs(conn, stat_type, table_name, dry_run=False):
    print(f"Loading NGS {stat_type} data...")
    df = _load_per_season(
        table_name, lambda yr: nr.load_nextgen_stats(seasons=[yr], stat_type=stat_type))
    if df is not None:
        safe_write(conn, to_pandas(df), table_name, dry_run=dry_run)


def load_pfr(conn, stat_type, table_name, dry_run=False):
    print(f"Loading PFR {stat_type} data...")
    df = _load_per_season(
        table_name,
        lambda yr: nr.load_pfr_advstats(seasons=[yr], stat_type=stat_type,
                                        summary_level="week"))
    if df is not None:
        safe_write(conn, to_pandas(df), table_name, dry_run=dry_run)


def load_rosters(conn, dry_run=False):
    # The old code called nfl.import_rosters(), which does not exist in
    # nfl_data_py 0.3.3 -- the bare except swallowed the AttributeError, so
    # RAW_ROSTERS was never populated. load_rosters_weekly is the real
    # equivalent. Aliases below preserve the nfl_data_py column names.
    print("Loading rosters...")
    df = _load_per_season("RAW_ROSTERS", lambda yr: nr.load_rosters_weekly(seasons=[yr]))
    if df is None:
        return
    df = df.with_columns(
        pl.col("gsis_id").alias("player_id"),
        pl.col("full_name").alias("player_name"),
        # nflreadpy drops nfl_data_py's computed `age`; approximate it as of
        # Sept 1 of the season.
        ((pl.date(pl.col("season"), 9, 1) - pl.col("birth_date")).dt.total_days() / 365.25)
        .round(1).alias("age"),
    )
    safe_write(conn, to_pandas(df), "RAW_ROSTERS", dry_run=dry_run)


def _week_starts(season):
    """First regular-season gameday per week, for dating depth-chart snapshots."""
    sched = nr.load_schedules(seasons=[season]).filter(pl.col("game_type") == "REG")
    return (sched.group_by("week")
            .agg(pl.col("gameday").str.to_date().min().alias("week_start"))
            .sort("week_start"))


# Normalised depth-chart schema written to RAW_DEPTH_CHARTS. The first seven
# columns reproduce the legacy nfl_data_py names that streamlit_app.py:80-84
# still queries; the rest are native columns from whichever source applies.
DC_COLUMNS = [
    "season", "week", "club_code", "full_name", "position", "depth_team",
    "gsis_id", "game_type", "depth_position", "jersey_number",
    "pos_abb", "pos_name", "pos_grp", "pos_slot", "dt", "schema_version",
]


def _depth_chart_legacy(dc, yr):
    """2018-2024: nflverse's original 15-column weekly schema."""
    return (dc.with_columns(
                pl.col("season").cast(pl.Int32),
                pl.col("week").cast(pl.Int32),
                pl.col("depth_team").cast(pl.String),
                pl.col("jersey_number").cast(pl.Int32),
                pl.lit(None, dtype=pl.String).alias("pos_abb"),
                pl.lit(None, dtype=pl.String).alias("pos_name"),
                pl.lit(None, dtype=pl.String).alias("pos_grp"),
                pl.lit(None, dtype=pl.Int32).alias("pos_slot"),
                pl.lit(None, dtype=pl.String).alias("dt"),
                pl.lit("legacy").alias("schema_version"))
              .select(DC_COLUMNS))


def _depth_chart_current(dc, yr):
    """2025+: nflverse rebuilt this release on a new, timestamped source.

    season/week/club_code/full_name/position/depth_team no longer exist, so we
    re-derive them and keep the legacy names, which lets streamlit_app.py keep
    working unchanged:

        full_name  <- player_name
        position   <- pos_abb   (LT/LG/C/RG/RT, all already in the app's IN list)
        depth_team <- pos_rank  (1 = starter; cast to String to match legacy)
        club_code  <- team
        week       <- derived from `dt` against the schedule

    The new feed also snapshots many times a day rather than once a week
    (~554k rows/season against ~37k), so we keep only the latest snapshot per
    team per week. That restores the legacy one-chart-per-team-per-week grain
    and keeps the table roughly its previous size.
    """
    wk = _week_starts(yr)
    max_week = int(wk["week"].max())
    dc = (dc.with_columns(pl.col("dt").str.slice(0, 10).str.to_date().alias("snap_date"))
            .sort("snap_date")
            # A depth chart applies to the next scheduled week, so take the
            # first week starting on or after the snapshot date.
            .join_asof(wk, left_on="snap_date", right_on="week_start",
                       strategy="forward")
            .with_columns(pl.col("week").fill_null(max_week).cast(pl.Int32)))
    # Latest snapshot per team per week.
    dc = dc.filter(pl.col("dt") == pl.col("dt").max().over(["week", "team"]))
    return (dc.with_columns(
                pl.lit(yr).cast(pl.Int32).alias("season"),
                pl.col("team").alias("club_code"),
                pl.col("player_name").alias("full_name"),
                pl.col("pos_abb").alias("position"),
                pl.col("pos_rank").cast(pl.String).alias("depth_team"),
                pl.lit(None, dtype=pl.String).alias("game_type"),
                pl.lit(None, dtype=pl.String).alias("depth_position"),
                pl.lit(None, dtype=pl.Int32).alias("jersey_number"),
                pl.col("pos_slot").cast(pl.Int32),
                pl.lit("2025+").alias("schema_version"))
              .select(DC_COLUMNS)), max_week


def load_depth_charts(conn, dry_run=False):
    """Two incompatible source schemas, normalised into one table.

    nflverse switched this release at the 2025 season: 2018-2024 still carry
    the legacy weekly schema, 2025+ carry the new timestamped one. Both
    libraries return whichever applies, so this split is not a migration
    artefact -- nfl_data_py has the same problem.
    """
    print("Loading depth charts...")
    frames = []
    for yr in target_seasons():
        try:
            dc = nr.load_depth_charts(seasons=[yr])
            if not dc.height:
                print(f"  {yr}: no data")
                continue
            if "dt" in dc.columns:
                out, max_week = _depth_chart_current(dc, yr)
                print(f"  {yr}: {dc.height} raw -> {out.height} rows "
                      f"(latest per team/week, weeks 1-{max_week}) [2025+ schema]")
            else:
                out = _depth_chart_legacy(dc, yr)
                print(f"  {yr}: {out.height} rows [legacy schema]")
            frames.append(out)
        except Exception as e:
            print(f"  {yr} skipped: {type(e).__name__}: {e}")
    if not frames:
        print("  RAW_DEPTH_CHARTS: nothing fetched")
        return
    df = frames[0] if len(frames) == 1 else pl.concat(frames, how="vertical")
    safe_write(conn, to_pandas(df), "RAW_DEPTH_CHARTS", dry_run=dry_run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="write CSVs to ./dryrun_out/ instead of Snowflake")
    args = ap.parse_args()

    print(f"nflreadpy {nr.__version__} | seasons {target_seasons()}")
    print(f"current season={nr.get_current_season()} roster year={nr.get_current_season(roster=True)}\n")

    conn = None if args.dry_run else get_conn()
    try:
        load_pbp(conn, args.dry_run)
        load_ngs(conn, 'passing', 'RAW_NGS_PASSING', args.dry_run)
        load_ngs(conn, 'rushing', 'RAW_NGS_RUSHING', args.dry_run)
        load_pfr(conn, 'pass', 'RAW_PFR_PASS', args.dry_run)
        load_pfr(conn, 'rush', 'RAW_PFR_RUSH', args.dry_run)
        load_rosters(conn, args.dry_run)
        load_depth_charts(conn, args.dry_run)
        print("\nAll data loaded successfully!")
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
