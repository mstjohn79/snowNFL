import os
import nfl_data_py as nfl
import snowflake.connector
import pandas as pd
from snowflake.connector.pandas_tools import write_pandas

CONN_NAME = os.getenv("SNOWFLAKE_CONNECTION_NAME") or "martydemo"
DB = "NFL_ANALYTICS"
SCHEMA = "OL_SCORING"
SEASONS = list(range(2018, 2026))

def get_conn():
    conn = snowflake.connector.connect(connection_name=CONN_NAME)
    conn.cursor().execute(f"USE DATABASE {DB}")
    conn.cursor().execute(f"USE SCHEMA {SCHEMA}")
    conn.cursor().execute("USE WAREHOUSE COMPUTE_WH")
    return conn

def clean_columns(df):
    df.columns = [c.upper().replace(" ", "_").replace(".", "_") for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.slice(0, 16000)
    for col in df.select_dtypes(include=["float16"]).columns:
        df[col] = df[col].astype("float32")
    bool_cols = df.select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        df[col] = df[col].astype("int8")
    return df

def safe_write(conn, df, table_name, overwrite=True):
    if df.empty:
        print(f"  Skipping {table_name} — empty DataFrame")
        return
    df = clean_columns(df)
    print(f"  Writing {len(df)} rows to {table_name} ({len(df.columns)} cols)...")
    write_pandas(conn, df, table_name, auto_create_table=True, overwrite=overwrite)
    print(f"  {table_name} done.")

def load_pbp(conn):
    print("Loading play-by-play data...")
    keep_cols = [
        'game_id','play_id','season','week','posteam','defteam','play_type',
        'yards_gained','epa','success','qb_hit','sack','rush_attempt','pass_attempt',
        'interception','fumble','penalty_team','penalty_type','down','ydstogo',
        'yardline_100','shotgun','no_huddle','qb_scramble','run_location','run_gap',
        'passer_player_name','rusher_player_name','score_differential',
        'half_seconds_remaining','wp','home_team','away_team',
        'offense_formation','offense_personnel','defenders_in_box',
        'number_of_pass_rushers','time_to_throw','was_pressure',
    ]
    first = True
    for yr in SEASONS:
        print(f"  Fetching {yr}...")
        try:
            df = nfl.import_pbp_data([yr])
            available = [c for c in keep_cols if c in df.columns]
            df = df[available].copy()
            if df.empty:
                print(f"  {yr}: no data, skipping")
                continue
            safe_write(conn, df, "RAW_PBP", overwrite=first)
            first = False
        except Exception as e:
            print(f"  {yr} failed: {e}")

def load_ngs(conn, stat_type, table_name):
    print(f"Loading NGS {stat_type} data...")
    try:
        df = nfl.import_ngs_data(stat_type, SEASONS)
        safe_write(conn, df, table_name)
    except Exception as e:
        print(f"  NGS {stat_type} failed: {e}")

def load_pfr(conn, stat_type, table_name):
    print(f"Loading PFR {stat_type} data...")
    try:
        df = nfl.import_weekly_pfr(stat_type, SEASONS)
        safe_write(conn, df, table_name)
    except Exception as e:
        print(f"  PFR {stat_type} failed: {e}")

def load_rosters(conn):
    print("Loading rosters...")
    try:
        df = nfl.import_rosters(SEASONS)
        safe_write(conn, df, "RAW_ROSTERS")
    except Exception as e:
        print(f"  Rosters failed: {e}")

def load_depth_charts(conn):
    print("Loading depth charts...")
    try:
        df = nfl.import_depth_charts(SEASONS)
        safe_write(conn, df, "RAW_DEPTH_CHARTS")
    except Exception as e:
        print(f"  Depth charts failed: {e}")

def main():
    conn = get_conn()
    try:
        load_pbp(conn)
        load_ngs(conn, 'passing', 'RAW_NGS_PASSING')
        load_ngs(conn, 'rushing', 'RAW_NGS_RUSHING')
        load_pfr(conn, 'pass', 'RAW_PFR_PASS')
        load_pfr(conn, 'rush', 'RAW_PFR_RUSH')
        load_rosters(conn)
        load_depth_charts(conn)
        print("\nAll data loaded successfully!")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
