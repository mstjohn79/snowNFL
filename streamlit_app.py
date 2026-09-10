import streamlit as st
import pandas as pd
from datetime import timedelta

st.set_page_config(page_title="NFL O-Line Scoring", page_icon="🏈", layout="wide")

# ---------------------------------------------------------------------------
# Connection wrapper: works both locally (st.connection) and in SiS
# (get_active_session).  Provides a .query(sql, params=) interface so all
# downstream code stays the same.
# ---------------------------------------------------------------------------
try:
    from snowflake.snowpark.context import get_active_session
    _session = get_active_session()

    class _SiSConnection:
        """Thin wrapper so conn.query() works in Streamlit-in-Snowflake."""
        def query(self, sql, params=None):
            if params:
                # Replace in reverse order so :10 is replaced before :1
                for i in range(len(params), 0, -1):
                    v = params[i - 1]
                    sql = sql.replace(f":{i}", "'" + str(v).replace("'", "''") + "'")
            return _session.sql(sql).to_pandas()

    conn = _SiSConnection()
except Exception:
    conn = st.connection("snowflake")

@st.cache_data(ttl=timedelta(minutes=30))
def get_season_rankings(season):
    return conn.query(
        "SELECT * FROM NFL_ANALYTICS.OL_SCORING.TEAM_OL_SEASON_RANKINGS WHERE SEASON = :1 ORDER BY SEASON_RANK",
        params=[season],
    )

@st.cache_data(ttl=timedelta(minutes=30))
def get_team_game_scores(team, season):
    return conn.query(
        "SELECT * FROM NFL_ANALYTICS.OL_SCORING.TEAM_OL_SCORES WHERE TEAM = :1 AND SEASON = :2 ORDER BY WEEK",
        params=[team, season],
    )

@st.cache_data(ttl=timedelta(minutes=30))
def get_all_teams(season):
    return conn.query(
        "SELECT DISTINCT TEAM FROM NFL_ANALYTICS.OL_SCORING.TEAM_OL_SEASON_RANKINGS WHERE SEASON = :1 ORDER BY TEAM",
        params=[season],
    )

@st.cache_data(ttl=timedelta(minutes=30))
def get_available_seasons():
    return conn.query(
        "SELECT DISTINCT SEASON FROM NFL_ANALYTICS.OL_SCORING.TEAM_OL_SEASON_RANKINGS ORDER BY SEASON DESC"
    )

@st.cache_data(ttl=timedelta(minutes=10))
def get_ai_summary(team, season, week):
    return conn.query(
        """SELECT AI_SUMMARY, COMPOSITE_OL_SCORE, PASS_BLOCK_SCORE, RUN_BLOCK_SCORE
           FROM NFL_ANALYTICS.OL_SCORING.GAME_AI_SUMMARIES
           WHERE TEAM = :1 AND SEASON = :2 AND WEEK = :3 LIMIT 1""",
        params=[team, season, week],
    )

@st.cache_data(ttl=timedelta(minutes=30))
def get_matchup_data(team1, team2, season):
    return conn.query(
        """SELECT TEAM, WEEK, COMPOSITE_OL_SCORE, PASS_BLOCK_SCORE, RUN_BLOCK_SCORE,
                  ROLLING_3_OL_SCORE, ROLLING_5_OL_SCORE
           FROM NFL_ANALYTICS.OL_SCORING.TEAM_OL_ROLLING
           WHERE TEAM IN (:1, :2) AND SEASON = :3
           ORDER BY WEEK""",
        params=[team1, team2, season],
    )

@st.cache_data(ttl=timedelta(minutes=30))
def get_player_depth_chart(team, season):
    return conn.query(
        """SELECT FULL_NAME, POSITION, DEPTH_TEAM, WEEK
           FROM NFL_ANALYTICS.OL_SCORING.RAW_DEPTH_CHARTS
           WHERE CLUB_CODE = :1 AND SEASON = :2
             AND POSITION IN ('C','OG','OT','G','T','LT','RT','LG','RG')
           ORDER BY WEEK DESC, POSITION, DEPTH_TEAM
           LIMIT 200""",
        params=[team, season],
    )

@st.cache_data(ttl=timedelta(minutes=30))
def get_ml_predictions():
    return conn.query(
        """SELECT TEAM, SEASON, WEEK,
                  ROUND(PREDICTED_OL_SCORE, 1) AS PREDICTED_OL_SCORE,
                  ROUND(PREDICTED_PASS_BLOCK, 1) AS PREDICTED_PASS_BLOCK,
                  ROUND(PREDICTED_RUN_BLOCK, 1) AS PREDICTED_RUN_BLOCK
           FROM NFL_ANALYTICS.OL_SCORING.ML_NEXT_GAME_PREDICTIONS
           ORDER BY PREDICTED_OL_SCORE DESC"""
    )

@st.cache_data(ttl=timedelta(minutes=30))
def get_ml_prediction_for_team(team):
    return conn.query(
        """SELECT PREDICTED_OL_SCORE, PREDICTED_PASS_BLOCK, PREDICTED_RUN_BLOCK
           FROM NFL_ANALYTICS.OL_SCORING.ML_NEXT_GAME_PREDICTIONS
           WHERE TEAM = :1 LIMIT 1""",
        params=[team],
    )

@st.cache_data(ttl=timedelta(minutes=30))
def get_model_metrics():
    return conn.query(
        """SELECT 'Composite OL' AS MODEL,
                  NFL_OL_COMPOSITE_PREDICTOR!GET_METRICS() AS METRICS
           UNION ALL
           SELECT 'Pass Block',
                  NFL_PASS_BLOCK_PREDICTOR!GET_METRICS()
           UNION ALL
           SELECT 'Run Block',
                  NFL_RUN_BLOCK_PREDICTOR!GET_METRICS()"""
    )

@st.cache_data(ttl=timedelta(minutes=30))
def get_situational_season(season):
    return conn.query(
        """SELECT s.TEAM,
                  ROUND(AVG(s.EARLY_DOWN_EPA), 3) AS AVG_EARLY_DOWN_EPA,
                  ROUND(AVG(s.LATE_DOWN_EPA), 3) AS AVG_LATE_DOWN_EPA,
                  ROUND(AVG(s.EARLY_VS_LATE_EPA_DIFF), 3) AS AVG_CLUTCH_DIFF,
                  ROUND(AVG(s.THIRD_SHORT_SUCCESS), 3) AS AVG_3RD_SHORT_SUCCESS,
                  ROUND(AVG(s.THIRD_LONG_SACK_RATE), 3) AS AVG_3RD_LONG_SACK_RATE,
                  ROUND(AVG(s.THIRD_LONG_PRESSURE_RATE), 3) AS AVG_3RD_LONG_PRESSURE_RATE,
                  ROUND(AVG(s.REDZONE_EPA), 3) AS AVG_REDZONE_EPA,
                  ROUND(AVG(s.REDZONE_STUFF_RATE), 3) AS AVG_REDZONE_STUFF_RATE,
                  ROUND(AVG(s.REDZONE_SACK_RATE), 3) AS AVG_REDZONE_SACK_RATE,
                  ROUND(AVG(s.OWN_TERRITORY_EPA), 3) AS AVG_OWN_TERRITORY_EPA,
                  ROUND(AVG(s.SITUATIONAL_EPA_VARIANCE), 3) AS AVG_VARIANCE
           FROM NFL_ANALYTICS.OL_SCORING.TEAM_SITUATIONAL_FEATURES s
           WHERE s.SEASON = :1
           GROUP BY s.TEAM
           ORDER BY AVG_REDZONE_EPA DESC""",
        params=[season],
    )

@st.cache_data(ttl=timedelta(minutes=30))
def get_situational_weekly(team, season):
    return conn.query(
        """SELECT WEEK, EARLY_DOWN_EPA, LATE_DOWN_EPA, EARLY_VS_LATE_EPA_DIFF,
                  THIRD_SHORT_SUCCESS, THIRD_LONG_SACK_RATE,
                  REDZONE_EPA, REDZONE_STUFF_RATE, OWN_TERRITORY_EPA
           FROM NFL_ANALYTICS.OL_SCORING.TEAM_SITUATIONAL_FEATURES
           WHERE TEAM = :1 AND SEASON = :2
           ORDER BY WEEK""",
        params=[team, season],
    )

@st.cache_data(ttl=timedelta(minutes=30))
def get_clutch_season(season):
    return conn.query(
        """SELECT TEAM,
                  ROUND(AVG(OL_CLUTCH_INDEX), 4) AS AVG_CLUTCH_INDEX,
                  ROUND(AVG(OL_SUCCESS_RATE), 4) AS AVG_OL_SUCCESS_RATE,
                  ROUND(AVG(OL_CLEAN_POCKET_RATE), 4) AS AVG_CLEAN_POCKET_RATE,
                  ROUND(AVG(OL_HIGH_LEVERAGE_SUCCESS), 4) AS AVG_HIGH_LEV_SUCCESS,
                  ROUND(AVG(OL_LOW_LEVERAGE_SUCCESS), 4) AS AVG_LOW_LEV_SUCCESS,
                  ROUND(AVG(OL_HIGH_LEVERAGE_EPA), 3) AS AVG_HIGH_LEV_EPA,
                  ROUND(AVG(OL_PROTECTION_CLUTCH), 4) AS AVG_PROTECTION_CLUTCH,
                  ROUND(AVG(OL_RUN_BLOCKING_CLUTCH), 4) AS AVG_RUN_CLUTCH,
                  ROUND(AVG(OL_RESPONSIBILITY_RATE), 4) AS AVG_RESP_RATE
           FROM NFL_ANALYTICS.OL_SCORING.TEAM_OL_CLUTCH_FEATURES
           WHERE SEASON = :1
           GROUP BY TEAM
           ORDER BY AVG_CLUTCH_INDEX DESC""",
        params=[season],
    )

@st.cache_data(ttl=timedelta(minutes=30))
def get_clutch_weekly(team, season):
    return conn.query(
        """SELECT WEEK, OL_CLUTCH_INDEX, OL_SUCCESS_RATE, OL_CLEAN_POCKET_RATE,
                  OL_HIGH_LEVERAGE_SUCCESS, OL_LOW_LEVERAGE_SUCCESS,
                  OL_HIGH_LEVERAGE_EPA, OL_PROTECTION_CLUTCH, OL_RUN_BLOCKING_CLUTCH
           FROM NFL_ANALYTICS.OL_SCORING.TEAM_OL_CLUTCH_FEATURES
           WHERE TEAM = :1 AND SEASON = :2
           ORDER BY WEEK""",
        params=[team, season],
    )

@st.cache_data(ttl=timedelta(minutes=30))
def get_clutch_league_trend():
    return conn.query(
        """SELECT SEASON,
                  ROUND(AVG(OL_CLUTCH_INDEX), 4) AS AVG_CLUTCH_INDEX,
                  ROUND(STDDEV(OL_CLUTCH_INDEX), 4) AS STD_CLUTCH_INDEX,
                  ROUND(AVG(OL_CLEAN_POCKET_RATE), 4) AS AVG_CLEAN_POCKET_RATE,
                  ROUND(AVG(OL_SUCCESS_RATE), 4) AS AVG_OL_SUCCESS_RATE,
                  ROUND(AVG(OL_PROTECTION_CLUTCH), 4) AS AVG_PROTECTION_CLUTCH,
                  ROUND(AVG(OL_RUN_BLOCKING_CLUTCH), 4) AS AVG_RUN_CLUTCH,
                  COUNT(DISTINCT TEAM) AS TEAMS,
                  COUNT(*) AS GAMES
           FROM NFL_ANALYTICS.OL_SCORING.TEAM_OL_CLUTCH_FEATURES
           GROUP BY SEASON
           ORDER BY SEASON"""
    )

@st.cache_data(ttl=timedelta(minutes=30))
def get_clutch_team_history(team):
    return conn.query(
        """SELECT SEASON,
                  ROUND(AVG(OL_CLUTCH_INDEX), 4) AS AVG_CLUTCH_INDEX,
                  ROUND(AVG(OL_CLEAN_POCKET_RATE), 4) AS AVG_CLEAN_POCKET_RATE,
                  ROUND(AVG(OL_SUCCESS_RATE), 4) AS AVG_OL_SUCCESS_RATE,
                  ROUND(AVG(OL_HIGH_LEVERAGE_SUCCESS), 4) AS AVG_HIGH_LEV_SUCCESS,
                  ROUND(AVG(OL_PROTECTION_CLUTCH), 4) AS AVG_PROTECTION_CLUTCH,
                  ROUND(AVG(OL_RUN_BLOCKING_CLUTCH), 4) AS AVG_RUN_CLUTCH
           FROM NFL_ANALYTICS.OL_SCORING.TEAM_OL_CLUTCH_FEATURES
           WHERE TEAM = :1
           GROUP BY SEASON
           ORDER BY SEASON""",
        params=[team],
    )

def get_ai_ol_answer(question, team, season):
    return conn.query(
        """WITH team_data AS (
               SELECT TEAM,
                      ROUND(AVG(COMPOSITE_OL_SCORE),1) AS AVG_OL_SCORE,
                      ROUND(AVG(PASS_BLOCK_SCORE),1) AS AVG_PASS_BLOCK,
                      ROUND(AVG(RUN_BLOCK_SCORE),1) AS AVG_RUN_BLOCK,
                      ROUND(AVG(SACK_RATE)*100,1) AS AVG_SACK_RATE,
                      ROUND(AVG(PRESSURE_RATE)*100,1) AS AVG_PRESSURE_RATE,
                      ROUND(AVG(STUFF_RATE)*100,1) AS AVG_STUFF_RATE,
                      ROUND(AVG(AVG_RUSH_YARDS),2) AS AVG_YPC,
                      ROUND(AVG(OL_CLUTCH_INDEX),4) AS AVG_CLUTCH_INDEX,
                      ROUND(AVG(OL_CLEAN_POCKET_RATE)*100,1) AS AVG_CLEAN_POCKET,
                      ROUND(AVG(OL_SUCCESS_RATE)*100,1) AS AVG_OL_SUCCESS,
                      ROUND(AVG(OL_PROTECTION_CLUTCH),4) AS AVG_PROT_CLUTCH,
                      ROUND(AVG(OL_RUN_BLOCKING_CLUTCH),4) AS AVG_RUN_CLUTCH,
                      ROUND(AVG(OL_HIGH_LEVERAGE_SUCCESS)*100,1) AS HIGH_LEV_SUCCESS,
                      COUNT(*) AS GAMES
               FROM NFL_ANALYTICS.OL_SCORING.TEAM_OL_SCORES
               WHERE TEAM = :1 AND SEASON = :2
               GROUP BY TEAM
           )
           SELECT SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b',
               'You are an expert NFL offensive line analyst. Answer the following question using ONLY the data provided. ' ||
               'Be specific, cite numbers, and give analytical insight. ' ||
               'Team: ' || TEAM || ' (' || :2 || ' season, ' || GAMES || ' games). ' ||
               'OL Score: ' || AVG_OL_SCORE || '/100. Pass Block: ' || AVG_PASS_BLOCK || '/100. Run Block: ' || AVG_RUN_BLOCK || '/100. ' ||
               'Sack Rate: ' || AVG_SACK_RATE || '%. Pressure Rate: ' || AVG_PRESSURE_RATE || '%. Stuff Rate: ' || AVG_STUFF_RATE || '%. ' ||
               'Avg YPC: ' || AVG_YPC || '. Clean Pocket Rate: ' || AVG_CLEAN_POCKET || '%. ' ||
               'OL Success Rate: ' || AVG_OL_SUCCESS || '%. ' ||
               'Clutch Index: ' || AVG_CLUTCH_INDEX || ' (positive=better in big moments). ' ||
               'High Leverage OL Success: ' || HIGH_LEV_SUCCESS || '%. ' ||
               'Pass Protection Clutch: ' || AVG_PROT_CLUTCH || ' (positive=more pressure in clutch). ' ||
               'Run Blocking Clutch: ' || AVG_RUN_CLUTCH || ' (positive=more stuffs in clutch). ' ||
               'Question: ' || :3
           ) AS AI_ANSWER
           FROM team_data""",
        params=[team, season, question],
    )

seasons = get_available_seasons()
season_list = seasons["SEASON"].tolist()

with st.sidebar:
    st.title("🏈 NFL O-Line Scorer")
    selected_season = st.selectbox("Season", season_list, index=0)
    page = st.radio("Navigate", [
        "Team Rankings", "Team Deep Dive", "Situational Analysis",
        "OL Clutch Index", "Matchup Predictor", "ML Predictions",
        "Game Summary", "Ask the OL Analyst", "Methodology",
    ])

def score_color(score):
    if score >= 70:
        return "🟢"
    elif score >= 50:
        return "🟡"
    else:
        return "🔴"

if page == "Team Rankings":
    st.header(f"NFL Offensive Line Rankings — {selected_season}")

    df = get_season_rankings(selected_season)

    top3 = df.head(3)
    cols = st.columns(3)
    for idx, (_, row) in enumerate(top3.iterrows()):
        with cols[idx]:
            st.metric(
                f"#{int(row['SEASON_RANK'])} {row['TEAM']}",
                f"{row['AVG_OL_SCORE']}",
                f"Pass: {row['AVG_PASS_BLOCK']} | Run: {row['AVG_RUN_BLOCK']}",
            )

    st.subheader("Full Rankings")
    display_df = df[[
        "SEASON_RANK", "TEAM", "GAMES", "AVG_OL_SCORE", "AVG_PASS_BLOCK",
        "AVG_RUN_BLOCK", "AVG_SACK_RATE", "AVG_PRESSURE_RATE",
        "AVG_STUFF_RATE", "AVG_YPC", "AVG_OL_PENALTIES_PER_GAME"
    ]].copy()
    display_df.columns = [
        "Rank", "Team", "Games", "OL Score", "Pass Block", "Run Block",
        "Sack Rate", "Pressure Rate", "Stuff Rate", "YPC", "Penalties/Game"
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
    )

elif page == "Team Deep Dive":
    st.header(f"Team Deep Dive — {selected_season}")

    teams = get_all_teams(selected_season)
    selected_team = st.selectbox("Select Team", teams["TEAM"].tolist())

    if selected_team:
        games = get_team_game_scores(selected_team, selected_season)

        if not games.empty:
            avg_ol = round(games["COMPOSITE_OL_SCORE"].mean(), 1)
            avg_pass = round(games["PASS_BLOCK_SCORE"].mean(), 1)
            avg_run = round(games["RUN_BLOCK_SCORE"].mean(), 1)
            avg_sack = round(games["SACK_RATE"].mean() * 100, 1)

            cols = st.columns(4)
            with cols[0]:
                st.metric("Avg OL Score", f"{avg_ol}")
            with cols[1]:
                st.metric("Pass Block", f"{avg_pass}")
            with cols[2]:
                st.metric("Run Block", f"{avg_run}")
            with cols[3]:
                st.metric("Sack Rate", f"{avg_sack}%")

            col1, col2 = st.columns(2)
            with col1:
                with st.container():
                    st.subheader("OL Score by Week")
                    chart_data = games[["WEEK", "COMPOSITE_OL_SCORE", "PASS_BLOCK_SCORE", "RUN_BLOCK_SCORE"]].set_index("WEEK")
                    chart_data.columns = ["Composite", "Pass Block", "Run Block"]
                    st.line_chart(chart_data)

            with col2:
                with st.container():
                    st.subheader("Key Metrics by Week")
                    metrics_data = games[["WEEK", "SACK_RATE", "PRESSURE_RATE", "STUFF_RATE"]].copy()
                    metrics_data["SACK_RATE"] = metrics_data["SACK_RATE"] * 100
                    metrics_data["PRESSURE_RATE"] = metrics_data["PRESSURE_RATE"] * 100
                    metrics_data["STUFF_RATE"] = metrics_data["STUFF_RATE"] * 100
                    metrics_data = metrics_data.set_index("WEEK")
                    metrics_data.columns = ["Sack Rate %", "Pressure Rate %", "Stuff Rate %"]
                    st.line_chart(metrics_data)

            with st.container():
                st.subheader("Game Log")
                game_log = games[[
                    "WEEK", "OPPONENT", "COMPOSITE_OL_SCORE", "PASS_BLOCK_SCORE",
                    "RUN_BLOCK_SCORE", "SACKS_ALLOWED", "DROPBACKS",
                    "RUSH_ATTEMPTS", "AVG_RUSH_YARDS", "OL_PENALTIES"
                ]].copy()
                game_log.columns = [
                    "Week", "Opponent", "OL Score", "Pass Block", "Run Block",
                    "Sacks", "Dropbacks", "Rush Att", "Avg Rush Yds", "Penalties"
                ]
                st.dataframe(game_log, use_container_width=True)

            st.subheader("O-Line Starters")
            depth = get_player_depth_chart(selected_team, selected_season)
            if not depth.empty:
                latest_week = depth["WEEK"].max()
                starters = depth[(depth["WEEK"] == latest_week) & (depth["DEPTH_TEAM"].astype(str) == "1")]
                if not starters.empty:
                    st.dataframe(
                        starters[["FULL_NAME", "POSITION"]].drop_duplicates(),
                        use_container_width=True,
                    )
                else:
                    st.info("No depth chart starters found for this week.")
            else:
                st.info("No depth chart data available.")

elif page == "Situational Analysis":
    st.header(f"Situational OL Performance — {selected_season}")
    st.caption("How does each O-line perform by down & distance, field position, and pressure situations?")

    sit = get_situational_season(selected_season)

    if not sit.empty:
        tab1, tab2, tab3 = st.tabs(["Redzone & Field Position", "Down & Distance", "Team Drill-Down"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                with st.container():
                    st.subheader("Redzone EPA by Team")
                    rz = sit[["TEAM", "AVG_REDZONE_EPA"]].set_index("TEAM").sort_values("AVG_REDZONE_EPA", ascending=False)
                    st.bar_chart(rz)
            with col2:
                with st.container():
                    st.subheader("Own Territory EPA (Backs Against the Wall)")
                    ot = sit[["TEAM", "AVG_OWN_TERRITORY_EPA"]].set_index("TEAM").sort_values("AVG_OWN_TERRITORY_EPA", ascending=False)
                    st.bar_chart(ot)

            with st.container():
                st.subheader("Redzone Breakdown")
                rz_detail = sit[["TEAM", "AVG_REDZONE_EPA", "AVG_REDZONE_STUFF_RATE", "AVG_REDZONE_SACK_RATE"]].copy()
                rz_detail.columns = ["Team", "Redzone EPA", "Redzone Stuff Rate", "Redzone Sack Rate"]
                rz_detail = rz_detail.sort_values("Redzone EPA", ascending=False)
                st.dataframe(rz_detail, use_container_width=True)

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                with st.container():
                    st.subheader("3rd & Short Success Rate")
                    ts = sit[["TEAM", "AVG_3RD_SHORT_SUCCESS"]].set_index("TEAM").sort_values("AVG_3RD_SHORT_SUCCESS", ascending=False)
                    st.bar_chart(ts)
            with col2:
                with st.container():
                    st.subheader("3rd & Long Sack Rate (Lower = Better)")
                    tl = sit[["TEAM", "AVG_3RD_LONG_SACK_RATE"]].set_index("TEAM").sort_values("AVG_3RD_LONG_SACK_RATE", ascending=True)
                    st.bar_chart(tl)

            with st.container():
                st.subheader("Early vs Late Down EPA Gap")
                st.caption("Positive = better on early downs, Negative = better under pressure on late downs")
                clutch = sit[["TEAM", "AVG_EARLY_DOWN_EPA", "AVG_LATE_DOWN_EPA", "AVG_CLUTCH_DIFF"]].copy()
                clutch.columns = ["Team", "Early Down EPA", "Late Down EPA", "Clutch Gap"]
                clutch = clutch.sort_values("Clutch Gap", ascending=True)
                st.dataframe(clutch, use_container_width=True)

        with tab3:
            teams = get_all_teams(selected_season)
            sel_team = st.selectbox("Select Team", teams["TEAM"].tolist(), key="sit_team")
            if sel_team:
                weekly = get_situational_weekly(sel_team, selected_season)
                if not weekly.empty:
                    rz_avg = round(weekly["REDZONE_EPA"].mean(), 2)
                    early_avg = round(weekly["EARLY_DOWN_EPA"].mean(), 2)
                    short_avg = round(weekly["THIRD_SHORT_SUCCESS"].dropna().mean() * 100, 1)
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Avg Redzone EPA", f"{rz_avg}")
                    with cols[1]:
                        st.metric("Avg Early Down EPA", f"{early_avg}")
                    with cols[2]:
                        st.metric("3rd & Short Success", f"{short_avg}%")

                    col1, col2 = st.columns(2)
                    with col1:
                        with st.container():
                            st.subheader("Redzone EPA by Week")
                            st.line_chart(weekly[["WEEK", "REDZONE_EPA"]].set_index("WEEK"))
                    with col2:
                        with st.container():
                            st.subheader("Early vs Late Down EPA")
                            epa_chart = weekly[["WEEK", "EARLY_DOWN_EPA", "LATE_DOWN_EPA"]].set_index("WEEK")
                            epa_chart.columns = ["Early Down", "Late Down"]
                            st.line_chart(epa_chart)

                    with st.container():
                        st.subheader("Weekly Situational Detail")
                        detail = weekly.copy()
                        detail.columns = ["Week", "Early Down EPA", "Late Down EPA", "Clutch Gap",
                                          "3rd Short Success", "3rd Long Sack Rate",
                                          "Redzone EPA", "Redzone Stuff Rate", "Own Territory EPA"]
                        st.dataframe(detail, use_container_width=True)
                else:
                    st.info("No situational data available for this team/season.")
    else:
        st.warning("No situational data available for this season.")

elif page == "OL Clutch Index":
    st.header(f"OL Responsibility & Clutch Index — {selected_season}")
    st.caption("Measures whether an OL performs better or worse in high-leverage situations (3rd/4th down, close game, 2nd half)")

    clutch = get_clutch_season(selected_season)

    if not clutch.empty:
        tab1, tab2, tab3 = st.tabs(["League Overview", "Team Drill-Down", "Season Story"])

        with tab1:
            best = clutch.iloc[0]
            worst = clutch.iloc[-1]
            league_avg = round(clutch["AVG_CLUTCH_INDEX"].mean(), 4)
            cols = st.columns(4)
            with cols[0]:
                st.metric("Most Clutch OL", f"{best['TEAM']}", f"{best['AVG_CLUTCH_INDEX']:+.4f}")
            with cols[1]:
                st.metric("Least Clutch OL", f"{worst['TEAM']}", f"{worst['AVG_CLUTCH_INDEX']:+.4f}")
            with cols[2]:
                st.metric("League Avg Clutch", f"{league_avg:+.4f}")
            with cols[3]:
                st.metric("Avg Clean Pocket Rate", f"{clutch['AVG_CLEAN_POCKET_RATE'].mean():.1%}")

            col1, col2 = st.columns(2)
            with col1:
                with st.container():
                    st.subheader("OL Clutch Index by Team")
                    st.caption("Positive = rises in big moments, Negative = chokes under pressure")
                    ci = clutch[["TEAM", "AVG_CLUTCH_INDEX"]].set_index("TEAM")
                    st.bar_chart(ci)
            with col2:
                with st.container():
                    st.subheader("High vs Low Leverage OL Success Rate")
                    hl = clutch[["TEAM", "AVG_HIGH_LEV_SUCCESS", "AVG_LOW_LEV_SUCCESS"]].set_index("TEAM")
                    hl.columns = ["High Leverage", "Low Leverage"]
                    st.bar_chart(hl)

            col1, col2 = st.columns(2)
            with col1:
                with st.container():
                    st.subheader("Pass Protection Clutch")
                    st.caption("Negative = lower pressure rate in clutch (good)")
                    pc = clutch[["TEAM", "AVG_PROTECTION_CLUTCH"]].set_index("TEAM").sort_values("AVG_PROTECTION_CLUTCH")
                    st.bar_chart(pc)
            with col2:
                with st.container():
                    st.subheader("Run Blocking Clutch")
                    st.caption("Negative = lower stuff rate in clutch (good)")
                    rc = clutch[["TEAM", "AVG_RUN_CLUTCH"]].set_index("TEAM").sort_values("AVG_RUN_CLUTCH")
                    st.bar_chart(rc)

            with st.container():
                st.subheader("Full Clutch Breakdown")
                display = clutch.copy()
                display.columns = [
                    "Team", "Clutch Index", "OL Success Rate", "Clean Pocket Rate",
                    "High Lev Success", "Low Lev Success", "High Lev EPA",
                    "Pass Prot Clutch", "Run Block Clutch", "Responsibility Rate"
                ]
                st.dataframe(display, use_container_width=True)

        with tab2:
            teams = get_all_teams(selected_season)
            sel_team = st.selectbox("Select Team", teams["TEAM"].tolist(), key="clutch_team")
            if sel_team:
                weekly = get_clutch_weekly(sel_team, selected_season)
                if not weekly.empty:
                    avg_clutch = round(weekly["OL_CLUTCH_INDEX"].mean(), 4)
                    avg_pocket = round(weekly["OL_CLEAN_POCKET_RATE"].mean() * 100, 1)
                    avg_success = round(weekly["OL_SUCCESS_RATE"].mean() * 100, 1)
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Avg Clutch Index", f"{avg_clutch:+.4f}")
                    with cols[1]:
                        st.metric("Avg Clean Pocket", f"{avg_pocket}%")
                    with cols[2]:
                        st.metric("OL Success Rate", f"{avg_success}%")

                    col1, col2 = st.columns(2)
                    with col1:
                        with st.container():
                            st.subheader("Clutch Index by Week")
                            st.line_chart(weekly[["WEEK", "OL_CLUTCH_INDEX"]].set_index("WEEK"))
                    with col2:
                        with st.container():
                            st.subheader("High vs Low Leverage Success")
                            lev = weekly[["WEEK", "OL_HIGH_LEVERAGE_SUCCESS", "OL_LOW_LEVERAGE_SUCCESS"]].set_index("WEEK")
                            lev.columns = ["High Leverage", "Low Leverage"]
                            st.line_chart(lev)

                    with st.container():
                        st.subheader("Weekly Clutch Detail")
                        detail = weekly.copy()
                        detail.columns = ["Week", "Clutch Index", "OL Success", "Clean Pocket",
                                          "High Lev Success", "Low Lev Success",
                                          "High Lev EPA", "Pass Prot Clutch", "Run Block Clutch"]
                        st.dataframe(detail, use_container_width=True)
                else:
                    st.info("No clutch data available for this team/season.")

        with tab3:
            st.subheader("The Clutch Index Story: 2018-2025")
            st.markdown("How has NFL offensive line clutch performance evolved over 8 seasons?")

            trend = get_clutch_league_trend()

            if not trend.empty:
                st.info(
                    "**Key finding:** The NFL saw a major shift in OL clutch performance starting in 2023. "
                    "League-wide clutch index jumped from near-zero (0.01-0.04) to 0.08-0.10, meaning OLs "
                    "collectively became significantly better in high-leverage situations. Clean pocket rates "
                    "also leaped from ~65% to ~71%."
                )

                col1, col2 = st.columns(2)
                with col1:
                    with st.container():
                        st.subheader("League Clutch Index by Season")
                        ci_trend = trend[["SEASON", "AVG_CLUTCH_INDEX"]].set_index("SEASON")
                        ci_trend.columns = ["Avg Clutch Index"]
                        st.line_chart(ci_trend)
                with col2:
                    with st.container():
                        st.subheader("Clean Pocket Rate & OL Success Rate")
                        rates = trend[["SEASON", "AVG_CLEAN_POCKET_RATE", "AVG_OL_SUCCESS_RATE"]].set_index("SEASON")
                        rates.columns = ["Clean Pocket Rate", "OL Success Rate"]
                        st.line_chart(rates)

                col1, col2 = st.columns(2)
                with col1:
                    with st.container():
                        st.subheader("Clutch Index Volatility (Std Dev)")
                        st.caption("Higher = more variation between teams that season")
                        vol = trend[["SEASON", "STD_CLUTCH_INDEX"]].set_index("SEASON")
                        vol.columns = ["Std Dev"]
                        st.bar_chart(vol)
                with col2:
                    with st.container():
                        st.subheader("Pass Protection vs Run Blocking Clutch")
                        st.caption("Positive = worse in high leverage (more pressure/stuffs)")
                        split = trend[["SEASON", "AVG_PROTECTION_CLUTCH", "AVG_RUN_CLUTCH"]].set_index("SEASON")
                        split.columns = ["Pass Prot Clutch", "Run Block Clutch"]
                        st.line_chart(split)

                with st.container():
                    st.subheader("Season-by-Season Data")
                    trend_display = trend.copy()
                    trend_display.columns = [
                        "Season", "Avg Clutch Index", "Std Dev", "Clean Pocket Rate",
                        "OL Success Rate", "Pass Prot Clutch", "Run Block Clutch",
                        "Teams", "Games"
                    ]
                    st.dataframe(trend_display, use_container_width=True)

                st.divider()
                st.subheader("Team Journey Across Seasons")
                all_teams = get_all_teams(selected_season)
                journey_team = st.selectbox("Select a team to trace their clutch story", all_teams["TEAM"].tolist(), key="journey_team")

                if journey_team:
                    history = get_clutch_team_history(journey_team)
                    if not history.empty:
                        latest = history.iloc[-1]
                        earliest = history.iloc[0]
                        delta = round(latest["AVG_CLUTCH_INDEX"] - earliest["AVG_CLUTCH_INDEX"], 4)
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric(f"{journey_team} Latest Clutch", f"{latest['AVG_CLUTCH_INDEX']:+.4f}",
                                      f"{delta:+.4f} since {int(earliest['SEASON'])}")
                        with cols[1]:
                            st.metric("Latest Clean Pocket", f"{latest['AVG_CLEAN_POCKET_RATE']:.1%}")
                        with cols[2]:
                            st.metric("Latest OL Success", f"{latest['AVG_OL_SUCCESS_RATE']:.1%}")

                        col1, col2 = st.columns(2)
                        with col1:
                            with st.container():
                                st.subheader(f"{journey_team} Clutch Index Over Time")
                                h_ci = history[["SEASON", "AVG_CLUTCH_INDEX"]].set_index("SEASON")
                                h_ci.columns = ["Clutch Index"]
                                st.line_chart(h_ci)
                        with col2:
                            with st.container():
                                st.subheader(f"{journey_team} Pocket & Success Rate")
                                h_rates = history[["SEASON", "AVG_CLEAN_POCKET_RATE", "AVG_OL_SUCCESS_RATE"]].set_index("SEASON")
                                h_rates.columns = ["Clean Pocket", "OL Success"]
                                st.line_chart(h_rates)

                        with st.container():
                            st.subheader(f"{journey_team} Full History")
                            h_display = history.copy()
                            h_display.columns = ["Season", "Clutch Index", "Clean Pocket", "OL Success",
                                                  "High Lev Success", "Pass Prot Clutch", "Run Block Clutch"]
                            st.dataframe(h_display, use_container_width=True)
                    else:
                        st.info("No historical data available for this team.")
            else:
                st.warning("No league trend data available.")
    else:
        st.warning("No clutch data available for this season.")

elif page == "Matchup Predictor":
    st.header(f"Matchup Predictor — {selected_season}")

    teams = get_all_teams(selected_season)
    team_list = teams["TEAM"].tolist()

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team A", team_list, index=0)
    with col2:
        team2 = st.selectbox("Team B", team_list, index=min(1, len(team_list) - 1))

    if team1 and team2 and team1 != team2:
        matchup = get_matchup_data(team1, team2, selected_season)

        if not matchup.empty:
            t1_data = matchup[matchup["TEAM"] == team1]
            t2_data = matchup[matchup["TEAM"] == team2]

            t1_rolling = t1_data["ROLLING_5_OL_SCORE"].dropna()
            t2_rolling = t2_data["ROLLING_5_OL_SCORE"].dropna()

            t1_pred = round(t1_rolling.iloc[-1], 1) if not t1_rolling.empty else "N/A"
            t2_pred = round(t2_rolling.iloc[-1], 1) if not t2_rolling.empty else "N/A"

            t1_ml = get_ml_prediction_for_team(team1)
            t2_ml = get_ml_prediction_for_team(team2)

            cols = st.columns(4)
            col_idx = 0
            with cols[col_idx]:
                st.metric(f"{team1} Rolling Avg", f"{t1_pred}")
            col_idx += 1
            if not t1_ml.empty:
                with cols[col_idx]:
                    st.metric(f"{team1} ML Predicted", f"{round(t1_ml.iloc[0]['PREDICTED_OL_SCORE'], 1)}")
                col_idx += 1
            with cols[col_idx]:
                st.metric(f"{team2} Rolling Avg", f"{t2_pred}")
            col_idx += 1
            if not t2_ml.empty and col_idx < 4:
                with cols[col_idx]:
                    st.metric(f"{team2} ML Predicted", f"{round(t2_ml.iloc[0]['PREDICTED_OL_SCORE'], 1)}")

            with st.container():
                st.subheader("Season OL Score Trend")
                pivot = matchup.pivot_table(
                    index="WEEK", columns="TEAM", values="COMPOSITE_OL_SCORE"
                )
                st.line_chart(pivot)

            col1, col2 = st.columns(2)
            with col1:
                with st.container():
                    st.subheader(f"{team1} — Pass vs Run Block")
                    t1_chart = t1_data[["WEEK", "PASS_BLOCK_SCORE", "RUN_BLOCK_SCORE"]].set_index("WEEK")
                    t1_chart.columns = ["Pass Block", "Run Block"]
                    st.line_chart(t1_chart)

            with col2:
                with st.container():
                    st.subheader(f"{team2} — Pass vs Run Block")
                    t2_chart = t2_data[["WEEK", "PASS_BLOCK_SCORE", "RUN_BLOCK_SCORE"]].set_index("WEEK")
                    t2_chart.columns = ["Pass Block", "Run Block"]
                    st.line_chart(t2_chart)
    elif team1 == team2:
        st.warning("Please select two different teams.")

elif page == "ML Predictions":
    st.header("ML-Powered Next-Game Predictions")
    st.caption("XGBoost models trained on rolling game features, registered in Snowflake Model Registry")

    preds = get_ml_predictions()

    if not preds.empty:
        top3 = preds.head(3)
        cols = st.columns(3)
        for idx, (_, row) in enumerate(top3.iterrows()):
            with cols[idx]:
                st.metric(
                    f"#{preds.index.get_loc(_) + 1} {row['TEAM']}",
                    f"{row['PREDICTED_OL_SCORE']}",
                    f"Pass: {row['PREDICTED_PASS_BLOCK']} | Run: {row['PREDICTED_RUN_BLOCK']}",
                )

        with st.container():
            st.subheader("All Teams — Predicted Next-Game OL Score")
            display = preds.copy()
            display.insert(0, "RANK", range(1, len(display) + 1))
            display.columns = ["Rank", "Team", "Last Season", "Last Week", "Pred OL Score", "Pred Pass Block", "Pred Run Block"]
            st.dataframe(
                display,
                use_container_width=True,
            )

        col1, col2 = st.columns(2)
        with col1:
            with st.container():
                st.subheader("Predicted OL Score Distribution")
                st.bar_chart(preds.set_index("TEAM")["PREDICTED_OL_SCORE"])
        with col2:
            with st.container():
                st.subheader("Pass vs Run Block Predictions")
                scatter_data = preds[["TEAM", "PREDICTED_PASS_BLOCK", "PREDICTED_RUN_BLOCK"]].set_index("TEAM")
                scatter_data.columns = ["Pass Block", "Run Block"]
                st.bar_chart(scatter_data)

        with st.container():
            st.subheader("Model Performance — v1 / v2 / v3")
            st.markdown("""
| Model | v1 MAE | v2 MAE | v3 MAE | v1 R2 | v2 R2 | v3 R2 |
|-------|--------|--------|--------|-------|-------|-------|
| **Composite OL** | 1.62 | 1.47 | **1.84** | 0.9815 | 0.9833 | **0.9801** |
| **Pass Block** | 1.68 | 1.01 | **1.20** | 0.9892 | 0.9953 | **0.9952** |
| **Run Block** | 1.28 | 0.91 | **0.93** | 0.9946 | 0.9969 | **0.9964** |
""")
            st.caption("v3: 49 features — added OL responsibility classification, leverage-weighted clutch index, protection/run clutch")
            st.caption("Top v3 feature: OL_SUCCESS_RATE (#1 at 0.294 importance) — the clutch framework is highly predictive")
    else:
        st.warning("No ML predictions available. Run the training pipeline first.")

elif page == "Game Summary":
    st.header(f"AI Game Summary — {selected_season}")

    teams = get_all_teams(selected_season)
    selected_team = st.selectbox("Select Team", teams["TEAM"].tolist(), key="summary_team")

    if selected_team:
        games = get_team_game_scores(selected_team, selected_season)
        if not games.empty:
            week_list = sorted(games["WEEK"].unique().tolist())
            selected_week = st.selectbox("Select Week", week_list, index=len(week_list) - 1)

            if st.button("Generate AI Summary", type="primary"):
                with st.spinner("Generating analysis with Cortex AI..."):
                    summary = get_ai_summary(selected_team, selected_season, selected_week)
                    if not summary.empty:
                        row = summary.iloc[0]
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("OL Score", f"{row['COMPOSITE_OL_SCORE']}")
                        with cols[1]:
                            st.metric("Pass Block", f"{row['PASS_BLOCK_SCORE']}")
                        with cols[2]:
                            st.metric("Run Block", f"{row['RUN_BLOCK_SCORE']}")

                        with st.container():
                            raw = row["AI_SUMMARY"]
                            cleaned = raw.strip('"').strip()
                            st.markdown(f"**AI Analysis:**\n\n{cleaned}")
                    else:
                        st.warning("No data available for this game.")

elif page == "Ask the OL Analyst":
    st.header("Ask the OL Analyst")
    st.caption("Ask any question about a team's offensive line performance — powered by Snowflake Cortex AI")

    teams = get_all_teams(selected_season)
    col1, col2 = st.columns(2)
    with col1:
        ask_team = st.selectbox("Select Team", teams["TEAM"].tolist(), key="ask_team")
    with col2:
        ask_season = st.selectbox("Season", season_list, index=0, key="ask_season")

    st.markdown("**Example questions:**")
    st.caption(
        "Is this team's OL a liability in clutch situations? | "
        "How does their pass protection compare to run blocking? | "
        "What's their biggest weakness? | "
        "Should I trust this OL in a playoff game?"
    )

    question = st.text_input("Your question:", placeholder="e.g., How clutch is this offensive line when the game is on the line?")

    if st.button("Ask Cortex AI", type="primary") and question:
        with st.spinner("Analyzing with Cortex AI..."):
            result = get_ai_ol_answer(question, ask_team, ask_season)
            if not result.empty:
                answer = result.iloc[0]["AI_ANSWER"]
                cleaned = answer.strip('"').strip()
                with st.container():
                    st.markdown(f"**AI Analysis for {ask_team} ({ask_season}):**\n\n{cleaned}")
            else:
                st.warning("No data available for this team/season combination.")

elif page == "Methodology":
    st.header("Methodology & Metric Glossary")
    st.caption("How every metric is built, what it means, and why it matters")

    st.markdown("""
## What Is This App?

This is an AI-powered NFL offensive line scoring system that combines play-by-play data, advanced feature
engineering, machine learning models, and Snowflake Cortex AI to evaluate and predict OL performance across
all 32 NFL teams from 2018-2025.

---

## Data Sources

| Source | What It Provides |
|--------|-----------------|
| **nflverse play-by-play** (via nflreadpy) | Every play from every game: sacks, pressures, EPA, yards, down/distance, win probability, score differential |
| **Next Gen Stats** | Time to throw, pass rushers faced, defenders in box |
| **Pro Football Reference** | Advanced pass/rush metrics |
| **Depth Charts** | Player-level OL starters by week |

---

## How the OL Score Works

The **Composite OL Score** (0-100) is a percentile-weighted blend of pass blocking, run blocking, and penalty metrics:

**Pass Block Score** (5 components):
- Sack Rate (30%) — lower = better
- QB Hit Rate (20%) — lower = better
- Pressure Rate (25%) — lower = better
- Pass EPA/play (15%) — higher = better
- Pass Success Rate (10%) — higher = better

**Run Block Score** (6 components):
- Stuff Rate (25%) — lower = better
- Avg Rush Yards (15%) — higher = better
- Rush EPA/play (25%) — higher = better
- Rush Success Rate (15%) — higher = better
- Explosive Run Rate (10%) — higher = better
- OL Penalties (10%) — lower = better

**Composite** blends pass (60% weight) and run (35%) with penalties (5%).

Each metric is converted to a league-wide percentile rank, then weighted and summed to 0-100.

---

## OL Responsibility Framework

Not every play outcome is the offensive line's fault. We classify each play:

**Pass Plays:**
- **OL Failed** = sack, pressure, or QB hit occurred (the OL didn't protect)
- **OL Succeeded** = clean pocket (no sack, no pressure, no hit)

**Run Plays:**
- **OL Failed** = stuffed at or behind the line of scrimmage (0 or fewer yards)
- **OL Succeeded** = created a hole (4+ yards gained)
- **Neutral** = 1-3 yards (ambiguous — could be OL, RB, or scheme)

---

## Leverage Tiers

We define how much pressure a play carries using game context:

| Tier | Definition |
|------|-----------|
| **High Leverage** | 3rd/4th down AND close game (within 8 pts) AND 2nd half |
| **Medium Leverage** | 3rd/4th down OR (close game AND 2nd half) |
| **Low Leverage** | 1st/2nd down in a blowout or early game |

---

## The Clutch Index

```
OL Clutch Index = (OL success rate in HIGH leverage) - (OL success rate in LOW leverage)
```

| Value | Interpretation |
|-------|---------------|
| **Positive (e.g., +0.10)** | This OL gets BETTER when it matters most |
| **Zero** | Consistent regardless of situation |
| **Negative (e.g., -0.05)** | This OL chokes under pressure |

**Related clutch metrics:**
- **OL Protection Clutch** = pressure rate gap (high vs low leverage) — positive means MORE pressure in clutch (bad)
- **OL Run Blocking Clutch** = stuff rate gap (high vs low leverage) — positive means MORE stuffs in clutch (bad)
- **OL High Leverage EPA** = expected points added on high-leverage plays

---

## Complete Metric Glossary

| Metric | Formula | What It Tells You |
|--------|---------|-------------------|
| Composite OL Score | Weighted percentile blend (0-100) | Overall OL quality for a game |
| Pass Block Score | Sack/hit/pressure percentiles weighted | Pass protection quality |
| Run Block Score | Stuff/yards/EPA percentiles weighted | Run blocking quality |
| Sack Rate | Sacks / Dropbacks | How often the QB gets taken down |
| Pressure Rate | Pressures / Dropbacks | How often the QB is hurried |
| QB Hit Rate | QB Hits / Dropbacks | How often the QB gets hit |
| Clean Pocket Rate | Plays with no sack/pressure/hit / Total pass plays | How often OL gives QB a clean pocket |
| Stuff Rate | Runs with 0 or fewer yards / Total runs | How often runs go nowhere |
| Explosive Run Rate | Runs of 10+ yards / Total runs | How often the OL creates big holes |
| EPA (Expected Points Added) | Statistical model of play value | Points above/below expectation per play |
| Success Rate | Plays meeting down-specific yardage thresholds | % of plays that "worked" |
| OL Responsibility Rate | (OL Failed + OL Succeeded) / Total plays | % of plays clearly attributable to OL |
| OL Success Rate | OL Succeeded / (OL Failed + OL Succeeded) | When the OL is responsible, how often do they succeed? |
| OL Clutch Index | High-leverage success - Low-leverage success | Does this OL rise or fall in big moments? |
| OL Protection Clutch | High-lev pressure rate - Low-lev pressure rate | Pass protection under pressure vs comfort |
| OL Run Blocking Clutch | High-lev stuff rate - Low-lev stuff rate | Run blocking under pressure vs comfort |
| Early Down EPA | Avg EPA on 1st/2nd down plays | OL performance in favorable counts |
| Late Down EPA | Avg EPA on 3rd/4th down plays | OL performance under down pressure |
| Redzone EPA | Avg EPA inside the 20-yard line | OL performance in scoring territory |
| 3rd & Short Success | Success rate on 3rd down with 1-2 yards to go | Can the OL convert when it should be easy? |
| 3rd & Long Sack Rate | Sack rate on 3rd and 7+ | Does the OL collapse on obvious passing downs? |

---

## Model Evolution

Three generations of XGBoost models, each adding richer features:

| Version | Features | Key Additions | Composite MAE | Composite R2 |
|---------|----------|---------------|---------------|-------------|
| **v1** | 26 | Base pass/run/penalty + rolling averages | 1.62 | 0.9815 |
| **v2** | 40 | +14 situational (down/distance, redzone, field position) | 1.47 | 0.9833 |
| **v3** | 49 | +9 clutch (OL responsibility, leverage tiers, clutch index) | 1.84 | 0.9801 |

**Why v3 composite MAE increased:** The new clutch features add variance that helps distinguish clutch OLs but
slightly reduces pure score prediction accuracy. However, **OL_SUCCESS_RATE became the #1 most important
feature** (0.294 importance), proving the framework captures real signal. Pass block (1.20 MAE) and run block
(0.93 MAE) models remain excellent.

**Top v3 Features by Importance:**
1. OL_SUCCESS_RATE (0.294) — new clutch feature
2. QB_HIT_RATE (0.144)
3. SACK_RATE (0.127)
4. RUSH_EPA_PER_PLAY (0.101)
5. AVG_RUSH_YARDS (0.070)
6. EARLY_DOWN_EPA (0.052)
7. PASS_EPA_PER_PLAY (0.042)
8. OL_CLEAN_POCKET_RATE (0.035) — new clutch feature

---

## Architecture

```
nflreadpy: pbp + participation / NGS / PFR
        |
        v
  [ingest_nfl_data.py]
        |
        v
  RAW_PBP (389K plays) + RAW_NGS + RAW_PFR + RAW_DEPTH_CHARTS
        |
        v
  [feature_engineering.sql]
        |
        v
  TEAM_PASS_BLOCKING_FEATURES ──┐
  TEAM_RUN_BLOCKING_FEATURES ───┤
  TEAM_PENALTY_FEATURES ────────┤──> TEAM_OL_FEATURES ──> TEAM_OL_SCORES
  TEAM_SITUATIONAL_FEATURES ────┤                              |
  TEAM_OL_CLUTCH_FEATURES ─────┘                              v
                                                    TEAM_OL_ROLLING
                                                         |
                                          ┌──────────────┼──────────────┐
                                          v              v              v
                                  ML_TRAINING_DATA  SEASON_RANKINGS  GAME_AI_SUMMARIES
                                          |
                                          v
                                  [train_models.py]
                                          |
                                          v
                              Snowflake Model Registry
                              (v1, v2, v3 XGBoost)
                                          |
                                          v
                              ML_NEXT_GAME_PREDICTIONS
                                          |
                                          v
                              [streamlit_app.py] ← Cortex AI Q&A
```

---

## Files

| File | Purpose |
|------|---------|
| `ingest_nfl_data.py` | Downloads NFL data via nflreadpy and loads into Snowflake |
| `feature_engineering.sql` | Creates all feature tables from raw play-by-play data |
| `train_models.py` | Trains XGBoost models and registers them in Snowflake Model Registry |
| `run_inference.py` | Runs inference on latest features to generate next-game predictions |
| `streamlit_app.py` | 9-page Streamlit dashboard with rankings, deep dives, clutch analysis, AI Q&A |
| `snowflake.yml` | Snowflake deployment manifest for Streamlit in Snowflake |
| `pyproject.toml` | Python project dependencies |
""")
