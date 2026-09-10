USE DATABASE NFL_ANALYTICS;
USE SCHEMA OL_SCORING;
USE WAREHOUSE COMPUTE_WH;

-- =============================================================================
-- TEAM_OL_CLUTCH_FEATURES -- RECONSTRUCTED DEFINITION.  READ THIS FIRST.
-- =============================================================================
-- The original DDL for TEAM_OL_CLUTCH_FEATURES is not in this repository and
-- never has been. It is read by feature_engineering.sql and by four queries in
-- streamlit_app.py, but nothing creates it: the initial commit already only
-- LEFT JOINed it, and .cortex/skills/nfl-ol-scoring/SKILL.md line 80 records it
-- as "(created separately)". The live table exists only in Snowflake.
--
-- This file is therefore a RECONSTRUCTION from the documented behaviour, not a
-- recovery of the original. It writes to
-- TEAM_OL_CLUTCH_FEATURES_RECONSTRUCTED so it CANNOT overwrite the live table.
-- Run the validation query at the bottom before promoting it.
--
-- Definitions were taken from:
--   streamlit_app.py:478  "high-leverage situations (3rd/4th down, close game,
--                          2nd half)"
--   streamlit_app.py:517  "Negative = lower pressure rate in clutch (good)"
--   streamlit_app.py:523  "Negative = lower stuff rate in clutch (good)"
--   streamlit_app.py:252-255  clutch index "positive=better in big moments",
--                          protection "positive=more pressure in clutch",
--                          run blocking "positive=more stuffs in clutch"
--   SKILL.md:12           OL Responsibility -- "Classifies plays as
--                          OL-attributable (clean pocket, sack, pressure) vs.
--                          ambiguous"
--   SKILL.md:13           Clutch Index -- "High-leverage success rate minus
--                          low-leverage success rate"
--
-- KNOWN AMBIGUITY: the two sources describe high leverage differently.
-- streamlit_app.py lists "3rd/4th down, close game, 2nd half". SKILL.md:13 says
-- "4th quarter, 3rd down, one-score game". Read as a strict AND, high-leverage
-- plays would number only a handful per game and the index would be pure noise,
-- so this implements the broader reading: any 3rd or 4th down, OR any second-half
-- play in a one-score game. If the validation below shows a systematic offset,
-- this is the first knob to turn -- see HIGH_LEVERAGE below.
--
-- DEPENDENCY WARNING: OL_RESPONSIBILITY_RATE, OL_SUCCESS_RATE,
-- OL_CLEAN_POCKET_RATE and OL_PROTECTION_CLUTCH all need play-level pressure,
-- which comes from pbp participation. nflverse publishes that retroactively, so
-- those four columns are NULL for any in-progress season and no live substitute
-- exists (FTN charting carries no pressure field). OL_CLUTCH_INDEX,
-- OL_HIGH/LOW_LEVERAGE_SUCCESS, OL_HIGH_LEVERAGE_EPA and OL_RUN_BLOCKING_CLUTCH
-- need only play-by-play and do work live.
-- =============================================================================
CREATE OR REPLACE TABLE TEAM_OL_CLUTCH_FEATURES_RECONSTRUCTED AS
WITH plays AS (
    SELECT
        GAME_ID,
        SEASON,
        WEEK,
        POSTEAM AS TEAM,
        PLAY_TYPE,
        EPA,
        SUCCESS,
        SACK,
        QB_SCRAMBLE,
        YARDS_GAINED,

        -- NULL when pressure was never charted, so "unknown" is never counted
        -- as "no pressure". Matches the WAS_PRESSURE_FLAG rule in
        -- feature_engineering.sql.
        CASE WHEN WAS_PRESSURE IS NULL
                  OR UPPER(CAST(WAS_PRESSURE AS VARCHAR)) IN ('NAN', 'NONE', '') THEN NULL
             WHEN UPPER(CAST(WAS_PRESSURE AS VARCHAR)) = 'TRUE' THEN 1
             WHEN TRY_CAST(WAS_PRESSURE AS FLOAT) = 1 THEN 1
             ELSE 0 END AS WAS_PRESSURE_FLAG,

        -- Any 3rd/4th down, or any second-half play in a one-score game.
        CASE WHEN DOWN IN (3, 4) THEN 1
             WHEN TRY_CAST(GAME_HALF AS VARCHAR) IN ('Half2', 'Overtime')
                  AND ABS(TRY_CAST(SCORE_DIFFERENTIAL AS FLOAT)) <= 8 THEN 1
             ELSE 0 END AS HIGH_LEVERAGE,

        CASE WHEN YARDS_GAINED <= 0 THEN 1 ELSE 0 END AS STUFFED
    FROM RAW_PBP
    WHERE PLAY_TYPE IN ('pass', 'run')
      AND POSTEAM IS NOT NULL
      AND POSTEAM != 'nan'
),
classified AS (
    SELECT *,
        -- OL-attributable: a run the OL actually blocked for, or a pass whose
        -- pressure state is known (sack / pressured / clean pocket).
        -- Scrambles and uncharted passes are ambiguous.
        CASE WHEN PLAY_TYPE = 'run' AND COALESCE(TRY_CAST(QB_SCRAMBLE AS FLOAT), 0) = 0 THEN 1
             WHEN PLAY_TYPE = 'pass' AND WAS_PRESSURE_FLAG IS NOT NULL THEN 1
             ELSE 0 END AS OL_ATTRIBUTABLE,
        -- Clean pocket: a dropback with neither sack nor charted pressure.
        CASE WHEN PLAY_TYPE = 'pass' AND WAS_PRESSURE_FLAG IS NOT NULL
             THEN CASE WHEN WAS_PRESSURE_FLAG = 0
                        AND COALESCE(TRY_CAST(SACK AS FLOAT), 0) = 0 THEN 1 ELSE 0 END
             ELSE NULL END AS CLEAN_POCKET
    FROM plays
)
SELECT
    GAME_ID,
    SEASON,
    WEEK,
    TEAM,

    ROUND(AVG(OL_ATTRIBUTABLE), 4) AS OL_RESPONSIBILITY_RATE,

    ROUND(AVG(CASE WHEN OL_ATTRIBUTABLE = 1 THEN SUCCESS END), 4) AS OL_SUCCESS_RATE,

    ROUND(AVG(CLEAN_POCKET), 4) AS OL_CLEAN_POCKET_RATE,

    ROUND(AVG(CASE WHEN OL_ATTRIBUTABLE = 1 AND HIGH_LEVERAGE = 1 THEN SUCCESS END), 4)
        AS OL_HIGH_LEVERAGE_SUCCESS,
    ROUND(AVG(CASE WHEN OL_ATTRIBUTABLE = 1 AND HIGH_LEVERAGE = 0 THEN SUCCESS END), 4)
        AS OL_LOW_LEVERAGE_SUCCESS,

    -- Positive = better in big moments.
    ROUND(AVG(CASE WHEN OL_ATTRIBUTABLE = 1 AND HIGH_LEVERAGE = 1 THEN SUCCESS END)
        - AVG(CASE WHEN OL_ATTRIBUTABLE = 1 AND HIGH_LEVERAGE = 0 THEN SUCCESS END), 4)
        AS OL_CLUTCH_INDEX,

    ROUND(AVG(CASE WHEN OL_ATTRIBUTABLE = 1 AND HIGH_LEVERAGE = 1 THEN EPA END), 4)
        AS OL_HIGH_LEVERAGE_EPA,

    -- Positive = more pressure allowed when it matters (bad).
    ROUND(AVG(CASE WHEN PLAY_TYPE = 'pass' AND HIGH_LEVERAGE = 1 THEN WAS_PRESSURE_FLAG END)
        - AVG(CASE WHEN PLAY_TYPE = 'pass' AND HIGH_LEVERAGE = 0 THEN WAS_PRESSURE_FLAG END), 4)
        AS OL_PROTECTION_CLUTCH,

    -- Positive = more stuffs allowed when it matters (bad).
    ROUND(AVG(CASE WHEN PLAY_TYPE = 'run' AND HIGH_LEVERAGE = 1 THEN STUFFED END)
        - AVG(CASE WHEN PLAY_TYPE = 'run' AND HIGH_LEVERAGE = 0 THEN STUFFED END), 4)
        AS OL_RUN_BLOCKING_CLUTCH
FROM classified
GROUP BY GAME_ID, SEASON, WEEK, TEAM;


-- =============================================================================
-- VALIDATION -- run this before promoting the reconstruction.
-- =============================================================================
-- Compares the reconstruction against the live table on the seasons where both
-- have data. CORR near 1.0 and a small MEAN_ABS_DIFF mean the definition is
-- right. A high correlation with a constant offset points at the HIGH_LEVERAGE
-- ambiguity noted above rather than at the metric itself.
--
-- SELECT
--     o.SEASON,
--     COUNT(*)                                                    AS GAMES_MATCHED,
--     ROUND(CORR(r.OL_CLUTCH_INDEX,  o.OL_CLUTCH_INDEX), 4)       AS CORR_CLUTCH,
--     ROUND(AVG(ABS(r.OL_CLUTCH_INDEX - o.OL_CLUTCH_INDEX)), 4)   AS MAD_CLUTCH,
--     ROUND(CORR(r.OL_SUCCESS_RATE,  o.OL_SUCCESS_RATE), 4)       AS CORR_SUCCESS,
--     ROUND(AVG(ABS(r.OL_SUCCESS_RATE - o.OL_SUCCESS_RATE)), 4)   AS MAD_SUCCESS,
--     ROUND(CORR(r.OL_CLEAN_POCKET_RATE, o.OL_CLEAN_POCKET_RATE), 4) AS CORR_POCKET,
--     ROUND(AVG(ABS(r.OL_CLEAN_POCKET_RATE - o.OL_CLEAN_POCKET_RATE)), 4) AS MAD_POCKET,
--     ROUND(CORR(r.OL_RESPONSIBILITY_RATE, o.OL_RESPONSIBILITY_RATE), 4) AS CORR_RESP,
--     ROUND(CORR(r.OL_PROTECTION_CLUTCH, o.OL_PROTECTION_CLUTCH), 4) AS CORR_PROT,
--     ROUND(CORR(r.OL_RUN_BLOCKING_CLUTCH, o.OL_RUN_BLOCKING_CLUTCH), 4) AS CORR_RUN
-- FROM TEAM_OL_CLUTCH_FEATURES_RECONSTRUCTED r
-- JOIN TEAM_OL_CLUTCH_FEATURES o
--     ON r.GAME_ID = o.GAME_ID AND r.TEAM = o.TEAM
-- GROUP BY o.SEASON
-- ORDER BY o.SEASON
--
-- To promote once validated:
--   CREATE OR REPLACE TABLE TEAM_OL_CLUTCH_FEATURES CLONE
--       TEAM_OL_CLUTCH_FEATURES_RECONSTRUCTED
-- Take a backup clone of the live table first.
