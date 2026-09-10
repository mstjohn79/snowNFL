# NFL Offensive Line Scoring System

An AI-powered analytics platform that scores, predicts, and analyzes NFL offensive line performance using play-by-play data, machine learning, and Snowflake Cortex AI.

## What Makes This Unique

- **OL Responsibility Classification** — Attributes each play's outcome to the offensive line (not the QB or RB), separating signal from noise
- **Leverage-Weighted Clutch Index** — Measures whether an OL rises or falls in high-pressure situations (3rd down, close games, 2nd half)
- **49-Feature XGBoost Models** — Three generations of ML models registered in Snowflake's Model Registry, evolving from base metrics to situational to clutch-aware
- **Cortex AI Q&A** — Ask natural language questions about any team's OL, grounded in real data via Snowflake Cortex COMPLETE
- **Full Pipeline in Snowflake** — Data ingestion, feature engineering, ML training, inference, and a 9-page Streamlit dashboard

## Architecture

```
nflreadpy (nflverse) / Next Gen Stats / Pro Football Reference
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
                              (v1-v4 XGBoost)
                                          |
                                          v
                              ML_NEXT_GAME_PREDICTIONS
                                          |
                                          v
                              [streamlit_app.py] ← Cortex AI Q&A
```

## Data Sources

| Source | Records | What It Provides |
|--------|---------|-----------------|
| nflverse play-by-play | 389,358 plays | Every play from 2018-2025: sacks, EPA, yards, down/distance, win probability |
| nflverse pbp participation | joined onto pbp | Pressures, time to throw, defenders in box, pass rushers faced |
| Next Gen Stats (passing) | 4,785 rows | Time to throw, pass rushers faced |
| Next Gen Stats (rushing) | 4,885 rows | Defenders in box, rush metrics |
| Pro Football Reference | 23,885 rows | Advanced pass/rush efficiency metrics |
| Depth Charts | 813,157 rows | Player-level OL starters by week |
| Rosters | 24,852 rows | Player positions and metadata |

## The OL Score (0-100)

A percentile-weighted composite of pass blocking, run blocking, and penalty metrics.

### Pass Block Score (5 components)
| Metric | Weight | Direction |
|--------|--------|-----------|
| Sack Rate | 30% | Lower is better |
| QB Hit Rate | 20% | Lower is better |
| Pressure Rate | 25% | Lower is better |
| Pass EPA/play | 15% | Higher is better |
| Pass Success Rate | 10% | Higher is better |

### Run Block Score (6 components)
| Metric | Weight | Direction |
|--------|--------|-----------|
| Stuff Rate | 25% | Lower is better |
| Rush EPA/play | 25% | Higher is better |
| Avg Rush Yards | 15% | Higher is better |
| Rush Success Rate | 15% | Higher is better |
| Explosive Run Rate | 10% | Higher is better |
| OL Penalties | 10% | Lower is better |

## OL Responsibility Framework

The core insight: **not every play outcome is the OL's fault**. We classify each play:

### Pass Plays
- **OL Failed** = sack, pressure, or QB hit occurred
- **OL Succeeded** = clean pocket (no sack, no pressure, no hit)

### Run Plays
- **OL Failed** = stuffed at or behind the line (0 or fewer yards)
- **OL Succeeded** = created a hole (4+ yards)
- **Neutral** = 1-3 yards (ambiguous)

## Leverage Tiers

| Tier | Definition |
|------|-----------|
| **High** | 3rd/4th down + close game (within 8 pts) + 2nd half |
| **Medium** | 3rd/4th down OR (close game + 2nd half) |
| **Low** | 1st/2nd down in a blowout or early game |

## The Clutch Index

```
OL Clutch Index = (OL success rate in HIGH leverage) - (OL success rate in LOW leverage)
```

- **Positive** = This OL gets BETTER when it matters (elite under pressure)
- **Zero** = Consistent regardless of situation
- **Negative** = This OL chokes under pressure (liability in big moments)

### Related Clutch Metrics
- **OL Protection Clutch** — Pressure rate gap (high vs low leverage). Positive = more pressure in clutch (bad).
- **OL Run Blocking Clutch** — Stuff rate gap (high vs low leverage). Positive = more stuffs in clutch (bad).
- **OL High Leverage EPA** — Expected points added specifically on high-leverage plays.
- **OL Clean Pocket Rate** — % of pass plays where the OL gives the QB a clean pocket.

## Complete Metric Glossary

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| Composite OL Score | Weighted percentile blend (0-100) | Overall OL quality per game |
| Pass Block Score | Sack/hit/pressure/EPA percentiles | Pass protection quality |
| Run Block Score | Stuff/yards/EPA percentiles | Run blocking quality |
| Sack Rate | Sacks / Dropbacks | How often QB gets taken down |
| Pressure Rate | Pressures / Dropbacks | How often QB is hurried |
| QB Hit Rate | QB Hits / Dropbacks | How often QB gets hit |
| Clean Pocket Rate | No sack/pressure/hit plays / Pass plays | How often OL gives clean pocket |
| Stuff Rate | Runs ≤0 yards / Total runs | How often runs go nowhere |
| Explosive Run Rate | Runs ≥10 yards / Total runs | How often OL creates big holes |
| EPA/play | Statistical play value model | Points above/below expectation |
| OL Responsibility Rate | OL-attributed plays / Total plays | % clearly the OL's fault/credit |
| OL Success Rate | OL Succeeded / OL-attributed plays | When it's on the OL, how often they win |
| OL Clutch Index | High-lev success - Low-lev success | Rise or fall in big moments |
| OL Protection Clutch | High-lev pressure% - Low-lev pressure% | Pass pro under pressure vs comfort |
| OL Run Blocking Clutch | High-lev stuff% - Low-lev stuff% | Run blocking under pressure vs comfort |
| Early Down EPA | Avg EPA on 1st/2nd down | Performance in favorable counts |
| Late Down EPA | Avg EPA on 3rd/4th down | Performance under down pressure |
| Redzone EPA | Avg EPA inside the 20 | Scoring territory performance |
| 3rd & Short Success | Success rate on 3rd & 1-2 yds | Converting when it should be easy |
| 3rd & Long Sack Rate | Sack rate on 3rd & 7+ | Collapsing on obvious passing downs |
| Situational EPA Variance | Variance of EPA across situations | How consistent is this OL? |

## Model Evolution

| Version | Features | Key Additions | Composite MAE | Pass MAE | Run MAE | Composite R2 |
|---------|----------|---------------|---------------|----------|---------|-------------|
| **v1** | 26 | Base pass/run/penalty + rolling averages | 1.62 | 1.68 | 1.28 | 0.9815 |
| **v2** | 40 | +14 situational (down/distance, redzone, field position) | 1.47 | 1.01 | 0.91 | 0.9833 |
| **v3** | 49 | +9 clutch (OL responsibility, leverage, clutch index) | 1.84 | 1.20 | 0.93 | 0.9801 |
| **v4** | 49 | Corrected percentile scoring (lower-is-better metrics fixed) | 1.83 | 1.21 | 0.97 | 0.9797 |

### Why v4

v4 uses the same 49 features as v3 but fixes a critical bug in the percentile scoring: "lower is better" metrics (sack rate, pressure rate, QB hit rate, stuff rate, penalties) were inverted, causing the worst OLs to score highest. After correction, **OL_SUCCESS_RATE remains the #1 most important feature** (0.267 importance score), confirming the clutch framework captures genuine signal.

### Top v4 Feature Importances
1. **OL_SUCCESS_RATE** (0.267) — *clutch feature*
2. QB_HIT_RATE (0.159)
3. SACK_RATE (0.103)
4. ROLLING_3_OL_SCORE (0.089)
5. RUSH_EPA_PER_PLAY (0.079)
6. EARLY_DOWN_EPA (0.057)
7. PASS_EPA_PER_PLAY (0.042)
8. **OL_CLEAN_POCKET_RATE** (0.035) — *clutch feature*

## Key Findings

- **League-wide clutch trend**: OL clutch index rose from ~0.01 (2018-2019) to ~0.10 (2023-2025). NFL OLs collectively became significantly better in high-leverage situations.
- **Clean pocket rates jumped**: From 63.5% (2018) to 71.5% (2023), suggesting offensive line play has materially improved.
- **2025 #1 OL**: LA Rams (66.6 composite score) — elite pass protection with a 3.9% sack rate.
- **2025 most clutch OL**: DEN (+0.34 clutch index) — elevates dramatically in big moments.
- **KC pass protection clutch**: +0.28 — Patrick Mahomes' OL tightens up when the game is on the line, matching the real-world narrative.
- **2025 least clutch**: NO (-0.13) — tends to crumble under pressure. CLE ranked dead last overall (33.8).

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Team Rankings** | Season-wide OL scores with progress bars, top 3 highlighted |
| **Team Deep Dive** | Weekly OL scores, game log, key metrics, depth chart starters |
| **Situational Analysis** | Redzone, down/distance, field position performance |
| **OL Clutch Index** | Clutch rankings, leverage analysis, multi-season story |
| **Matchup Predictor** | Head-to-head OL comparison with ML predictions |
| **ML Predictions** | Next-game predictions from XGBoost v4, model performance |
| **Game Summary** | Per-game AI-generated analysis via Cortex COMPLETE |
| **Ask the OL Analyst** | Free-text Q&A about any team, powered by Cortex AI |
| **Methodology** | Full metric glossary, scoring methodology, model evolution |

## Files

| File | Purpose |
|------|---------|
| `ingest_nfl_data.py` | Downloads NFL data via nflreadpy and loads into Snowflake raw tables (`--dry-run` writes CSVs instead) |
| `feature_engineering.sql` | SQL to create all feature tables from raw play-by-play data |
| `train_models.py` | Trains 3 XGBoost models (composite, pass, run) and registers in Snowflake Model Registry |
| `run_inference.py` | Loads v4 models from registry and generates next-game predictions |
| `streamlit_app.py` | 9-page Streamlit dashboard (1,049 lines) |
| `snowflake.yml` | Snowflake deployment manifest for Streamlit in Snowflake |
| `pyproject.toml` | Python project dependencies |

## Setup & Running

### Prerequisites
- Snowflake account with Cortex AI enabled
- Python 3.11+ with `nflreadpy`, `polars`, `snowflake-connector-python`, `snowflake-ml-python`, `xgboost`, `scikit-learn`
  (ingestion deps: `pip install -e '.[ingest]'`)
- A named Snowflake connection (default: `martydemo`)

### 1. Ingest Data
```bash
SNOWFLAKE_CONNECTION_NAME=martydemo python ingest_nfl_data.py
```

### 2. Build Feature Tables
Run `feature_engineering.sql` in Snowflake (creates all intermediate tables).

### 3. Train Models
```bash
SNOWFLAKE_CONNECTION_NAME=martydemo python train_models.py
```

### 4. Run Inference
```bash
SNOWFLAKE_CONNECTION_NAME=martydemo python run_inference.py
```

### 5. Launch Dashboard
```bash
streamlit run streamlit_app.py --server.port 8501
```

### Deploy to Snowflake
```bash
snow streamlit deploy --replace
```

## Snowflake Objects

- **Database**: `NFL_ANALYTICS`
- **Schema**: `OL_SCORING`
- **Warehouse**: `COMPUTE_WH` (XS)
- **Models**: `NFL_OL_COMPOSITE_PREDICTOR`, `NFL_PASS_BLOCK_PREDICTOR`, `NFL_RUN_BLOCK_PREDICTOR` (v1-v4)
- **Cortex AI**: `SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b', ...)` for game summaries and interactive Q&A
