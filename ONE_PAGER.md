# NFL Offensive Line Scoring & Clutch Index
### A Data-Driven Framework for Evaluating What Offensive Lines Actually Do

---

## The Problem

Traditional NFL analytics grade offensive lines using raw box score stats — sacks allowed, yards per carry, penalties. But these metrics are noisy. A sack might be the QB holding the ball too long. A stuffed run might be a bad playcall. And none of these stats tell you whether an OL performs differently when the game is actually on the line.

## The Solution

We built a three-layer evaluation system that isolates OL-specific performance, scores it against the league, and measures how it changes under pressure.

### Layer 1: The OL Score (0-100)

Every game, each offensive line receives a composite score built from 11 underlying metrics converted to league-wide percentiles:

- **Pass Protection (60%)** — Sack rate, QB hit rate, pressure rate, pass EPA, pass success rate
- **Run Blocking (35%)** — Stuff rate, yards per carry, rush EPA, rush success rate, explosive run rate, penalties
- **Penalties (5%)** — Holding, false starts, and other OL-attributable flags

A score of 50 is league average. The LA Rams led the NFL in 2025 at 66.6; Cleveland was last at 33.8.

### Layer 2: OL Responsibility Classification

Not every play outcome belongs to the offensive line. We classify each play:

- **Pass plays:** Clean pocket = OL succeeded. Sack, pressure, or QB hit = OL failed.
- **Run plays:** 4+ yards = OL succeeded. Stuffed at the line = OL failed. 1-3 yards = ambiguous, excluded.

This gives us **OL Success Rate** — when the outcome is clearly attributable to the line, how often do they win? This metric became the single most important feature in our predictive models, more predictive than sack rate or pressure rate alone.

### Layer 3: The Clutch Index

We define three leverage tiers using game context:

| Tier | Definition |
|------|-----------|
| **High** | 3rd/4th down + close game (within 8 pts) + 2nd half |
| **Medium** | 3rd/4th down OR (close game + 2nd half) |
| **Low** | Early downs in a comfortable lead |

**OL Clutch Index = High-leverage success rate - Low-leverage success rate**

- **Positive:** This OL elevates in big moments (DEN led 2025 at +0.34)
- **Zero:** Consistent regardless of situation
- **Negative:** This OL folds under pressure (NO was -0.13 in 2025)

## What the Data Reveals

The league-wide clutch index rose from near-zero (2018-2019) to +0.10 (2023-2025), meaning NFL offensive lines have collectively become significantly better in high-leverage situations. Clean pocket rates jumped from 63.5% to 71.5% over the same period.

Our XGBoost models (v4, 49 features, R2=0.98) confirm the framework captures real signal: OL Success Rate is the #1 predictor of overall OL quality, and clutch features account for 3 of the top 8 most important model features.

## Technical Stack

- **Data:** 389K plays from nfl_data_py (2018-2025), Next Gen Stats, Pro Football Reference
- **Compute:** Snowflake (feature engineering, model registry, Cortex AI for natural language Q&A)
- **ML:** XGBoost regression, 49 features, 4 model generations
- **App:** 9-page Streamlit dashboard with rankings, deep dives, clutch analysis, matchup predictor, and AI analyst

---

*Built by Marty St. John | Data from nfl_data_py | Powered by Snowfl
