## LinkedIn Post

---

Offensive lines are the most important and least understood position group in football. So I built a system to fix that.

The problem with traditional OL metrics: a sack might be the QB holding the ball. A stuffed run might be a bad playcall. And no stat tells you whether an OL actually shows up when the game is on the line.

I built a three-layer framework to answer this:

1. OL Score (0-100) — A percentile-weighted composite of 11 pass protection, run blocking, and penalty metrics. Every team, every game, graded against the league.

2. OL Responsibility Classification — Not every play is the line's fault. We classify each play as OL-succeeded (clean pocket, 4+ yard run) or OL-failed (sack, pressure, stuffed at the line) and throw out the noise in between.

3. The Clutch Index — High-leverage OL success rate minus low-leverage OL success rate. Do they rise to the occasion on 3rd down in a close game in the 4th quarter, or do they fold?

What the data says (2025 season):
- The LA Rams' OL ranked #1 overall (66.6/100) with a 3.9% sack rate and league-best 13.4% stuff rate
- Denver had the highest clutch index (+0.34) — their OL elevated dramatically in high-leverage moments
- Buffalo #2 (60.6), Pittsburgh #3 (58.4) — elite pass protection driving the scores
- Las Vegas ranked #31 (33.9) with a 10.7% sack rate and 27.7% stuff rate. Cleveland dead last again at #32
- League-wide, OLs have gotten significantly more clutch since 2018, with clean pocket rates jumping from 63.5% to 71%

The most interesting finding: when I trained XGBoost models to predict OL performance, OL Success Rate (from the responsibility framework) became the #1 most important feature — more predictive than sack rate, pressure rate, or any traditional metric. The clutch framework isn't just a story. It's signal.

Technical stack: 389K plays from nfl_data_py, feature engineering and model registry in Snowflake, Cortex AI for natural language Q&A, and a 9-page Streamlit dashboard.

The full codebase is on GitHub: https://github.com/mstjohn79/snowNFL

#NFL #DataScience #MachineLearning #Snowflake #Analytics #Football #SportsTech

---
