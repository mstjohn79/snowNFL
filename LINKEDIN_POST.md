## Short Post Blurb

---

I wanted to understand offensive line efficiency beyond sacks allowed and yards per carry.

Those numbers lie. A sack might be the QB's fault. A stuffed run might be a bad playcall. And none of them tell you if the line actually holds up when the game is on the line.

So I built two metrics:

𝗢𝗟 𝗦𝗰𝗼𝗿𝗲 (0-100) — 11 features across pass protection, run blocking, and penalties, percentile-ranked against the league and weighted into one number. One score to size up the entire unit.

𝗖𝗹𝘂𝘁𝗰𝗵 𝗜𝗻𝗱𝗲𝘅 — How much better (or worse) the OL performs under pressure: 4th quarter, 3rd down, one-score game. Positive = rises to the moment. Negative = folds.

The 2025-26 results:

🏆 LA Rams #1 overall (66.6/100) — 3.9% sack rate, league-best 13.4% stuff rate
📈 Denver most clutch OL in the league (+0.34) — they rose to the moment all season
📉 Cleveland dead last. Again. (33.8/100)
💪 Buffalo #2, Pittsburgh #3 — elite pass protection top to bottom
🧊 Chicago quietly top-5 (57.9) with the lowest sack rate in the league at 3.7%

The whole thing — data pipeline, feature engineering, ML training, interactive dashboard — runs on Snowflake. Cortex Code was the copilot for the entire build. Streamlit in Snowflake hosts the live app.

This is v1: team-level grades from public play-by-play data. It tells you WHAT happened. The next step is WHO and WHY — player tracking, pre-snap reads, scheme classification, opponent-adjusted scoring. The framework scales. The data just needs to get richer.

Code: https://github.com/mstjohn79/snowNFL

#NFL #Snowflake #CortexCode #DataScience #MachineLearning #StreamlitInSnowflake #SportsTech #Analytics

---

---

## Full LinkedIn Post

---

Offensive lines are the most important and least understood position group in football.

So I built a system to grade them. From scratch. In a weekend.

Here's the thing — traditional OL metrics are broken.

A sack? Might be the QB holding the ball too long.
A stuffed run? Could be a bad playcall.
And nothing tells you if an OL actually shows up when the game is on the line.

So I built a three-layer framework using publicly available play-by-play data:

𝗟𝗮𝘆𝗲𝗿 𝟭: 𝗢𝗟 𝗦𝗰𝗼𝗿𝗲 (𝟬-𝟭𝟬𝟬)
11 metrics across pass protection, run blocking, and penalties — all converted to league-wide percentiles and weighted into a single composite score.

𝗟𝗮𝘆𝗲𝗿 𝟮: 𝗢𝗟 𝗥𝗲𝘀𝗽𝗼𝗻𝘀𝗶𝗯𝗶𝗹𝗶𝘁𝘆
Not every play is the line's fault. Clean pocket = OL won. Sack or pressure = OL lost. 1-3 yard run? Ambiguous — throw it out.

𝗟𝗮𝘆𝗲𝗿 𝟯: 𝗧𝗵𝗲 𝗖𝗹𝘂𝘁𝗰𝗵 𝗜𝗻𝗱𝗲𝘅
High-leverage success rate minus low-leverage success rate. Does this OL rise in the 4th quarter on 3rd down in a close game — or fold?

What the 2025-26 data says:

- LA Rams ranked #1 (66.6/100) — 3.9% sack rate, league-best 13.4% stuff rate
- Denver was the most clutch OL in the league (+0.34)
- Buffalo #2, Pittsburgh #3 — elite pass protection
- Cleveland dead last. Again. (33.8/100)
- Chicago quietly top-5 (57.9) with the league's lowest sack rate at 3.7%

The most interesting finding?

When I trained XGBoost models (49 features, R2=0.98) to predict OL quality, the #1 most important feature wasn't sack rate or pressure rate.

It was OL Success Rate — the metric from the responsibility framework.

The clutch framework isn't just a story. It's signal.

---

𝗛𝗼𝘄 𝗜 𝗯𝘂𝗶𝗹𝘁 𝗶𝘁 (𝘁𝗵𝗲 𝗦𝗻𝗼𝘄𝗳𝗹𝗮𝗸𝗲 𝘀𝘁𝗮𝗰𝗸):

This entire project — data pipeline, feature engineering, ML training, inference, and a live dashboard — runs on Snowflake:

- Cortex Code helped me build, debug, and iterate on the entire codebase — SQL, Python, Streamlit — in a single session. It caught a critical scoring bug (inverted percentile logic), rebuilt the pipeline, retrained models, and deployed the fix.
- Snowflake Model Registry stores 4 generations of XGBoost models with full versioning and lineage
- Cortex AI powers a natural language Q&A page — ask "How did the Chiefs' OL perform in the clutch?" and get a data-grounded answer
- Streamlit in Snowflake hosts the 9-page interactive dashboard — no infra to manage, governed by default
- All feature engineering runs as SQL inside Snowflake — PERCENT_RANK(), window functions, CTEs — zero external compute

---

𝗪𝗵𝗮𝘁 𝘁𝗵𝗶𝘀 𝗶𝘀 — 𝗮𝗻𝗱 𝘄𝗵𝗮𝘁 𝗶𝘁 𝗶𝘀𝗻'𝘁:

This is a team-level view built on publicly available play-by-play data. It tells you WHAT happened on each play and grades the unit as a whole. That's genuinely useful — but it's a starting point.

What it can't tell you yet is WHO on the line was responsible, or WHY the outcome happened. Was it a scheme issue? A blown assignment by the left guard? A pre-snap read that the center missed?

That's where additional data sources open the door:

- 𝗡𝗲𝘅𝘁 𝗚𝗲𝗻 𝗦𝘁𝗮𝘁𝘀 𝗽𝗹𝗮𝘆𝗲𝗿 𝘁𝗿𝗮𝗰𝗸𝗶𝗻𝗴 — individual lineman movement, gap assignments, double-team rates. This moves us from team grades to player grades.
- 𝗣𝗿𝗲-𝘀𝗻𝗮𝗽 𝗮𝗹𝗶𝗴𝗻𝗺𝗲𝗻𝘁 𝗱𝗮𝘁𝗮 — defensive front alignment, blitz tells, box counts. Were they schemed into a bad look, or did they fail against a vanilla front?
- 𝗣𝗙𝗙-𝘀𝘁𝘆𝗹𝗲 𝗴𝗿𝗮𝗱𝗲𝘀 𝗼𝗿 𝗰𝗵𝗮𝗿𝘁𝗶𝗻𝗴 𝗱𝗮𝘁𝗮 — play-level assignment grades per lineman. This is the gold standard but proprietary.
- 𝗢𝗽𝗽𝗼𝗻𝗲𝗻𝘁-𝗮𝗱𝗷𝘂𝘀𝘁𝗲𝗱 𝘀𝗰𝗼𝗿𝗶𝗻𝗴 — facing Aaron Donald vs. a backup DE should matter. Weighting performance by opponent pass rush quality changes the picture.
- 𝗦𝗰𝗵𝗲𝗺𝗲 𝗰𝗹𝗮𝘀𝘀𝗶𝗳𝗶𝗰𝗮𝘁𝗶𝗼𝗻 — zone vs. gap run schemes, play-action vs. dropback. Did the OL succeed because of scheme design or despite it?

The v1 answers: "How good is this offensive line, and do they show up when it matters?"

The v2 question is: "Why — and who's responsible?"

That's where this is headed. The framework is built. The pipeline scales. The data just needs to get richer.

Code is public: https://github.com/mstjohn79/snowNFL

#NFL #Snowflake #CortexAI #CortexCode #DataScience #MachineLearning #StreamlitInSnowflake #SportsTech #Football #Analytics

---
