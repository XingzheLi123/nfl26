# Chance Created and Actionable Chance

Passing-opportunity metrics built from NFL player tracking and PFF scouting data.

## Definitions

A play has a **Chance Created** when at least one route runner:

1. reaches or passes the first-down line;
2. has at least 3.5 yards of separation from the nearest coverage defender;
3. retains at least 2 projected yards of separation 0.5 seconds later;
4. satisfies these conditions during the final three tracking frames, about 0.3 seconds.

The decision frame is pass release, sack, or 2.5 seconds after the snap when
neither event is available.

An **Actionable Chance** is a Chance Created on a play without a PFF-recorded
hurry, hit, or sack.

```text
Chance Created     = qualifying receiver window
Actionable Chance  = Chance Created and no hurry/hit/sack
Actionability Rate = Actionable Chances / Chances Created
```

Actionability measures how often protection and play context preserve a created
chance. It is not a pure offensive-line grade: QB timing, blitz pickup, running
backs, tight ends, and play design also affect pressure.

## Duplicate chances

Team receives at most one Chance Created per play because only one pass can be
completed. If `k` receivers qualify, each receives `1/k` fractional credit.

```text
Receiver Chance Created per 100 routes = 100 * fractional credits / routes
Team Chance Created rate               = 100 * chance plays / dropbacks
Actionable Chance rate                 = 100 * actionable plays / dropbacks
```

## Build

Run from repository root:

```bash
python3 chance_stat/build_chance_stat.py
python3 chance_stat/build_dashboard.py
```

Processing occurs one game at a time, avoiding a full in-memory load of the
tracking dataset.

Optional threshold example:

```bash
python3 chance_stat/build_chance_stat.py \
  --separation 3.0 \
  --projected-separation 1.5 \
  --required-frames 2
```

## Current validation

All results use the current Chance Created/Actionable Chance regime.

| Play state | Plays | First-down rate | Yards/dropback |
|---|---:|---:|---:|
| No Chance Created | 6,051 | 33.0% | 6.40 |
| Chance Created | 2,482 | 41.4% | 6.97 |
| No Actionable Chance | 7,189 | 32.2% | 5.98 |
| Actionable Chance | 1,344 | 52.5% | 9.68 |

Overall Actionability Rate is **54.1%**: 1,344 of 2,482 created chances were
actionable.

QB-game validation, excluding games with fewer than 10 QB dropbacks:

| Metric | First-down correlation | Yards/dropback correlation |
|---|---:|---:|
| Chance Created rate | 0.015 | -0.012 |
| Actionable Chance rate | 0.152 | 0.139 |

Defensive evaluation across 32 teams:

| Metric allowed | First-down correlation | TD correlation | Yards/dropback correlation |
|---|---:|---:|---:|
| Chance Created rate | 0.164 | 0.032 | -0.036 |
| Actionable Chance rate | 0.267 | 0.140 | 0.081 |

Only eight weeks are available, so team-game and QB-game correlations remain
noisy. These are relevance checks, not causal estimates.

## Notebook experiments

Open `chance_stat/chance_stat_analysis.ipynb`. It contains:

- dataset summary;
- Chance Created and Actionable Chance play-level validation;
- combined no-chance, pressure-blocked, and actionable states;
- receiver leaderboard;
- offensive-line actionability proxy by team;
- QB actionability stability;
- QB summary and metric correlations;
- defensive table and metric correlations;
- four defensive scatter plots, repeating first-down and TD experiments for
  both Chance Created and Actionable Chance;
- embedded interactive QB dashboard.

## Interactive QB dashboard

Open `chance_stat/output/qb_dashboard.html` directly in a browser.

Controls:

- select QB;
- toggle X-axis between Chance Created and Actionable Chance.

Plots:

- selected opportunity rate versus first-down rate;
- selected opportunity rate versus yards per dropback.

Each point represents one QB-game.

## Outputs

- `output/route_chances.csv`: one row per route runner per play
- `output/receiver_scores.csv`: receiver leaderboard
- `output/team_game_summary.csv`: team-game chances and outcomes
- `output/team_correlations.csv`: within-team game correlations
- `output/qb_game_summary.csv`: QB-game chances and outcomes
- `output/defense_summary.csv`: defensive chances and outcomes allowed
- `output/qb_dashboard.html`: standalone interactive visualization

