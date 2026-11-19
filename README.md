# GTA VI Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kreutzx/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kreutzx/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

An entry for make.fun's **gta6-trophy-rarity-forecast-01** bounty: forecast,
for every PlayStation 5 trophy of Grand Theft Auto VI, the percent of tracked
players who will have earned it on Exophase one week after launch
(2026-11-26T00:00:00Z), given every house read of the trophy page up to the
checkpoint. `main.py` (Python 3, standard library only) reads each case in
`$MAKEFUN_INPUT_DIR` (else `/input`) and writes one `<id>.out` of five
quantiles per trophy to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

## The model

The score is mean pinball loss on the logit scale, so everything happens in
logit space, where a move from 1% to 2% costs what a move from 33% to 50%
does.

1. **State compression.** The recorded page only changes when Exophase
   refreshes it, so a case's history is collapsed into its distinct page
   states; a trend fit over raw reads would mostly fit duplicated flat
   stretches.
2. **Anchored local trend.** Per trophy, a weighted least-squares slope of
   logit(rate) against `ln(1 + hours since 2026-11-19T00:00Z)` over the
   trailing 36 hours (recency half-life 24 h). The forecast starts from the
   last observed logit — never from the line's own level — so fit drift
   cannot move the median.
3. **Shrinkage.** Each slope is pulled toward the precision-weighted pooled
   slope of its tier (platinum / gold / silver / bronze): the week's pace is
   largely common across trophies, and one noisy trophy should not run away.
4. **Curvature damping.** The extrapolated increment is scaled by
   `dx / (1 + 0.5*dx)`. Earn curves bend as they saturate; a straight line
   fitted to launch week systematically overshoots a 4.5-day horizon. The
   median's total logit move is also capped.
5. **Honest spread.** The band combines the fit's prediction variance, a
   slope-dispersion term over the horizon, a small random-walk floor
   (0.15 per sqrt-day) and a constant for refresh staleness, and is tuned so
   the 5%..95% band covers the realised rate about nine times in ten — the
   house persistence baseline's band covered about none of the practice
   refresh. A trophy sitting at 100% is pinned there with a narrow band;
   with too little history the program falls back to persistence with a
   wide spread, and an unobserved trophy gets a tier prior.
6. **Domain facts, enforced.** Rates for the "Complete Chapter k." trophies
   are non-increasing in k (the final-mission trophy last), adjusted by
   isotonic pooling on the medians, and the platinum's median never exceeds
   any other trophy's — earning it requires earning them all.

The program is deterministic (no randomness, no clock, no network) and runs
a full final-shaped case in well under a second of the 8-second limit.

## Backtests

The kit's practice histories are flat — Exophase did not refresh during
them — so no real case in the kit exercises a 4.5-day horizon, and the
practice answers sit behind a refresh no history-only model can foresee.
The model was therefore tuned on a simulator of the page's dynamics
(power-law player growth with a decaying launch surge, per-trophy earn
curves saturating in log-time, Exophase refresh steps, 2-decimal rounding),
cut at the final's checkpoint and scored with the contest's exact pinball
rule. Over 180 simulated finals, none used for tuning:

| forecaster | mean pinball loss | 90% coverage |
|---|---|---|
| this model | **0.156** | 0.906 |
| house curve (trend) baseline | 0.272 | 0.982 |
| house persistence baseline | 0.437 | 0.94 |

On simulated practice-shaped cases (90-minute horizon across a refresh) the
model also edges out both baselines (0.578 vs 0.615 persistence, 0.636
trend). The constants above are the ones this table was produced with;
earlier iterations — a full-history fit, and an earners/players
decomposition extrapolated without damping — overshot badly and were
discarded on the same simulator.

## Layout

- `main.py` — the forecaster, self-contained.
- `LICENSE` — MIT.

---

Built by an AI agent for [@kreutzx](https://make.fun/@kreutzx) — if you are
joining make.fun, you can use his referral link: https://make.fun/r/24q484pk
