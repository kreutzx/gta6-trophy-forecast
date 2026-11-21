"""Week-one GTA VI trophy rarity forecaster, v3 (make.fun:
gta6-trophy-rarity-forecast-01).

v3 rebuilds the model around how the page is actually produced,
following the approach of the strongest public entries here:

1. Earners space. A printed rate is 100 x earners / tracked, and the
   two factors move for different reasons, so nothing is fitted to the
   rate itself. Every read is converted to implied earners, the growth
   curve is fitted on earners, the tracked pool is forecast on its own,
   and the rate is divided back out only at the end.

2. Segmentation at refreshes. The tracked count the page prints jumps
   in steps when Exophase republishes (launch night: 638 -> 3,868 in
   one refresh). The pool's growth rate is estimated only from refresh
   points after the last big step, so no fitted line ever extrapolates
   a step into the future.

3. Distinct page states. Between refreshes the page is a step
   function; runs of identical reads collapse to one observation
   before anything is fitted.

4. Persistence is the anchor. The median starts at the last printed
   rate; the earners-space forecast (slope shrunk toward the tier
   pool, extrapolation damped) only adjusts it, by a weight a
   rolling-origin backtest inside the program earns per trophy.
   Earners never decrease in the model, and dilution -- the pool
   growing into every rate -- is a first-class term, not a residual.

5. The band is built from components, not one calibration: the
   growth fit's residual, a horizon random walk of about 0.5 logit
   per root day, a refresh-risk term that grows with the refreshes
   expected before the target and widens when the trailing page
   state has been frozen for hours, the disagreement between
   candidate medians, and a pull back toward the anchor (the last
   printed state is itself a plausible answer). Together they aim
   at roughly 0.98 empirical coverage, the level the leaders hold.

6. Censored ends. Rates cap at 99.9; a near-zero forecast floors at
   the smallest positive rate in the case; a trophy nobody has
   earned yet gets a tier emergence prior instead of a fitted zero.

The domain facts stay: chapter-trophy medians are non-increasing
along the story chain, and the platinum never sits above a trophy
it requires. Standard library only, no randomness: two runs write
identical bytes.
"""

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
ND = NormalDist()

DAMP = 0.5           # curvature damping on extrapolated increments
SHRINK_K = 3.0       # slope shrinkage toward the tier pool
HL_HOURS = 30.0      # recency half-life of the fit weights, hours
PREC_K = 25.0        # earners behind a rate for half information weight
MAX_DLOGIT = 1.8     # largest logit move the median may make
RW_DAY = 0.5         # horizon random walk, logit per sqrt(day)
SIGMA_CAL = 0.90     # global band calibration toward ~0.98 coverage
SIGMA_MAX = 3.0      # widest band the program will write
ANCHOR_PULL = 0.45   # band widening per logit the median leaves the anchor
SOFTMAX_T = 0.01     # backtest-loss temperature of the blend weights
PERSIST_FLOOR = 0.15 # the anchor always keeps at least this blend weight
TIER_CAP = {"platinum": 20.0, "gold": 55.0, "silver": 80.0, "bronze": 97.0}
TIER_PRIOR = {"platinum": 0.03, "gold": 0.25, "silver": 0.6, "bronze": 1.2}
CANDS = ("persist", "a35", "a65", "earners", "trend", "sat")
DEFAULT_W = {"persist": 0.34, "a35": 0.16, "a65": 0.12,
             "earners": 0.12, "trend": 0.10, "sat": 0.16}


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_of(ts):
    """Hours since the epoch; reads before it land a little under 0."""
    return (ts - EPOCH).total_seconds() / 3600.0


def x_of_h(h):
    """ln(1 + hours), the log-time axis every growth fit uses."""
    return math.log1p(max(h, -0.95))


def clamp_pct(v):
    return min(99.99, max(0.01, v))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def logistic(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def z_for(tau):
    return ND.inv_cdf(tau)


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x; None when there is nothing
    to fit. Carries the residual sd and the moments shrinkage needs."""
    total_w = sum(ws)
    if total_w <= 0:
        return None
    xm = sum(w * x for x, w in zip(xs, ws)) / total_w
    ym = sum(w * y for y, w in zip(ys, ws)) / total_w
    sxx = sum(w * (x - xm) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return {"a": ym, "b": 0.0, "resid": 0.0, "W": total_w}
    sxy = sum(w * (x - xm) * (y - ym) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = ym - b * xm
    dof = max(1.0, total_w - 2.0)
    resid = math.sqrt(sum(w * (y - (a + b * x)) ** 2
                          for x, y, w in zip(xs, ys, ws)) / dof)
    return {"a": a, "b": b, "resid": resid, "W": total_w}


def median_of(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def page_states(history):
    """Collapse runs of identical reads into distinct page states:
    one entry per republication, stamped when it was first seen."""
    states = []
    prev = None
    for read in history:
        rates = read.get("rates") or ()
        sig = (read.get("players_tracked"), tuple(rates))
        if sig == prev:
            continue
        prev = sig
        ts = parse_ts(read["at"])
        states.append({"ts": ts, "h": hours_of(ts),
                       "n": read.get("players_tracked") or 0,
                       "rates": list(rates)})
    return states


def rate_of(state, i):
    rates = state["rates"]
    if i < len(rates):
        return rates[i]
    return None


def thin_states(states, cap):
    """Bound the states a fit or backtest walks over, always keeping
    the first, the last, and every state where the tracked count
    moved (a refresh boundary carries the republication arithmetic)."""
    n = len(states)
    if n <= cap:
        return states
    stride = -(-n // cap)
    keep = set(range(0, n, stride))
    keep.add(n - 1)
    for j in range(1, n):
        if states[j]["n"] != states[j - 1]["n"]:
            keep.add(j)
    return [states[j] for j in sorted(keep)]


def refresh_points(states):
    """Indices of the states where the tracked count moved: each one
    is a republication sample of the pool."""
    return [j for j, s in enumerate(states)
            if j == 0 or s["n"] != states[j - 1]["n"]]


def tracked_slope(states):
    """The pool's current growth rate per unit of log-time x,
    estimated only from refresh points after the last big step (a
    jump of 40% or more). Pool growth decelerates in clock hours --
    each doubling of elapsed time adds a similar multiple -- so the
    fit runs on the log-time axis like every other fit here; a
    per-hour slope extrapolated flat over days explodes the pool
    and crushes every forecast rate with it."""
    idx = refresh_points(states)
    pts = [(x_of_h(states[j]["h"]), math.log(max(states[j]["n"], 1)))
           for j in idx]
    if len(pts) < 2:
        return 0.0
    start = 0
    for k in range(1, len(pts)):
        if pts[k][1] - pts[k - 1][1] >= math.log(1.4):
            start = k
    seg = pts[start:]
    slope = None
    if len(seg) >= 2 and seg[-1][0] - seg[0][0] >= 0.05:
        x_last = seg[-1][0]
        ws = [0.5 ** ((x_last - x) / 1.5) for x, _ in seg]
        fit = wls([p[0] for p in seg], [p[1] for p in seg], ws)
        if fit is not None:
            slope = fit["b"]
    if slope is None:
        (x1, l1), (x2, l2) = pts[-2], pts[-1]
        if x2 > x1:
            slope = 0.6 * (l2 - l1) / (x2 - x1)
    if slope is None:
        slope = 0.0
    # Sanity: never more than twice the median per-interval slope
    # the page itself has shown, and never a shrinking pool.
    ivs = [(pts[k][1] - pts[k - 1][1]) / (pts[k][0] - pts[k - 1][0])
           for k in range(1, len(pts)) if pts[k][0] > pts[k - 1][0]]
    if ivs:
        slope = min(slope, max(0.0, 2.0 * median_of(ivs)))
    return min(max(slope, 0.0), 4.0)


def tracked_forecast(states, target_h):
    """The tracked pool at the target: last printed count grown at
    the segmented log-time rate, the increment damped like every
    other extrapolation here and the total growth capped at a
    week-one-plausible multiple."""
    n_last = max(states[-1]["n"], 1)
    dx = max(0.0, x_of_h(target_h) - x_of_h(states[-1]["h"]))
    growth = math.exp(tracked_slope(states) * dx / (1.0 + DAMP * dx))
    return n_last * min(growth, 12.0)


def trophy_series(states, i):
    """One trophy's observations over the page states: a list of
    (x, hours, rate, tracked, earners) with the rate present, where
    earners is the count the printed rate implies."""
    out = []
    for s in states:
        r = rate_of(s, i)
        if r is not None:
            n = max(s["n"], 0)
            out.append((x_of_h(s["h"]), s["h"], r, n, r / 100.0 * n))
    return out


def series_weights(series):
    """Recency x information weights. The information part is the
    earner count behind the printed rate: a rate backed by PREC_K
    earners carries half the weight of one backed by a huge pool,
    because binomial noise on a small count is wide."""
    h_last = series[-1][1]
    ws = []
    for x, h, r, n, e in series:
        info = 0.15 + 0.85 * e / (e + PREC_K)
        ws.append((0.5 ** ((h_last - h) / HL_HOURS)) * info)
    return ws


def earners_points(series, ws):
    """Points for the earners growth fit: (x, earners, weight)."""
    return [(x, e, w) for (x, h, r, n, e), w in zip(series, ws)]


def rate_points(series, ws):
    """Points for the rate trend fit: rates strictly inside (0, 100),
    so a printed 0.00 from a tiny pool cannot wreck the line."""
    return [(x, r, w) for (x, h, r, n, e), w in zip(series, ws)
            if 0.02 <= r <= 99.98]


def raw_fits(states, trophies):
    """First pass: every trophy's unshrunk earners-growth and
    rate-trend fits, from which the tier pools are built."""
    out = []
    for i in range(len(trophies)):
        series = trophy_series(states, i)
        if not series:
            out.append((None, None, series))
            continue
        ws = series_weights(series)
        ep = earners_points(series, ws)
        ef = None
        if len(ep) >= 3 and ep[-1][0] - ep[0][0] >= 0.10:
            ef = wls([p[0] for p in ep], [math.log(p[1] + 0.5) for p in ep],
                     [p[2] for p in ep])
        rp = rate_points(series, ws)
        rf = None
        if len(rp) >= 3 and rp[-1][0] - rp[0][0] >= 0.10:
            rf = wls([p[0] for p in rp], [logit(p[1]) for p in rp],
                     [p[2] for p in rp])
        out.append((ef, rf, series))
    return out


def pooled_slope(raw, trophies, which, tier=None):
    """Precision-weighted mean slope over a pool of fits; `which`
    picks the earners fit (0) or the rate fit (1)."""
    num = den = 0.0
    for t, entry in zip(trophies, raw):
        fit = entry[which]
        if fit is not None and (tier is None or t["tier"] == tier):
            prec = fit["W"] / (fit["resid"] ** 2 + 0.09)
            num += fit["b"] * prec
            den += prec
    return num / den if den > 0 else None


def shrunk(slope, w, pool):
    return (w / (w + SHRINK_K)) * slope + (SHRINK_K / (w + SHRINK_K)) * pool


def sat_pred(series, ws, xt, tier):
    """Saturating growth: logit(rate / ceiling) linear in x, the
    ceiling picked per trophy by weighted fit quality over a
    geometric ladder above the highest rate seen, topped by the
    tier cap. A rate bending toward its ceiling is this page's
    common shape, and a straight logit line cannot bend."""
    pts = rate_points(series, ws)
    if len(pts) < 4 or pts[-1][0] - pts[0][0] < 0.15:
        return None
    rmax = max(p[1] for p in pts)
    r_last = series[-1][2]
    ceilings = []
    mult = 1.12
    while rmax * mult <= min(99.9, TIER_CAP[tier] * 1.15):
        ceilings.append(rmax * mult)
        mult *= 1.35
    if TIER_CAP[tier] > rmax * 1.05:
        ceilings.append(TIER_CAP[tier])
    best = None
    for ceil in ceilings:
        sub = [(x, r, w) for x, r, w in pts if r < 0.985 * ceil]
        if len(sub) < 4:
            continue
        fit = wls([p[0] for p in sub],
                  [math.log(p[1] / (ceil - p[1])) for p in sub],
                  [p[2] for p in sub])
        if fit is None or fit["b"] <= -0.05:
            continue
        if best is None or fit["resid"] < best[0]:
            z = fit["a"] + fit["b"] * xt
            pred = ceil / (1.0 + math.exp(-z)) if z > -30 else 0.0
            resid = fit["resid"] / max(0.2, 1.0 - r_last / ceil)
            best = (fit["resid"], logit(min(pred, ceil * 0.999)),
                    min(1.5, max(0.05, resid)))
    if best is None:
        return None
    return best[1], best[2]


def candidates_for(states, trophies, xt, target_h, n_hat):
    """Every candidate's (median logit, residual) per trophy at the
    target. Candidates: persistence; the earners-space forecast; two
    persistence-anchored pulls toward it (35% and 65%); and the
    shrunk log-time trend of the printed rate itself."""
    raw = raw_fits(states, trophies)
    pools = {}
    for which, dflt in ((0, 0.6), (1, 0.3)):
        all_pool = pooled_slope(raw, trophies, which)
        if all_pool is None:
            all_pool = dflt
        pools[which] = {}
        for tier in ("platinum", "gold", "silver", "bronze"):
            p = pooled_slope(raw, trophies, which, tier)
            pools[which][tier] = all_pool if p is None else p
    out = []
    for i, t in enumerate(trophies):
        ef, rf, series = raw[i]
        preds = {}
        if series:
            anchor = logit(series[-1][2])
            p_last = min(0.98, max(0.005, series[-1][2] / 100.0))
            dx = max(0.0, xt - series[-1][0])
            ws = series_weights(series)
            preds["persist"] = (anchor, 0.25)
            if ef is not None:
                slope = min(3.5, max(0.0, shrunk(
                    ef["b"], ef["W"], pools[0][t["tier"]])))
                # The earners curve decelerates: blend the window
                # fit's slope with the local slope over the trailing
                # half of the x-range, so the extrapolation leaves
                # from the current growth rate, not the average of
                # the steep early climb.
                ep = earners_points(series, ws)
                if len(ep) >= 4:
                    x0, x1 = ep[0][0], ep[-1][0]
                    seg = [p for p in ep if p[0] >= x0 + 0.5 * (x1 - x0)]
                    if len(seg) >= 2 and x1 - seg[0][0] >= 0.15:
                        local = (math.log(ep[-1][1] + 0.5)
                                 - math.log(seg[0][1] + 0.5)) / (x1 - seg[0][0])
                        slope = 0.45 * slope + 0.55 * min(3.5, max(0.0, local))
                log_e = math.log(series[-1][4] + 0.5) + \
                    slope * dx / (1.0 + DAMP * dx)
                e_hat = max(math.exp(log_e) - 0.5, 0.0)
                rate_e = min(99.9, max(0.01, 100.0 * e_hat / max(n_hat, 1.0)))
                med_e = min(anchor + 1.35, max(anchor - 1.35, logit(rate_e)))
                resid_e = min(1.5, max(0.08, ef["resid"] / (1.0 - p_last)))
                preds["earners"] = (med_e, resid_e)
                preds["a35"] = (anchor + 0.35 * (med_e - anchor), resid_e)
                preds["a65"] = (anchor + 0.65 * (med_e - anchor), resid_e)
            if rf is not None:
                slope = min(3.0, max(-1.0, shrunk(
                    rf["b"], rf["W"], pools[1][t["tier"]])))
                med_t = anchor + slope * dx / (1.0 + DAMP * dx)
                preds["trend"] = (med_t, min(1.5, max(0.05, rf["resid"])))
            sp = sat_pred(series, ws, xt, t["tier"])
            if sp is not None:
                preds["sat"] = sp
        out.append(preds)
    return out


def pinball(med, sigma, actual, levels, zs):
    """The competition's loss of one quantile row against one
    realised logit; the in-program backtest scores with it."""
    total = 0.0
    for tau, z in zip(levels, zs):
        d = actual - (med + z * sigma)
        total += 2.0 * max(tau * d, (tau - 1.0) * d)
    return total / len(levels)


def blend_weights(states, trophies, levels, zs, xt, target_h):
    """Rolling-origin backtest on the case's own page states: refit
    every candidate on histories truncated at earlier states and
    score its quantile row against the last observed state. Origins
    sit just before refreshes as well as evenly spaced, because a
    candidate that cannot straddle a republication is worthless
    here; origins whose gap is closer to the case's real horizon
    count more. A softmax over the losses sets the weights, and
    persistence always keeps a floor share: it is the anchor."""
    bt = thin_states(states, 140)
    n = len(bt)
    if n < 6:
        return [dict(DEFAULT_W) for _ in trophies], [None] * len(trophies)
    x_last = x_of_h(bt[-1]["h"])
    dx_final = max(0.05, xt - x_last)
    origins = set()
    for k in range(4, n - 1):
        if bt[k + 1]["n"] != bt[k]["n"]:
            origins.add(k)
    step = (n - 6) / 5.0
    for j in range(6):
        origins.add(4 + round(j * step))
    origins.add(n - 3)
    origins.add(n - 2)
    origins = sorted(k for k in origins if 3 <= k <= n - 2)
    if len(origins) > 8:
        origins.sort(key=lambda k: abs((x_last - x_of_h(bt[k]["h"])) - dx_final))
        origins = sorted(origins[:8])
    if not origins:
        return [dict(DEFAULT_W) for _ in trophies], [None] * len(trophies)
    losses = [dict() for _ in trophies]
    counts = [dict() for _ in trophies]
    origin_preds = []
    for k in origins:
        sub = bt[:k + 1]
        gap = x_last - x_of_h(bt[k]["h"])
        # Origins whose gap matches the case's real horizon (on a
        # log scale) say the most about it: a candidate that wins
        # at a 2-hour gap may not win at 4.5 days, or the reverse.
        ow = 0.25 + 0.75 * math.exp(
            -abs(math.log((gap + 0.02) / (dx_final + 0.02))))
        n_hat_o = tracked_forecast(sub, bt[-1]["h"])
        preds = candidates_for(sub, trophies, x_last, bt[-1]["h"], n_hat_o)
        actuals = []
        for i in range(len(trophies)):
            r = rate_of(bt[-1], i)
            actuals.append(None if r is None else logit(r))
        origin_preds.append((ow, preds, actuals))
        for i in range(len(trophies)):
            actual_rate = rate_of(bt[-1], i)
            if actual_rate is None:
                continue
            actual = logit(actual_rate)
            for cand, (med, resid) in preds[i].items():
                sigma = min(2.5, max(0.30, resid))
                losses[i][cand] = losses[i].get(cand, 0.0) + \
                    ow * pinball(med, sigma, actual, levels, zs)
                counts[i][cand] = counts[i].get(cand, 0) + ow
    weights = []
    for i in range(len(trophies)):
        scored = {c: losses[i][c] / counts[i][c]
                  for c in counts[i] if counts[i][c] >= 1.0}
        if not scored:
            weights.append(dict(DEFAULT_W))
            continue
        best = min(scored.values())
        raw_w = {c: math.exp(-(v - best) / SOFTMAX_T)
                 for c, v in scored.items()}
        total = sum(raw_w.values())
        w = {c: raw_w.get(c, 0.0) / total for c in CANDS}
        if w["persist"] < PERSIST_FLOOR:
            rest = sum(v for c, v in w.items() if c != "persist")
            scale = (1.0 - PERSIST_FLOOR) / rest if rest > 0 else 0.0
            w = {c: (PERSIST_FLOOR if c == "persist" else v * scale)
                 for c, v in w.items()}
        weights.append(w)
    # Second pass over the same origins: the blended median's own
    # out-of-sample error, gap-weighted the same way. This is the
    # honest size of band component (a) -- the blend's mistakes are
    # smaller than a weighted mix of candidate residuals suggests,
    # because the candidates' errors partly cancel.
    bt_err = [None] * len(trophies)
    num = [0.0] * len(trophies)
    den = [0.0] * len(trophies)
    for ow, preds, actuals in origin_preds:
        for i in range(len(trophies)):
            if actuals[i] is None:
                continue
            avail = {c: weights[i].get(c, 0.0) for c in preds[i]}
            tw = sum(avail.values())
            if tw <= 0:
                continue
            med = sum(avail[c] / tw * preds[i][c][0] for c in avail)
            num[i] += ow * (med - actuals[i]) ** 2
            den[i] += ow
    for i in range(len(trophies)):
        if den[i] > 0:
            bt_err[i] = math.sqrt(num[i] / den[i])
    return weights, bt_err


def refresh_component(states, horizon_h, at_ts, n_trophies):
    """The refresh-risk band term. Its scale is the RMS logit move
    the page itself makes between consecutive states; it ramps in
    with the chance that a refresh lands inside the horizon at all,
    and a trailing state frozen for hours widens it further -- a
    refresh is overdue, and the answer will be a refreshed page."""
    starts = [s["h"] for s in states]
    gaps = [starts[j] - starts[j - 1] for j in range(1, len(states))]
    med_gap = max(median_of(gaps) if gaps else 6.0, 0.25)
    # The answer is one snapshot, lagging the latent state by up to
    # one refresh interval: the term is one step's shock, ramped in
    # by the chance a refresh lands inside the horizon at all.
    ramp = math.sqrt(min(1.0, horizon_h / med_gap))
    ss = 0.0
    cnt = 0
    for j in range(1, len(states)):
        for i in range(n_trophies):
            r0 = rate_of(states[j - 1], i)
            r1 = rate_of(states[j], i)
            if r0 is not None and r1 is not None and \
                    0.02 <= r0 <= 99.98 and 0.02 <= r1 <= 99.98:
                d = logit(r1) - logit(r0)
                ss += d * d
                cnt += 1
    shock = min(math.sqrt(ss / cnt) if cnt else 0.35, 1.1)
    frozen_h = max(0.0, (at_ts - states[-1]["ts"]).total_seconds() / 3600.0)
    freeze_mult = 1.0 + 0.5 * min(1.0, frozen_h / 4.0)
    return shock * ramp * freeze_mult, frozen_h


def emergence_prior(states, trophies, i, floor_rate):
    """What a trophy nobody has earned yet is worth: the median
    rate at which same-tier trophies first appeared during the
    recorded history, or the case's smallest positive rate when
    no same-tier trophy emerged on record."""
    tier = trophies[i]["tier"]
    firsts = []
    for j, t in enumerate(trophies):
        if t["tier"] != tier:
            continue
        seen_zero = False
        for s in states:
            r = rate_of(s, j)
            if r is None:
                continue
            if r <= 0.0:
                seen_zero = True
            elif seen_zero:
                firsts.append(r)
                break
    return median_of(firsts) if firsts else floor_rate


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    levels = case.get("levels") or LEVELS
    zs = [z_for(t) for t in levels]
    at = parse_ts(case["at"])
    target = parse_ts(case["target"])
    horizon_min = case.get("horizon_minutes")
    if horizon_min is None:
        horizon_min = (target - at).total_seconds() / 60.0
    days = horizon_min / 1440.0
    horizon_h = horizon_min / 60.0
    target_h = hours_of(target)
    xt = x_of_h(target_h)
    n_tr = len(trophies)

    last_read = history[-1] if history else {}
    last_rates = last_read.get("rates") or ()
    positives = [r for r in last_rates if r is not None and r > 0]
    floor_rate = min(positives) if positives else 0.05

    def row(med_pct, sigma):
        lg = logit(med_pct)
        qs = [round(min(99.9, max(0.01, logistic(lg + z * sigma))), 4)
              for z in zs]
        if qs[0] < 0.10:
            # The band's bottom sits at the printed grid's floor:
            # write the censored end itself. Scoring clamps
            # quantiles to [0.01, 99.99] for the loss, so this is
            # free there, and a realised 0.00 is then covered.
            qs[0] = 0.0
        if qs[-1] >= 99.9:
            # Same at the top: a saturated band top means the
            # censored 100.00 itself.
            qs[-1] = 100.0
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        return qs

    states = page_states(history)

    if len(states) <= 1:
        # One frozen page (the practice shape): nothing has ever
        # refreshed on record. Persistence is the only honest
        # median, shifted down by the dilution a first refresh
        # brings while the pool is still small, under a band wide
        # enough to hold that refresh.
        n_pool = states[0]["n"] if states else \
            (last_read.get("players_tracked") or 0)
        first_ts = states[0]["ts"] if states else at
        frozen_h = max(0.0, (at - first_ts).total_seconds() / 3600.0)
        small = 5000.0 / (5000.0 + max(n_pool, 0))
        # A first refresh multiplies the pool several-fold while
        # earners only grow: the median rate scales down by the
        # expected dilution factor (a multiplicative shift, so the
        # logit move is large for rare trophies, small for common
        # ones), scaled by how small the pool still is and how
        # long the page has been frozen.
        factor = 1.0 - 0.55 * small * min(1.0, frozen_h / 3.0)
        shock = 1.35 * (0.35 + 0.65 * small)
        sigma = math.sqrt((RW_DAY * math.sqrt(days)) ** 2 + shock ** 2)
        rows = {}
        for i, t in enumerate(trophies):
            r = last_rates[i] if i < len(last_rates) else None
            if r is None:
                r = TIER_PRIOR[t["tier"]]
            if r >= 99.9:
                q99 = row(99.9, 0.04)
                q99[-1] = 100.0
                rows[t["id"]] = q99
            elif r <= 0.0:
                med = emergence_prior(states, trophies, i, floor_rate)
                rows[t["id"]] = row(med, max(sigma, 1.0))
            else:
                med = r * factor
                rows[t["id"]] = row(min(med, TIER_CAP[t["tier"]]), sigma)
        return {"trophies": [{"id": t["id"], "q": rows[t["id"]]}
                             for t in trophies]}

    n_hat = tracked_forecast(states, target_h)
    preds = candidates_for(states, trophies, xt, target_h, n_hat)
    weights, bt_err = blend_weights(states, trophies, levels, zs,
                                    xt, target_h)
    refresh_sig, _frozen = refresh_component(states, horizon_h, at, n_tr)
    rw = RW_DAY * math.sqrt(days)

    medians = {}
    sigmas = {}
    full_ids = set()
    for i, t in enumerate(trophies):
        tid = t["id"]
        series = trophy_series(states, i)
        last_rate = series[-1][2] if series else None
        if last_rate is None:
            medians[tid] = min(TIER_PRIOR[t["tier"]], TIER_CAP[t["tier"]])
            sigmas[tid] = 1.2
            continue
        if last_rate >= 99.9:
            medians[tid] = 99.9
            sigmas[tid] = 0.04
            full_ids.add(tid)
            continue
        if last_rate <= 0.0:
            medians[tid] = emergence_prior(states, trophies, i, floor_rate)
            sigmas[tid] = max(1.0, refresh_sig)
            continue
        cand = preds[i]
        w = weights[i]
        avail = {c: w.get(c, 0.0) for c in cand}
        total_w = sum(avail.values())
        if total_w <= 0:
            avail = {"persist": 1.0}
            total_w = 1.0
        med = sum(avail[c] / total_w * cand[c][0] for c in avail)
        resid = math.sqrt(sum(avail[c] / total_w * cand[c][1] ** 2
                              for c in avail))
        if bt_err[i] is not None:
            resid = min(2.0, max(0.15, bt_err[i]))
        disagree = math.sqrt(sum(avail[c] / total_w * (cand[c][0] - med) ** 2
                                 for c in avail))
        anchor = logit(last_rate)
        med = min(anchor + MAX_DLOGIT, max(anchor - MAX_DLOGIT, med))
        med_pct = clamp_pct(min(logistic(med), TIER_CAP[t["tier"]]))
        if med_pct < floor_rate:
            med_pct = floor_rate
        var = resid ** 2 + rw ** 2 + refresh_sig ** 2 + disagree ** 2 + \
            (ANCHOR_PULL * abs(logit(med_pct) - anchor)) ** 2
        sigma = SIGMA_CAL * math.sqrt(var)
        if days >= 0.5:
            sigma = max(sigma, 0.34)
        medians[tid] = med_pct
        sigmas[tid] = min(sigma, SIGMA_MAX)

    # Story order is a fact, not a fit: chapter rates cannot rise
    # along the chain, and the platinum cannot sit above any trophy
    # it requires. Pool adjacent violators on the logit medians and
    # apply only the decreases the fact forces.
    chain = []
    for t in trophies:
        m = re.search(r"Complete Chapter (\d+)\.", t.get("description") or "")
        if m:
            chain.append((int(m.group(1)), t["id"]))
    chain.sort()
    chain_ids = [tid for _, tid in chain]
    for t in trophies:
        if (t.get("description") or "").strip() == "Complete the final mission.":
            chain_ids.append(t["id"])
    if len(chain_ids) >= 2:
        blocks = []
        for tid in chain_ids:
            blocks.append([logit(medians[tid]), 1])
            while len(blocks) >= 2 and \
                    blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
                top = blocks.pop()
                blocks[-1][0] += top[0]
                blocks[-1][1] += top[1]
        flat = []
        for total, count in blocks:
            flat.extend([total / count] * count)
        for tid, lv in zip(chain_ids, flat):
            lowered = logistic(lv)
            if lowered < medians[tid]:
                medians[tid] = lowered
    plat = next((t for t in trophies if t["tier"] == "platinum"), None)
    if plat is not None:
        others = [medians[t["id"]] for t in trophies if t["id"] != plat["id"]]
        if others and medians[plat["id"]] > min(others):
            medians[plat["id"]] = min(others)

    out = {}
    for t in trophies:
        q = row(medians[t["id"]], sigmas[t["id"]])
        if t["id"] in full_ids:
            # Already at the printed maximum: the answer can be the
            # censored 100.00 itself, so the top quantile goes there.
            q[-1] = 100.0
        out[t["id"]] = q
    return {"trophies": [{"id": t["id"], "q": out[t["id"]]}
                         for t in trophies]}


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        (out_dir / f"{path.stem}.out").write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
