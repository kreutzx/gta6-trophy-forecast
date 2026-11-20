"""Week-one GTA VI trophy rarity forecaster, v2 (make.fun:
gta6-trophy-rarity-forecast-01).

What v2 adds over v1, following what the strongest public entrants do:

1. It models the measurement process, not only the earning process. The
   page a case records only changes when Exophase republishes it, so the
   history is compressed into its distinct page states, and each state is
   weighted by how much information it carries: a rate printed from 638
   tracked players is a binomial draw with a handful of earners behind it,
   so fit weights scale with the earner count behind each rate, not just
   with recency.

2. It runs a pool of candidate models per trophy and blends them:
   - a shrunk log-time trend (v1's model: WLS of logit(rate) on
     ln(1 + hours), slope shrunk to the tier pool, damped extrapolation);
   - a saturating logistic-growth curve (logit(p / ceiling) linear in
     ln(1 + hours), ceiling picked per trophy by fit quality);
   - a counts model built on the house's own republication arithmetic:
     earners and tracked players are extrapolated separately in log
     space and their slope difference — the rate's growth with the
     denominator's movement removed — drives an anchored forecast, so
     when a refresh jumps the denominator and the printed rate drops
     while earners keep rising, this candidate's slope stays positive
     where the raw rate trend's goes negative;
   - persistence, which is the right model inside a republication plateau.
   The blend weights come from a rolling-origin backtest run inside the
   program on the case's own states: each candidate is refit on truncated
   histories and scored with the competition's pinball loss against the
   last observed state, and a softmax over those losses sets the weights.

3. The spread is built from the blended fit residual and the forecast
   horizon, and is widened by how much the candidate medians disagree:
   where a saturating curve and a straight trend tell different stories,
   the honest band is wider.

The domain facts from v1 are kept: chapter-trophy medians are
non-increasing along the story chain, and the platinum never sits above a
trophy it requires. Standard library only, no randomness: two runs write
identical bytes.
"""

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_TABLE = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}

WIN_H = 72.0        # trailing window for the fits, hours
MIN_WIN = 8         # ...but never fewer than this many states
HL_HOURS = 24.0     # recency half-life of the fit weights, hours
SHRINK_K = 3.0      # trend slope shrinkage toward the tier pool
DAMP = 0.5          # curvature damping of the trend increment
TAU_B = 0.12        # per-trophy slope dispersion, per unit of x travelled
C_RW = 0.08         # random-walk spread floor, logit per sqrt(day)
C0 = 0.08           # constant logit spread (refresh staleness)
SIGMA_MIN = 0.30    # smallest spread once the horizon reaches half a day
MAX_DLOGIT = 2.3    # largest logit move the median may make
PREC_K = 25.0       # earners behind a rate for half information weight
SOFTMAX_T = 0.008   # backtest-loss temperature of the blend weights
DIS_W = 1.0         # how much candidate disagreement widens the spread
RESID_MULT = 1.0    # multiplier on the blended fit residual's variance
TIER_CAP = {"platinum": 20.0, "gold": 55.0, "silver": 80.0, "bronze": 97.0}
TIER_PRIOR = {"platinum": 0.03, "gold": 0.25, "silver": 0.6, "bronze": 1.2}
CANDS = ("trend", "logistic", "counts", "persist")


def z_for(tau):
    """The standard normal's quantile at tau: the scored levels use the
    table; any other level a case might ask for goes through Acklam's
    rational approximation, so an unexpected level can never crash a run."""
    if tau in Z_TABLE:
        return Z_TABLE[tau]
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    if tau < 0.02425:
        q = math.sqrt(-2.0 * math.log(tau))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if tau > 0.97575:
        q = math.sqrt(-2.0 * math.log(1.0 - tau))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = tau - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def x_of(ts):
    """ln(1 + hours since the epoch); hours before it stay above -1."""
    h = (ts - EPOCH).total_seconds() / 3600.0
    return math.log(1.0 + max(h, -0.95))


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


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x with residual sd and fit moments."""
    total_w = sum(ws)
    if total_w <= 0:
        return None
    xm = sum(w * x for x, w in zip(xs, ws)) / total_w
    ym = sum(w * y for y, w in zip(ys, ws)) / total_w
    sxx = sum(w * (x - xm) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return {"a": ym, "b": 0.0, "resid": 0.0, "xm": xm, "sxx": sxx, "W": total_w}
    sxy = sum(w * (x - xm) * (y - ym) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = ym - b * xm
    dof = max(1.0, total_w - 2.0)
    resid = math.sqrt(sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / dof)
    return {"a": a, "b": b, "resid": resid, "xm": xm, "sxx": sxx, "W": total_w}


def distinct_states(history):
    """Collapse runs of identical reads: the page between two refreshes."""
    states = []
    prev = None
    for read in history:
        sig = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if sig != prev:
            states.append(read)
            prev = sig
    return states


def rate_at(read, i):
    rates = read.get("rates") or ()
    if i < len(rates):
        return rates[i]
    return None


def pinball(med, sigma, actual, levels):
    """The competition's loss of one quantile row against one realised
    logit, used by the in-program rolling-origin backtest."""
    total = 0.0
    for tau in levels:
        d = actual - (med + z_for(tau) * sigma)
        total += 2.0 * max(tau * d, (tau - 1.0) * d)
    return total / len(levels)


def series_of(states, i):
    """One trophy's observations over the page states: a list of
    (x, rate, players_tracked, time) with the rate present."""
    out = []
    for s in states:
        r = rate_at(s, i)
        if r is not None:
            ts = parse_ts(s["at"])
            out.append((x_of(ts), r, s.get("players_tracked") or 0, ts))
    return out


def weights_of(series):
    """Recency x information weights. The information part is the earner
    count behind the printed rate: earners = rate% x players tracked, and
    a rate backed by PREC_K earners carries half the weight of one backed
    by a huge tracked pool (binomial noise on the logit scale runs like
    1/sqrt(earners)). A floor keeps early low-count states in the fit."""
    t_last = series[-1][3]
    ws = []
    for x, r, n, ts in series:
        age_h = (t_last - ts).total_seconds() / 3600.0
        earners = max(0.0, r) / 100.0 * max(n, 0)
        info = 0.15 + 0.85 * earners / (earners + PREC_K)
        ws.append((0.5 ** (age_h / HL_HOURS)) * info)
    return ws


def windowed(series, ws):
    """Keep the trailing WIN_H hours of states, at least MIN_WIN of them."""
    if len(series) <= MIN_WIN:
        return series, ws
    t_last = series[-1][3]
    keep = [j for j, s in enumerate(series)
            if (t_last - s[3]).total_seconds() / 3600.0 <= WIN_H]
    if len(keep) < MIN_WIN:
        keep = list(range(len(series) - MIN_WIN, len(series)))
    return [series[j] for j in keep], [ws[j] for j in keep]


def fit_points(series, ws):
    """Points usable for curve fitting, with their weights: rates strictly
    inside (0, 100), so a printed 0.00 from a tiny tracked pool cannot
    wreck a fit. Returns (x, rate, players, weight) rows."""
    return [(x, r, n, w) for (x, r, n, ts), w in zip(series, ws)
            if 0.02 <= r <= 99.98]


def trend_fit(series, ws):
    pts = fit_points(series, ws)
    if len(pts) < 3 or pts[-1][0] - pts[0][0] < 0.10:
        return None
    return wls([p[0] for p in pts], [logit(p[1]) for p in pts],
               [p[3] for p in pts])


def trend_pred(series, ws, xt, pool_slope):
    """Anchored trend: start from the last observed logit and travel with
    the shrunk slope, damped, exactly v1's median for this candidate."""
    fit = trend_fit(series, ws)
    if fit is None:
        return None
    wgt = fit["W"] / (fit["W"] + SHRINK_K)
    slope = wgt * fit["b"] + (1.0 - wgt) * pool_slope
    slope = min(3.0, max(-1.0, slope))
    dx = max(0.0, xt - series[-1][0])
    med = logit(series[-1][1]) + slope * dx / (1.0 + DAMP * dx)
    return med, max(fit["resid"], 0.05)


def logistic_pred(series, ws, xt, tier):
    """Saturating growth: logit(p / ceiling) linear in x, the ceiling
    chosen per trophy by weighted fit quality in logit space."""
    pts = fit_points(series, ws)
    if len(pts) < 4:
        return None
    xs = [p[0] for p in pts]
    if xs[-1] - xs[0] < 0.15:
        return None
    rmax = max(p[1] for p in pts)
    cap = TIER_CAP[tier]
    ceils = []
    for c in (rmax * 1.15, rmax * 1.4, rmax * 1.8, rmax * 2.4, rmax * 3.2,
              cap * 0.7, cap * 0.85, cap, min(99.9, cap * 1.15)):
        if rmax * 1.05 < c <= 99.95 and all(abs(c - o) > 0.5 for o in ceils):
            ceils.append(c)
    best = None
    for ceil in ceils:
        sub = [(x, r, w) for x, r, n, w in pts if r < 0.985 * ceil]
        if len(sub) < 4:
            continue
        sx = [p[0] for p in sub]
        zy = [math.log(p[1] / (ceil - p[1])) for p in sub]
        sw = [p[2] for p in sub]
        fit = wls(sx, zy, sw)
        if fit is None or fit["b"] <= -0.05:
            continue
        sse = 0.0
        wsum = 0.0
        for (x, r, w) in sub:
            # Same overflow guard the forecast below already uses: a steep
            # fit evaluated at an early x can push z far negative, and
            # math.exp(-z) would raise instead of scoring the ceiling.
            z = fit["a"] + fit["b"] * x
            pred = ceil / (1.0 + math.exp(-z)) if z > -30 else 0.0
            sse += w * (logit(pred) - logit(r)) ** 2
            wsum += w
        if wsum <= 0:
            continue
        score = sse / wsum
        if best is None or score < best[0]:
            z = fit["a"] + fit["b"] * xt
            pred = ceil / (1.0 + math.exp(-z)) if z > -30 else 0.0
            pred = min(pred, ceil * 0.999)
            best = (score, logit(max(pred, 0.01)), math.sqrt(score))
    if best is None:
        return None
    return best[1], max(best[2], 0.05)


def counts_pred(series, ws, xt, pool_slope):
    """Republication arithmetic, in slope form. Earners and tracked
    players are extrapolated separately in log space; their slope
    difference is the rate's growth with the denominator's movement
    removed, converted to a logit slope at the last rate and anchored
    there like the trend. When a refresh jumps the denominator and the
    printed rate drops while earners keep rising, this candidate's slope
    stays positive where the raw rate trend's goes negative; and being
    anchored and damped, it cannot run away the way a ratio of two
    extrapolated levels can."""
    pts = [(x, r, n, w) for (x, r, n, ts), w in zip(series, ws)
           if n and n > 0 and r is not None]
    if len(pts) < 3 or pts[-1][0] - pts[0][0] < 0.10:
        return None
    xs = [p[0] for p in pts]
    wsub = [p[3] for p in pts]
    fe = wls(xs, [math.log(p[1] / 100.0 * p[2] + 0.5) for p in pts], wsub)
    fn = wls(xs, [math.log(p[2]) for p in pts], wsub)
    if fe is None or fn is None:
        return None
    p_last = min(0.98, max(0.005, series[-1][1] / 100.0))
    slope = (fe["b"] - fn["b"]) / (1.0 - p_last)
    slope = 0.5 * slope + 0.5 * pool_slope
    slope = min(3.0, max(-1.0, slope))
    dx = max(0.0, xt - series[-1][0])
    med = logit(series[-1][1]) + slope * dx / (1.0 + DAMP * dx)
    return med, max(0.15, 0.5 * fe["resid"])


def candidate_preds(states, trophies, xt):
    """Every candidate's (median logit, residual) per trophy for a target
    x, with trend slopes shrunk toward precision-weighted tier pools."""
    series = [series_of(states, i) for i in range(len(trophies))]
    weights = [weights_of(s) if s else [] for s in series]
    win = [windowed(s, w) if s else (s, w) for s, w in zip(series, weights)]
    fits = [trend_fit(s, w) if s else None for s, w in win]

    def pooled(tier=None):
        num = den = 0.0
        for t, fit in zip(trophies, fits):
            if fit is not None and (tier is None or t["tier"] == tier):
                prec = fit["W"] / (fit["resid"] ** 2 + 0.09)
                num += fit["b"] * prec
                den += prec
        return num / den if den > 0 else None

    pool_all = pooled()
    if pool_all is None:
        pool_all = 0.4
    pool_tier = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        p = pooled(tier)
        pool_tier[tier] = pool_all if p is None else p

    out = []
    for i, t in enumerate(trophies):
        s, w = win[i]
        preds = {}
        if s:
            tp = trend_pred(s, w, xt, pool_tier[t["tier"]])
            if tp is not None:
                preds["trend"] = tp
            lp = logistic_pred(s, w, xt, t["tier"])
            if lp is not None:
                preds["logistic"] = lp
            cp = counts_pred(s, w, xt, pool_tier[t["tier"]])
            if cp is not None:
                preds["counts"] = cp
            preds["persist"] = (logit(s[-1][1]), 0.5)
        out.append(preds)
    return out


def blend_weights(states, trophies, levels, xt):
    """Rolling-origin backtest on the case's own page states: refit every
    candidate on histories truncated at earlier states and score its
    quantile row against the last observed state with the competition's
    pinball loss. Origins whose forecast gap is closer to the case's real
    horizon count more (a candidate that wins at a 2-hour gap may not win
    at 4.5 days). Softmax over the losses gives each trophy's blend
    weights; trophies without usable origins get a fixed prior blend."""
    n = len(states)
    if n > 160:
        # A very chatty recorder: thin the states for the backtest only
        # (the final forecast still uses every state), keeping the runtime
        # comfortably inside the sandbox limit. The stride sample is joined
        # by every state where the tracked-player count jumps by a quarter
        # or more, because the counts candidate reads the republication
        # arithmetic off those boundaries; a stride that skipped one would
        # smooth the jump away and mis-score that candidate.
        stride = -(-n // 160)
        keep = set(range(0, n, stride))
        keep.add(n - 1)
        for j in range(1, n):
            prev_n = states[j - 1].get("players_tracked") or 0
            this_n = states[j].get("players_tracked") or 0
            if prev_n > 0 and abs(this_n - prev_n) >= 0.25 * prev_n:
                keep.add(j)
        states = [states[j] for j in sorted(keep)]
        n = len(states)
    x_last = x_of(parse_ts(states[-1]["at"]))
    dx_final = max(0.05, xt - x_last)
    origins = []
    for k in range(5, n - 1):
        if x_last - x_of(parse_ts(states[k]["at"])) >= 0.30:
            origins.append(k)
    if len(origins) > 6:
        step = (len(origins) - 1) / 5.0
        origins = sorted({origins[round(j * step)] for j in range(6)})
    losses = [dict() for _ in trophies]
    counts = [dict() for _ in trophies]
    for k in origins:
        gap = x_last - x_of(parse_ts(states[k]["at"]))
        ow = 0.35 + 0.65 * min(1.0, gap / dx_final)
        preds = candidate_preds(states[:k + 1], trophies, x_last)
        for i, t in enumerate(trophies):
            actual_rate = rate_at(states[-1], i)
            if actual_rate is None:
                continue
            actual = logit(actual_rate)
            for cand, (med, resid) in preds[i].items():
                sigma = min(2.5, max(0.30, resid))
                losses[i][cand] = losses[i].get(cand, 0.0) + \
                    ow * pinball(med, sigma, actual, levels)
                counts[i][cand] = counts[i].get(cand, 0) + ow
    weights = []
    for i, t in enumerate(trophies):
        scored = {c: losses[i][c] / counts[i][c]
                  for c in counts[i] if counts[i][c] >= 1.0}
        if not scored:
            weights.append({"trend": 0.45, "logistic": 0.30,
                            "counts": 0.25, "persist": 0.0})
            continue
        best = min(scored.values())
        raw = {c: math.exp(-(v - best) / SOFTMAX_T) for c, v in scored.items()}
        total = sum(raw.values())
        weights.append({c: raw.get(c, 0.0) / total for c in CANDS})
    return weights


def row_from(median, sigma, levels):
    lg = logit(median)
    return [round(logistic(lg + z_for(tau) * sigma), 4) for tau in levels]


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    levels = case.get("levels") or LEVELS
    at = parse_ts(case["at"])
    target = parse_ts(case["target"])
    horizon_min = case.get("horizon_minutes")
    if horizon_min is None:
        horizon_min = (target - at).total_seconds() / 60.0
    days = horizon_min / 1440.0
    xt = x_of(target)

    def fallback_row(i, spread):
        rate = rate_at(history[-1], i) if history else None
        lg = logit(rate if rate is not None else TIER_PRIOR[trophies[i]["tier"]])
        return [round(logistic(lg + z_for(tau) * spread), 4) for tau in levels]

    states = distinct_states(history)
    xs_all = [x_of(parse_ts(s["at"])) for s in states]
    if len(states) < 3 or xs_all[-1] - xs_all[0] < 0.03:
        # One page state (the practice cases, or a launch-night plateau):
        # nothing to fit, so persistence with a refresh-sized spread.
        spread = max(0.5 * math.sqrt(days), 0.55)
        return {"trophies": [{"id": t["id"], "q": fallback_row(i, spread)}
                             for i, t in enumerate(trophies)]}

    x_last = xs_all[-1]
    dx = max(0.0, xt - x_last)
    preds = candidate_preds(states, trophies, xt)
    weights = blend_weights(states, trophies, levels, xt)

    rows = {}
    medians = {}
    for i, t in enumerate(trophies):
        tid = t["id"]
        last_rate = rate_at(history[-1], i)
        if last_rate is None:
            for s in reversed(states):
                if rate_at(s, i) is not None:
                    last_rate = rate_at(s, i)
                    break
        if last_rate is None:
            med = min(TIER_PRIOR[t["tier"]] * math.exp(0.4 * dx / (1.0 + DAMP * dx)),
                      TIER_CAP[t["tier"]])
            medians[tid] = med
            rows[tid] = (med, 1.2)
            continue
        if last_rate >= 99.9:
            medians[tid] = 99.99
            rows[tid] = (99.99, 0.05)
            continue
        cand = preds[i]
        w = dict(weights[i])
        avail = {c: w.get(c, 0.0) for c in cand}
        total_w = sum(avail.values())
        if total_w <= 0:
            avail = {"trend": 1.0} if "trend" in cand else {next(iter(cand)): 1.0}
            total_w = 1.0
        med = sum(avail[c] / total_w * cand[c][0] for c in avail)
        resid = sum(avail[c] / total_w * cand[c][1] for c in avail)
        disagree = math.sqrt(sum(avail[c] / total_w * (cand[c][0] - med) ** 2
                                 for c in avail))
        anchor = logit(last_rate)
        med = min(anchor + MAX_DLOGIT, max(anchor - MAX_DLOGIT, med))
        med_pct = clamp_pct(min(logistic(med), TIER_CAP[t["tier"]]))
        var = (resid ** 2) * RESID_MULT + (dx * TAU_B) ** 2 + \
            (C_RW * math.sqrt(days)) ** 2 + C0 ** 2 + (DIS_W * disagree) ** 2
        sigma = math.sqrt(var)
        if days >= 0.5:
            sigma = max(sigma, SIGMA_MIN)
        medians[tid] = med_pct
        rows[tid] = (med_pct, min(sigma, 3.0))

    # Story order is a fact, not a fit: chapter rates cannot rise along the
    # chain, and the platinum cannot sit above any trophy it requires.
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
            while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
                top = blocks.pop()
                blocks[-1][0] += top[0]
                blocks[-1][1] += top[1]
        adjusted = []
        for total, count in blocks:
            adjusted.extend([total / count] * count)
        for tid, lv in zip(chain_ids, adjusted):
            new_med = logistic(lv)
            if new_med < medians[tid]:
                rows[tid] = (new_med, rows[tid][1])
                medians[tid] = new_med
    plat = next((t for t in trophies if t["tier"] == "platinum"), None)
    if plat is not None:
        others = [medians[t["id"]] for t in trophies if t["id"] != plat["id"]]
        if others and medians[plat["id"]] > min(others):
            rows[plat["id"]] = (min(others), rows[plat["id"]][1])
            medians[plat["id"]] = min(others)

    return {"trophies": [{"id": t["id"], "q": row_from(*rows[t["id"]], levels)}
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
