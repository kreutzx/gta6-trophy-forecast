"""Week-one GTA VI trophy rarity forecaster (make.fun: gta6-trophy-rarity-forecast-01).

Reads every case file (<id>.in) in the input directory and writes, for each
trophy, five quantiles of its Exophase earn rate at the case's target.

Model, in one paragraph: the page a case's history records only changes when
Exophase refreshes it, so the history is first compressed into its distinct
page states. On the logit scale each trophy's rate is then extrapolated from
its last observed value with a local trend: a weighted least-squares slope of
logit(rate) against ln(1 + hours since 2026-11-19T00:00Z) over the trailing
36 hours (recency half-life 24 h), shrunk toward the pooled slope of its tier
so one noisy trophy cannot run away, and damped over the horizon
(dx / (1 + 0.5*dx)) because earn curves bend as they saturate while a line
does not. The spread around the median combines the fit's prediction
variance, a slope-dispersion term over the horizon, a small random-walk
floor and a constant for refresh staleness; it is calibrated so the
5%..95% band covers the realised rate about nine times in ten. Two facts of
the domain are enforced on the medians: rates for "Complete Chapter k."
trophies are non-increasing in k (with the final-mission trophy last), and
the platinum's rate never exceeds any other trophy's, since earning it
requires earning them all. With too little history the program falls back
to persistence with a wide spread.

Standard library only, no randomness: two runs write identical bytes.
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

WIN_H = 36.0        # trailing window for the trend fit, hours
HL_HOURS = 24.0     # recency half-life of the fit weights, hours
SHRINK_K = 3.0      # slope shrinkage toward the tier pool
DAMP = 0.5          # curvature damping of the extrapolated increment
TAU_B = 0.12        # per-trophy slope dispersion, per unit of x travelled
C_RW = 0.15         # random-walk spread floor, logit per sqrt(day)
C0 = 0.08           # constant logit spread (refresh staleness)
SIGMA_MIN = 0.30    # smallest spread once the horizon reaches half a day
MAX_DLOGIT = 2.3    # largest logit move the median may make
TIER_CAP = {"platinum": 20.0, "gold": 55.0, "silver": 80.0, "bronze": 97.0}
TIER_PRIOR = {"platinum": 0.03, "gold": 0.25, "silver": 0.6, "bronze": 1.2}


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


def pred_var(fit, xt):
    if fit is None or fit["sxx"] <= 1e-12:
        return 0.25
    return fit["resid"] ** 2 * (1.0 + 1.0 / fit["W"] + (xt - fit["xm"]) ** 2 / fit["sxx"])


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
        spread = max(0.5 * math.sqrt(days), 0.55)
        return {"trophies": [{"id": t["id"], "q": fallback_row(i, spread)}
                             for i, t in enumerate(trophies)]}

    x_last = xs_all[-1]
    dx = max(0.0, xt - x_last)
    dx_eff = dx / (1.0 + DAMP * dx)
    t_last = parse_ts(states[-1]["at"])

    win_idx = [j for j, s in enumerate(states)
               if (t_last - parse_ts(s["at"])).total_seconds() / 3600.0 <= WIN_H]
    if len(win_idx) < 5:
        win_idx = list(range(max(0, len(states) - 5), len(states)))
    win_states = [states[j] for j in win_idx]
    win_xs = [xs_all[j] for j in win_idx]
    win_ws = [0.5 ** ((t_last - parse_ts(s["at"])).total_seconds() / 3600.0 / HL_HOURS)
              for s in win_states]

    fits = []
    for i in range(len(trophies)):
        xs, ys, ws_ = [], [], []
        for s, x, w in zip(win_states, win_xs, win_ws):
            r = rate_at(s, i)
            if r is not None:
                xs.append(x)
                ys.append(logit(r))
                ws_.append(w)
        fits.append(wls(xs, ys, ws_) if len(xs) >= 3 and xs[-1] - xs[0] >= 0.10 else None)

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
            med = min(TIER_PRIOR[t["tier"]] * math.exp(pool_tier[t["tier"]] * dx_eff),
                      TIER_CAP[t["tier"]])
            medians[tid] = med
            rows[tid] = (med, 1.2)
            continue
        if last_rate >= 99.9:
            medians[tid] = 99.99
            rows[tid] = (99.99, 0.05)
            continue
        fit = fits[i]
        if fit is not None:
            wgt = fit["W"] / (fit["W"] + SHRINK_K)
            slope = wgt * fit["b"] + (1.0 - wgt) * pool_tier[t["tier"]]
            var = pred_var(fit, xt) + (dx * TAU_B) ** 2
        else:
            slope = pool_tier[t["tier"]]
            var = 0.45 ** 2 + (dx * 0.30) ** 2
        slope = min(3.0, max(-1.0, slope))
        dl = min(MAX_DLOGIT, max(-MAX_DLOGIT, slope * dx_eff))
        med = logistic(logit(last_rate) + dl)
        med = clamp_pct(min(med, TIER_CAP[t["tier"]]))
        sigma = math.sqrt(var + (C_RW * math.sqrt(days)) ** 2 + C0 ** 2)
        if days >= 0.5:
            sigma = max(sigma, SIGMA_MIN)
        medians[tid] = med
        rows[tid] = (med, min(sigma, 3.0))

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
