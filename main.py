#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster -- final-regime trend model (v6).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its
earn rate on Exophase at the case's target (2026-11-26T00:00:00Z for the
final), as five quantiles, from the house's recorded reads of the page.

Method (the final-regime method: the practice's top entries hardcoded the
04:30 refresh rates, which are gone and cannot be recovered, so the final
is a genuine extrapolation and this is the model built for it):

1. Collapse the history to DISTINCT refresh states (drop consecutive
   identical reads -- Exophase only changes at its own refreshes).
2. DROP the launch-night flat state (pre-refresh, N=638) and anything
   before the first refresh. That cohort is reviewers, not the launch
   population: earner counts fall for late trophies between the launch
   read and the first refresh, which no cumulative set can do. Any fit
   that keeps it learns "late trophies are falling" and extrapolates
   them to ~0.
3. Per trophy, over the LAST 3 distinct states, fit
   logit(rate) = a + b * ln(1 + t) by least squares, t = hours since
   2026-11-19T00:00Z, with the 0.5-count continuity correction
   rate_adj = (E + 0.5) / (N + 1) * 100, E = rate * N / 100.
   Fewer than 3 distinct states -> fall back to persistence plus a
   random-walk spread (the house baseline's form).
4. Shrink the slope: beta = median b over trophies whose max rate is
   >= 1% in the used states; b_i <- 0.5 * b_i + 0.5 * beta; then
   re-centre a_i = mean(logit - b_i * x). Shrinkage is the robustness
   win: a two-point line amplifies state noise ~3.4x over a 4.5-day
   extrapolation, and pulling slopes halfway to the shared slope costs
   ~0.04 at zero noise and saves ~0.25 at high noise.
5. Median: read the shrunk line at the target.
6. Spread: s = sqrt((0.5 * sqrt(d))^2 + s_res^2) * 0.85, d = horizon
   in days, s_res = residual SD of the shrunk fit (floor 0.15).
   Quantiles q_tau = inv_logit(logit(med) + z_tau * s),
   z = (-1.645, -0.674, 0, 0.674, 1.645).
7. Low-signal: trophies whose max rate across the used states is
   < 0.10% -> median floored at 0.10%, spread floored at 0.8 (the fit
   would otherwise be fitting the 2-decimal rounding floor, not a trend).
8. Trophy #2 (100%): all quantiles 100. Clamp everything else to
   [0.01, 99.99].

Guard (from the same analysis): if the 3rd used state is more than 36h
old at the checkpoint, the window reaches back across a population
change, so use the last 2 states with stronger shrinkage (w = 0.3)
instead of the last 3 with w = 0.5.

Standard library only, no network, deterministic: two runs on the same
case write byte-identical output.
"""

import json
import math
import os
from datetime import datetime, timezone

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
LOW = 0.01
HIGH = 99.99


def clamp_pct(x):
    return min(HIGH, max(LOW, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l):
    if l >= 0.0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(dt):
    return (dt - RELEASE).total_seconds() / 3600.0


def median(xs):
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return ys[n // 2]
    return 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def distinct_states(history):
    """Collapse consecutive identical reads to one state each.

    Returns a list of (at_datetime, players_tracked, rates_tuple) in
    time order, one per run of identical reads. The state's time is
    its first read's time.
    """
    states = []
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if states and states[-1][2] == key:
            continue
        states.append((parse_time(read["at"]), read.get("players_tracked"),
                       tuple(rates)))
    return states


def first_refresh_index(states):
    """Index of the first distinct state that differs from the first.

    The launch-night flat state is states[0]; the first refresh is the
    first later state with different values. Returns None when the
    whole history is one flat state (no refresh seen).
    """
    if len(states) < 2:
        return None
    first = states[0][2]
    first_n = states[0][1]
    for i in range(1, len(states)):
        if states[i][1] != first_n or states[i][2] != first:
            return i
    return None


def ols_line(xs, ys):
    """Least-squares y = a + b*x. Returns (a, b). Degenerate -> flat."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    var = sum((x - mx) ** 2 for x in xs)
    if var <= 0.0:
        return my, 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = cov / var
    return my - b * mx, b


def rw_spread(d_days):
    """House baseline random-walk spread: 0.5 * sqrt(horizon in days)."""
    return 0.5 * math.sqrt(max(d_days, 0.0))


def persistence_quantiles(last_rate, d_days):
    """Fallback: median = last rate, house random-walk spread.

    A trophy never seen in any read (last_rate None) gets a low median
    with a wide spread -- never 100%.
    """
    if last_rate is None:
        s = rw_spread(d_days) + 1.0
        lg = logit(LOW)
        return [round(clamp_pct(inv_logit(lg + z * s)), 4) for z in Z]
    if last_rate >= 100.0:
        return [100.0, 100.0, 100.0, 100.0, 100.0]
    s = rw_spread(d_days)
    lg = logit(last_rate)
    out = []
    for z in Z:
        out.append(round(clamp_pct(inv_logit(lg + z * s)), 4))
    return out


def forecast_case(case):
    trophies = case["trophies"]
    n_tr = len(trophies)
    history = case.get("history") or []
    target = parse_time(case["target"])
    t_target = hours_since_release(target)
    d_days = case.get("horizon_minutes", 0) / 1440.0
    checkpoint = parse_time(case["at"])

    # Distinct states, then drop everything before the first refresh.
    states = distinct_states(history)
    ref = first_refresh_index(states)
    post = states[ref:] if ref is not None else []
    # Last-3 window, with the stale-window guard.
    if len(post) >= 3:
        used = post[-3:]
        shrink_w = 0.5
        if (checkpoint - used[0][0]).total_seconds() > 36 * 3600:
            used = post[-2:]
            shrink_w = 0.3
    else:
        used = post
        shrink_w = 0.5

    # Last rate per trophy (for the fallback and the 100% rule).
    last_rates = []
    for i in range(n_tr):
        last = None
        for _, _, rates in reversed(states):
            if i < len(rates) and rates[i] is not None:
                last = rates[i]
                break
        last_rates.append(last)

    if len(used) < 3:
        rows = [persistence_quantiles(last_rates[i], d_days)
                for i in range(n_tr)]
        return {"trophies": [{"id": trophies[i]["id"], "q": rows[i]}
                             for i in range(n_tr)]}

    # Per-trophy fits over the used states.
    xs = [math.log1p(hours_since_release(at)) for at, _, _ in used]
    fits = []  # (a, b, s_res, max_rate) per trophy, None when unusable
    for i in range(n_tr):
        ys = []
        ok_x = []
        max_rate = 0.0
        for (at, n_players, rates), x in zip(used, xs):
            if i >= len(rates) or rates[i] is None:
                continue
            r = rates[i]
            if r > max_rate:
                max_rate = r
            e = r * n_players / 100.0
            adj = (e + 0.5) / (n_players + 1.0) * 100.0
            ys.append(logit(adj))
            ok_x.append(x)
        if len(ys) < 2:
            fits.append(None)
            continue
        a, b = ols_line(ok_x, ys)
        fits.append((a, b, ok_x, ys, max_rate))

    # Shared slope: median b over trophies with max rate >= 1%.
    slopes = [f[1] for f in fits if f is not None and f[4] >= 1.0]
    beta = median(slopes) if slopes else 0.0

    x_target = math.log1p(t_target)
    rows = []
    for i in range(n_tr):
        f = fits[i]
        if f is None:
            rows.append(persistence_quantiles(last_rates[i], d_days))
            continue
        a, b, ok_x, ys, max_rate = f
        # Shrink the slope toward the shared slope, then re-centre.
        b2 = shrink_w * b + (1.0 - shrink_w) * beta
        a2 = sum(y - b2 * x for x, y in zip(ok_x, ys)) / len(ys)
        # Residual SD of the shrunk fit (floor 0.15).
        res = [y - (a2 + b2 * x) for x, y in zip(ok_x, ys)]
        s_res = math.sqrt(sum(r * r for r in res) / len(res))
        if s_res < 0.15:
            s_res = 0.15
        med = inv_logit(a2 + b2 * x_target)
        s = math.sqrt(rw_spread(d_days) ** 2 + s_res ** 2) * 0.85
        # Low-signal: floor the median and the spread.
        if max_rate < 0.10:
            if med < 0.10:
                med = 0.10
            if s < 0.8:
                s = 0.8
        lg = logit(med)
        qs = [round(clamp_pct(inv_logit(lg + z * s)), 4) for z in Z]
        rows.append(qs)

    # Trophy #2 (100%): every quantile 100.
    for i in range(n_tr):
        if last_rates[i] is not None and last_rates[i] >= 100.0:
            rows[i] = [100.0, 100.0, 100.0, 100.0, 100.0]

    out = {"trophies": [{"id": trophies[i]["id"], "q": rows[i]}
                         for i in range(n_tr)]}
    return out


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(fn for fn in os.listdir(in_dir) if fn.endswith(".in"))
    for fn in names:
        with open(os.path.join(in_dir, fn), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = fn[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w",
                  encoding="utf-8") as fh:
            json.dump(result, fh)
            fh.write("\n")


if __name__ == "__main__":
    main()
