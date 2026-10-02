"""Cash-loss GARCH and HS comparators with the future-start label clock.

At cutoff origin t, the most recent observable outcome has origin t-2.
The scalar GARCH forecast therefore advances two outcome-index steps. The
first unobserved cash-loss innovation is integrated by scrambled Sobol draws;
the second Student-t shock is integrated analytically. Student-t is applied
directly to CASH LOSS, never exponentiated into exchange-rate prices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import fx_windows_platform_compat  # noqa: F401
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import stdtr, stdtrit, gammaln
from scipy.stats import qmc

from joint17_models import fit_garch, variance_path, fz0, risk_pair

PROJECT = Path(r"D:\MLWork\FXTailRisk")


def stable_seed(book, report, year, offset=0):
    digest = hashlib.sha256(f"joint17_direct|{book}|{report}|{year}".encode()).digest()
    return (int.from_bytes(digest[:4], "little") + 2026100210 + offset) % (2 ** 32)


def first_step_shocks(nu, power, seed):
    u = qmc.Sobol(d=1, scramble=True, seed=seed).random_base2(power).ravel()
    # Sobol values are interior almost surely; guard machine endpoints before
    # inverse CDF rather than silently truncating actual scenario cash losses.
    if np.any((u <= 0) | (u >= 1)):
        raise ValueError("Sobol inverse-CDF endpoint")
    return stdtrit(nu, u) * math.sqrt((nu - 2.0) / nu)


def two_step_risk(fit, h_next, shocks, tau):
    """VaR/ES of a finite mixture of analytic conditional Student-t laws.

    h_next predicts y_{t-1} conditional on y_{t-2}. Simulated first shock
    generates h_two for y_t. This integrates BOTH missing outcome states.
    Output is normalized cash loss (fit uses percentage cash-loss units).
    """
    mu, nu = fit["mu"], fit["nu"]
    if nu <= 2 or h_next <= 0:
        raise ValueError("Invalid finite-variance cash Student-t GARCH state")
    h_two = fit["omega"] + fit["alpha"] * h_next * shocks ** 2 + fit["beta"] * h_next
    scales = np.sqrt(h_two) * math.sqrt((nu - 2.0) / nu)

    def cdf(value):
        return float(np.mean(stdtr(nu, (value - mu) / scales)))

    spread = max(float(np.sqrt(np.mean(h_two))), 1e-8)
    lower, upper = mu - spread, mu + spread
    for _ in range(100):
        if cdf(lower) <= tau:
            break
        lower = mu - 2.0 * (mu - lower)
    for _ in range(100):
        if cdf(upper) >= tau:
            break
        upper = mu + 2.0 * (upper - mu)
    if not cdf(lower) <= tau <= cdf(upper):
        raise ValueError("Cash-mixture quantile root not bracketed")
    q_percent = float(brentq(lambda x: cdf(x) - tau, lower, upper, xtol=1e-12, rtol=1e-12))
    z = (q_percent - mu) / scales
    logpdf = (gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi)
              - (nu + 1.0) / 2.0 * np.log1p(z * z / nu))
    first_moment = (nu + z * z) / (nu - 1.0) * np.exp(logpdf)
    tail = stdtr(nu, -z)
    excess = scales * first_moment - (q_percent - mu) * tail
    es_percent = q_percent + float(np.mean(excess)) / (1.0 - tau)
    q, e = q_percent / 100.0, es_percent / 100.0
    if not np.isfinite(q) or not np.isfinite(e) or e <= 0 or e < q:
        raise ValueError(f"Direct cash risk outside FZ0 domain q={q}, e={e}")
    return q, e


def prior_variance_states(y_percent, fit, initial_variance):
    """Filter from a prior-only annual initialization, never future variance."""
    if initial_variance <= 0:
        raise ValueError("Initialization must be a prior fitted variance")
    h = np.empty(len(y_percent))
    h[0] = initial_variance
    for i in range(1, len(h)):
        h[i] = max(fit["omega"] + fit["alpha"] * (y_percent[i - 1] - fit["mu"]) ** 2
                   + fit["beta"] * h[i - 1], 1e-10)
    return h


def run_direct(outcomes, start_year=2018, end_year=2025, power=12, max_origins=0):
    required = {"origin_date", "start_date", "target_date", "book", "report", "actual_loss"}
    if required.difference(outcomes.columns):
        raise ValueError(f"Missing outcome columns: {sorted(required.difference(outcomes.columns))}")
    f = outcomes.copy()
    for key in ("origin_date", "start_date", "target_date"):
        f[key] = pd.to_datetime(f[key])
    if f.duplicated(["origin_date", "book", "report"]).any():
        raise ValueError("Duplicate direct outcome keys")
    records, fits, skipped, precision = [], [], [], []
    for (book, report), group in f.groupby(["book", "report"], sort=True):
        g = group.sort_values("origin_date").reset_index(drop=True)
        origin = g.origin_date.to_numpy()
        target = g.target_date.to_numpy()
        y = g.actual_loss.to_numpy()
        if not np.isfinite(y).all():
            raise ValueError("Nonfinite scalar cash outcomes")
        for year in range(start_year, end_year + 1):
            ids = np.flatnonzero(g.origin_date.dt.year.to_numpy() == year)
            if not len(ids):
                continue
            # Shared annual parameter cutoff: last common reference date of
            # the previous year, matching the joint scenario model estimates.
            cutoff = origin[np.flatnonzero(g.origin_date.dt.year.to_numpy() < year)[-1]]
            prior = np.flatnonzero((target <= cutoff) & (origin < cutoff))
            if len(prior) < 1250:
                skipped.append({"year": year, "book": book, "report": report,
                                "completed_cash_labels": len(prior), "required": 1250})
                continue
            train = prior[-1250:]
            fit_start = int(train[0])
            y_train = y[train] * 100.0
            fit = fit_garch(y_train)
            initial_variance = max(float(np.var(y_train)), 1e-8)
            # Full future VALUES are present offline, but every h[j] recursion
            # uses only earlier scalar outcomes, with its initial variance
            # computed exclusively from the prior annual estimation labels.
            prefix_end = int(np.max(np.flatnonzero(target <= origin[ids[-1]])))
            filtered = prior_variance_states(y[fit_start:prefix_end + 1] * 100.0, fit, initial_variance)
            same = variance_path(y_train, fit["omega"], fit["alpha"], fit["beta"], fit["mu"])
            if not np.allclose(same, filtered[:len(train)], rtol=1e-13, atol=1e-13):
                raise AssertionError("Direct annual initialization disagrees with fit filter")
            seed = stable_seed(book, report, year)
            shocks = first_step_shocks(fit["nu"], power, seed)
            if max_origins and len(ids) > max_origins:
                ids = ids[np.unique(np.linspace(0, len(ids) - 1, max_origins, dtype=int))]
            fits.append({"fit_year": year, "book": book, "report": report,
                "cutoff": str(pd.Timestamp(cutoff).date()), "training_rows": len(train),
                "training_first_origin": str(g.iloc[train[0]].origin_date.date()),
                "training_last_origin": str(g.iloc[train[-1]].origin_date.date()),
                "training_last_endpoint": str(g.iloc[train[-1]].target_date.date()),
                "fit": fit, "initial_variance": initial_variance,
                "simulation_seed": seed, "first_step_draws": len(shocks),
                "second_step_integration": "analytic conditional Student-t tail"})
            precision_ids = {int(ids[0]), int(ids[len(ids)//2]), int(ids[-1])}
            shocks_precision = first_step_shocks(fit["nu"], power + 2, seed)
            for index in ids:
                available = np.flatnonzero((target <= origin[index]) & (origin < origin[index]))
                last = int(available[-1])
                # Common dates satisfy target(y_{t-2})=origin(t), exactly.
                if int(index) - last != 2 or target[last] != origin[index]:
                    raise ValueError(f"Two-step label-clock assumption failed at {g.iloc[index].origin_date}")
                h_last = filtered[last - fit_start]
                h_next = max(fit["omega"] + fit["alpha"] * (y[last] * 100.0 - fit["mu"]) ** 2
                             + fit["beta"] * h_last, 1e-10)
                for tau in (0.95, 0.99):
                    q, e = two_step_risk(fit, h_next, shocks, tau)
                    model_risks = [("GARCH_CASH_T_2STEP", q, e)]
                    for window in (500, 1250):
                        hq, he = risk_pair(y[available[-window:]], tau)
                        model_risks.append((f"HS_CASH{window}", hq, he))
                    for model, qv, ev in model_risks:
                        row = g.iloc[index]
                        records.append({"origin_date": str(row.origin_date.date()),
                            "start_date": str(row.start_date.date()), "target_date": str(row.target_date.date()),
                            "book": book, "report": report, "tau": tau, "model": model,
                            "train_reference": "not_applicable", "actual_loss": float(y[index]),
                            "var": float(qv), "es": float(ev), "score": float(fz0(y[index], qv, ev, tau)),
                            "strict_hit": int(y[index] > qv), "fit_year": year,
                            "latest_complete_origin": str(g.iloc[last].origin_date.date()),
                            "latest_complete_endpoint": str(g.iloc[last].target_date.date())})
                    if int(index) in precision_ids:
                        qhi, ehi = two_step_risk(fit, h_next, shocks_precision, tau)
                        precision.append({"origin_date": str(g.iloc[index].origin_date.date()), "book": book,
                            "report": report, "tau": tau, "base_draws": len(shocks),
                            "higher_draws": len(shocks_precision), "var_base": q, "es_base": e,
                            "var_higher": qhi, "es_higher": ehi, "var_difference": qhi - q,
                            "es_difference": ehi - e, "seed": seed,
                            "role": "Prespecified first/middle/last annual origins; numerical assessment, no configuration selection"})
    return pd.DataFrame(records), fits, skipped, pd.DataFrame(precision)


def self_check():
    from scipy.stats import t
    nu, sigma, mu = 7.0, 0.3, -0.01
    fit = {"mu": mu, "nu": nu, "omega": sigma ** 2, "alpha": 0.0, "beta": 0.0}
    shocks = first_step_shocks(nu, 10, 2026100210)
    errors = []
    for tau in (0.95, 0.99):
        q, e = two_step_risk(fit, sigma ** 2, shocks, tau)
        critical = t.ppf(tau, nu)
        scale = sigma * math.sqrt((nu - 2.0) / nu)
        qtrue = (mu + scale * critical) / 100.0
        etrue = (mu + scale * (nu + critical ** 2) / (nu - 1.0) * t.pdf(critical, nu) / (1.0 - tau)) / 100.0
        errors.append(max(abs(q - qtrue), abs(e - etrue)))
    if max(errors) > 1e-12:
        raise AssertionError(f"Conditional Student-t risk identity failure {errors}")
    # Two-step dispersion reacts to first simulated innovation when alpha>0.
    fit2 = {**fit, "omega": 0.01, "alpha": 0.1, "beta": 0.8}
    qzero, ezero = two_step_risk(fit2, sigma ** 2, np.zeros(1024), .99)
    qshock, eshock = two_step_risk(fit2, sigma ** 2, np.ones(1024) * 3, .99)
    if not qshock > qzero or not eshock > ezero:
        raise AssertionError("First future shock does not update second future variance")
    return {"analytic_t_qes_max_error": float(max(errors)), "two_step_variance_response_verified": True,
            "note": "Synthetic formula/state checks; not empirical prediction evidence"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROJECT / "results/joint_fx_20261002/data/outcome_tasks.csv")
    parser.add_argument("--out", type=Path, default=PROJECT / "results/joint_fx_20261002/direct/full")
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--power", type=int, default=12)
    parser.add_argument("--max-origins", type=int, default=0)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), indent=2))
        return
    if args.out.exists():
        raise FileExistsError(f"New run requires a new output directory: {args.out}")
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    input_sha = hashlib.sha256(args.input.read_bytes()).hexdigest()
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dependency_sha = hashlib.sha256((PROJECT / "scripts/joint17_models.py").read_bytes()).hexdigest()
    frame = pd.read_csv(args.input)
    prediction, fits, skipped, precision = run_direct(frame, args.start_year, args.end_year, args.power, args.max_origins)
    prediction.to_csv(args.out / "predictions.csv", index=False)
    precision.to_csv(args.out / "numerical_precision.csv", index=False)
    (args.out / "annual_fits.json").write_text(json.dumps(fits, indent=2), encoding="utf-8")
    if len(prediction):
        prediction.groupby(["model", "tau", "book", "report"]).agg(
            rows=("score", "size"), mean_score=("score", "mean"), strict_hits=("strict_hit", "sum"),
            mean_var=("var", "mean"), mean_es=("es", "mean")).reset_index().to_csv(args.out / "summary.csv", index=False)
    manifest = {"input": str(args.input), "input_sha256": input_sha, "start_year": args.start_year,
        "end_year": args.end_year, "cash_garch_annual_fit_records": len(fits), "prediction_rows": len(prediction),
        "skipped": skipped, "runtime_seconds": time.perf_counter() - started,
        "first_step_draws": 2 ** args.power, "label_clock": "last completed y origin t-2; forecast two scalar GARCH states",
        "training_window": 1250, "second_step": "analytic Student-t conditional tail integration",
        "scope": "Direct normalized cash-loss marginals; not a joint FX scenario law", "formula_checks": self_check(),
        "script_sha256_at_start": script_sha, "shared_model_sha256_at_start": dependency_sha,
        "annual_fit_cutoff": "last common reference origin of previous year"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("cash_garch_annual_fit_records", "prediction_rows", "runtime_seconds", "skipped")}, indent=2))


if __name__ == "__main__":
    main()
