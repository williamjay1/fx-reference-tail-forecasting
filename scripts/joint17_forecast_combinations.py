"""Prior-only Taylor-style VaR/ES combinations for the joint FX redesign.

Adaptations, not replications of the original equity candidate pools. Each
cash book/report/quantile has its own annual, completed-label weights. No
combination in this module claims to supply a complete joint FX scenario law.
"""
from __future__ import annotations
from project_config import ROOT as PUBLIC_WORK_ROOT

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import fx_windows_platform_compat  # noqa: F401
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

PROJECT = PUBLIC_WORK_ROOT
CANDIDATES = ("FHS250", "FHS500", "FHS1250", "TCOP250", "TCOP500", "TCOP1250")
RIDGE_GRID = (0.0, 0.001, 0.01, 0.1, 1.0)
REFERENCES = ("EUR", "USD", "GBP", "JPY", "CHF")
RELIABLE_SELECTORS = (
    ("HS1250", "ALL_EQUIVARIANT"),
    ("EWMA_GAUSS", "ALL_EQUIVARIANT"),
    ("EWMA_SELFNORMALIZED", "ALL_EQUIVARIANT"),
    ("IN_FHS1250", "EUR"),
    ("IN_TCOP1250", "EUR"),
    ("GARCH_CASH_T_2STEP", "not_applicable"),
)


def fz0(y, q, e, tau):
    y, q, e = np.broadcast_arrays(y, q, e)
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(q)):
        raise ValueError("Nonfinite loss or VaR")
    if not np.all(np.isfinite(e)) or np.any(e <= 0) or np.any(e < q - 1e-12):
        raise ValueError("FZ0 requires finite e>0 and e>=q")
    return np.maximum(y - q, 0.0) / ((1.0 - tau) * e) + q / e + np.log(e) - 1.0


def simplex_forecast(v, e, a, b):
    q = v @ a
    return q, q + (e - v) @ b


def objective_gradient(weights, y, v, d, tau, ridge=(0.0, 0.0)):
    m = v.shape[1]
    a, b = weights[:m], weights[m:]
    q = v @ a
    e = q + d @ b
    if np.any(e <= 0):
        return 1e30, np.zeros(2 * m)
    hit = (y > q).astype(float)
    excess = np.maximum(y - q, 0.0)
    alpha = 1.0 - tau
    aa = excess / alpha + q
    dq = (1.0 - hit / alpha) / e
    de = 1.0 / e - aa / (e * e)
    val = np.mean(aa / e + np.log(e) - 1.0)
    grad = np.r_[np.mean(v * (dq + de)[:, None], axis=0),
                 np.mean(d * de[:, None], axis=0)]
    val += ridge[0] * np.dot(a, a) + ridge[1] * np.dot(b, b)
    grad += np.r_[2.0 * ridge[0] * a, 2.0 * ridge[1] * b]
    return float(val), grad


def optimize_simplex(y, v, e, tau, ridge=(0.0, 0.0), starts=None):
    m = v.shape[1]
    d = e - v
    if np.any(d < -1e-12) or np.any(v <= 0):
        raise ValueError("This comparator specification requires component q>0, e>=q")
    equal = np.full(2 * m, 1.0 / m)
    best = int(np.argmin(np.mean(fz0(y[:, None], v, e, tau), axis=0)))
    vertex = np.zeros(2 * m)
    vertex[best] = vertex[m + best] = 1.0
    initial = [equal, vertex] if starts is None else [*starts, equal, vertex]
    jac_eq = np.zeros((2, 2 * m))
    jac_eq[0, :m] = 1.0
    jac_eq[1, m:] = 1.0
    constraints = {"type": "eq", "fun": lambda z: np.array([z[:m].sum() - 1.0, z[m:].sum() - 1.0]),
                   "jac": lambda z: jac_eq}
    records = []
    feasible = []
    for z0 in initial:
        val0, _ = objective_gradient(z0, y, v, d, tau, ridge)
        feasible.append((val0, z0.copy(), False, "feasible_start"))
        result = minimize(objective_gradient, z0, args=(y, v, d, tau, ridge),
                          method="SLSQP", jac=True, bounds=[(0.0, 1.0)] * (2 * m),
                          constraints=constraints,
                          options={"maxiter": 1000, "ftol": 1e-10, "disp": False})
        z = np.maximum(np.asarray(result.x), 0.0)
        z[:m] /= z[:m].sum()
        z[m:] /= z[m:].sum()
        val, _ = objective_gradient(z, y, v, d, tau, ridge)
        records.append({"success": bool(result.success), "message": str(result.message),
                        "iterations": int(result.nit), "objective": val})
        if np.isfinite(val):
            feasible.append((val, z, bool(result.success), str(result.message)))
    chosen = min(feasible, key=lambda x: x[0])
    return chosen[1][:m], chosen[1][m:], {"objective": chosen[0],
        "selected_optimizer_success": chosen[2], "selected_message": chosen[3],
        "starts": records, "ridge_var": ridge[0], "ridge_spacing": ridge[1]}


def optimize_relative(y, v, e, tau):
    # Relative weights use SUMS of the historical component scores. Eta is a
    # numerically stable rescaling lambda*max(sumS-min(sumS)), not a new rule.
    sums = np.sum(fz0(y[:, None], v, e, tau), axis=0)
    shifted = sums - np.min(sums)
    scale = float(np.max(shifted))
    if scale < 1e-12:
        w = np.full(v.shape[1], 1.0 / v.shape[1])
        return w, {"lambda": 0.0, "eta": 0.0, "objective": float(np.mean(fz0(y, v @ w, e @ w, tau))),
                   "score_sum_scale": scale, "boundary": "all_component_score_sums_equal"}
    delta = shifted / scale

    def weights(eta):
        raw = np.exp(-eta * delta)
        return raw / raw.sum()

    def loss(eta):
        w = weights(eta)
        return float(np.mean(fz0(y, v @ w, e @ w, tau)))

    grid = np.r_[0.0, np.geomspace(1e-6, 1000.0, 100)]
    vals = np.array([loss(x) for x in grid])
    candidates = [(float(vals[i]), float(grid[i]), weights(grid[i]), "grid") for i in range(len(grid))]
    # Refine every sampled local minimum, rather than presuming a unimodal
    # score objective in lambda. Include the exact infinite-lambda limit.
    for i in range(1, len(grid) - 1):
        if vals[i] <= vals[i - 1] and vals[i] <= vals[i + 1]:
            r = minimize_scalar(loss, bounds=(grid[i - 1], grid[i + 1]), method="bounded",
                                options={"xatol": 1e-10})
            candidates.append((float(r.fun), float(r.x), weights(r.x), "local_refinement"))
    win = np.isclose(shifted, 0.0, rtol=0.0, atol=1e-12).astype(float)
    win /= win.sum()
    candidates.append((float(np.mean(fz0(y, v @ win, e @ win, tau))), math.inf, win, "infinite_lambda_limit"))
    value, eta, w, where = min(candidates, key=lambda x: x[0])
    return w, {"lambda": None if not math.isfinite(eta) else eta / scale,
               "eta": None if not math.isfinite(eta) else eta, "objective": value,
               "score_sum_scale": scale, "boundary": where}


def optimize_regularized(y, v, e, tau):
    split = int(math.floor(0.75 * len(y)))
    if split < 50 or len(y) - split < 20:
        raise ValueError("Insufficient chronological ridge validation support")
    trials = []
    for lam1 in RIDGE_GRID:
        for lam2 in RIDGE_GRID:
            a, b, info = optimize_simplex(y[:split], v[:split], e[:split], tau, (lam1, lam2))
            q_valid, e_valid = simplex_forecast(v[split:], e[split:], a, b)
            val = float(np.mean(fz0(y[split:], q_valid, e_valid, tau)))
            trials.append({"ridge_var": lam1, "ridge_spacing": lam2,
                           "validation_score": val, "training_optimizer_success": info["selected_optimizer_success"]})
    # Fixed tie convention uses the listed low-to-high penalty grid order.
    chosen = min(trials, key=lambda row: row["validation_score"])
    ridge = (chosen["ridge_var"], chosen["ridge_spacing"])
    a, b, info = optimize_simplex(y, v, e, tau, ridge)
    info.update({"chronological_training_rows": split, "chronological_validation_rows": len(y) - split,
                 "selected_validation_score": chosen["validation_score"], "penalty_trials": trials})
    return a, b, info


def fit_combinations(y, v, e, tau, candidates=CANDIDATES):
    a, b, min_info = optimize_simplex(y, v, e, tau)
    w, relative_info = optimize_relative(y, v, e, tau)
    ra, rb, ridge_info = optimize_regularized(y, v, e, tau)
    mean = np.full(v.shape[1], 1.0 / v.shape[1])
    winner = np.zeros(v.shape[1])
    winner[int(np.argmin(np.mean(fz0(y[:, None], v, e, tau), axis=0)))] = 1.0
    return {
        "TAYLOR_OPT": (a, b, min_info),
        "TW_RELATIVE": (w, w, relative_info),
        "TW_MIN_RIDGE": (ra, rb, ridge_info),
        "FUNCTIONAL_EQUAL": (mean, mean, {"fitted_parameters": 0}),
        "PAST_BEST": (winner, winner, {"fitted_parameters": 0, "selected_candidate": candidates[int(np.argmax(winner))]}),
    }


def prepare_component_tasks(frame, selectors=None):
    needed = {"origin_date", "start_date", "target_date", "book", "report", "tau", "model",
              "train_reference", "actual_loss", "var", "es"}
    missing = needed.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing component columns: {sorted(missing)}")
    selectors = tuple((name, "EUR") for name in CANDIDATES) if selectors is None else tuple(selectors)
    candidates = tuple(name for name, reference in selectors)
    selected = np.zeros(len(frame), dtype=bool)
    for model, reference in selectors:
        selected |= (frame.model == model) & (frame.train_reference == reference)
    f = frame[selected].copy()
    if set(f.model) != set(candidates):
        raise ValueError(f"Missing fixed candidate models: {set(candidates).difference(f.model)}")
    for col in ("origin_date", "start_date", "target_date"):
        f[col] = pd.to_datetime(f[col])
    if f.duplicated(["origin_date", "book", "report", "tau", "model"]).any():
        raise ValueError("Duplicate canonical component prediction keys")
    tasks = []
    for (book, report, tau), group in f.groupby(["book", "report", "tau"], sort=True):
        counts = group.groupby("origin_date").model.nunique()
        if np.any(counts != len(candidates)):
            raise ValueError(f"Incomplete six-model support for {(book, report, tau)}")
        for col in ("start_date", "target_date"):
            if np.any(group.groupby("origin_date")[col].nunique() != 1):
                raise ValueError(f"Component disagreement in {col}")
        # Different explicitly manifested CSV serializers can differ by one
        # ulp on the same normalized source outcome; clock keys remain exact.
        outcome_ranges = group.groupby("origin_date").actual_loss.agg(["min", "max"])
        if np.any((outcome_ranges["max"] - outcome_ranges["min"]) > 1e-13):
            raise ValueError("Component disagreement in actual_loss beyond CSV rounding")
        meta = group.groupby("origin_date", sort=True).first().reset_index()
        v = group.pivot(index="origin_date", columns="model", values="var").reindex(columns=candidates).to_numpy()
        e = group.pivot(index="origin_date", columns="model", values="es").reindex(columns=candidates).to_numpy()
        y = meta.actual_loss.to_numpy()
        fz0(y[:, None], v, e, float(tau))
        tasks.append(((book, report, float(tau)), meta, v, e))
    return tasks


def run_combinations(frame, start_year=2018, end_year=2025, history=900, selectors=None):
    rows, fits, skipped = [], [], []
    candidates = CANDIDATES if selectors is None else tuple(model for model, reference in selectors)
    for (book, report, tau), meta, v, e in prepare_component_tasks(frame, selectors):
        dates = meta.origin_date.to_numpy()
        targets = meta.target_date.to_numpy()
        for year in range(start_year, end_year + 1):
            ids = np.flatnonzero(meta.origin_date.dt.year.to_numpy() == year)
            if not len(ids):
                continue
            prior_year_ids = np.flatnonzero(meta.origin_date.dt.year.to_numpy() < year)
            if not len(prior_year_ids):
                skipped.append({"year": year, "book": book, "report": report, "tau": tau,
                                "available_completed_predictions": 0, "required": history,
                                "reason": "No prior-year component prediction origin"})
                continue
            cutoff = dates[prior_year_ids[-1]]
            available = np.flatnonzero((targets <= cutoff) & (dates < cutoff))
            if len(available) < history:
                skipped.append({"year": year, "book": book, "report": report, "tau": tau,
                                "available_completed_predictions": len(available), "required": history})
                continue
            train = available[-history:]
            y_train = meta.actual_loss.to_numpy()[train]
            systems = fit_combinations(y_train, v[train], e[train], tau, candidates)
            for model, (a, b, info) in systems.items():
                q, ee = simplex_forecast(v[ids], e[ids], a, b)
                yy = meta.actual_loss.to_numpy()[ids]
                scores = fz0(yy, q, ee, tau)
                fits.append({"fit_year": year, "book": book, "report": report, "tau": tau, "model": model,
                    "cutoff": str(pd.Timestamp(cutoff).date()), "training_rows": len(train),
                    "training_first_origin": str(meta.iloc[train[0]].origin_date.date()),
                    "training_last_origin": str(meta.iloc[train[-1]].origin_date.date()),
                    "training_last_endpoint": str(meta.iloc[train[-1]].target_date.date()),
                    "var_weights": a.tolist(), "spacing_weights": b.tolist(), "fit_info": info})
                for k, index in enumerate(ids):
                    source = meta.iloc[index]
                    rows.append({"origin_date": str(source.origin_date.date()),
                        "start_date": str(source.start_date.date()), "target_date": str(source.target_date.date()),
                        "book": book, "report": report, "tau": tau, "model": model, "train_reference": "not_applicable",
                        "actual_loss": float(yy[k]), "var": float(q[k]), "es": float(ee[k]), "score": float(scores[k]),
                        "strict_hit": int(yy[k] > q[k]), "fit_year": year})
    return pd.DataFrame(rows), fits, skipped


def run_reference_selection(frame, start_year=2018, end_year=2025, history=900, families=("FHS1250", "TCOP1250")):
    """One global prior-q95 reference choice per family, reused by all tasks.

    The selected scenario member supplies one coherent law for every book,
    report and BOTH quantiles; q99 outcomes never select its reference.
    """
    f = frame[frame.model.isin(families) & frame.train_reference.isin(REFERENCES)].copy()
    for col in ("origin_date", "start_date", "target_date"):
        f[col] = pd.to_datetime(f[col])
    if f.duplicated(["origin_date", "book", "report", "tau", "model", "train_reference"]).any():
        raise ValueError("Duplicate full-reference scenario keys")
    observed_tasks = set(zip(f.book, f.report))
    expected_tasks = {(book, report) for book in ("funded_assets", "net_cashflows") for report in ("EUR", "USD")}
    if observed_tasks != expected_tasks:
        raise ValueError(f"Reference selection expects exactly four common cash tasks: {observed_tasks}")
    predictions, choices, skipped = [], [], []
    for family in families:
        member = f[f.model == family]
        q95 = member[np.isclose(member.tau, 0.95)]
        counts = q95.groupby(["origin_date", "train_reference"]).size()
        if np.any(counts != 4) or q95.groupby("origin_date").size().ne(20).any():
            raise ValueError(f"Incomplete q95 reference/task support for {family}")
        for col in ("start_date", "target_date"):
            if q95.groupby("origin_date")[col].nunique().ne(1).any():
                raise ValueError(f"Reference clocks disagree in {col}")
        recomputed = fz0(q95.actual_loss.to_numpy(), q95["var"].to_numpy(), q95.es.to_numpy(), 0.95)
        if not np.allclose(recomputed, q95.score.to_numpy(), rtol=1e-12, atol=1e-12):
            raise ValueError("Source q95 reference scores disagree with FZ0")
        performance = q95.groupby(["origin_date", "train_reference"]).score.mean().unstack().reindex(columns=REFERENCES)
        metadata = q95.groupby("origin_date", sort=True).first()
        dates = performance.index.to_numpy()
        targets = metadata.target_date.to_numpy()
        for year in range(start_year, end_year + 1):
            current_ids = np.flatnonzero(performance.index.year == year)
            previous_ids = np.flatnonzero(performance.index.year < year)
            if not len(current_ids):
                continue
            if not len(previous_ids):
                skipped.append({"model": family, "year": year, "available_complete_dates": 0, "required": history})
                continue
            cutoff = dates[previous_ids[-1]]
            available = np.flatnonzero((targets <= cutoff) & (dates < cutoff))
            if len(available) < history:
                skipped.append({"model": family, "year": year, "available_complete_dates": len(available), "required": history})
                continue
            train = available[-history:]
            losses = performance.to_numpy()[train].mean(axis=0)
            if not np.isfinite(losses).all():
                raise ValueError("Incomplete prior reference score matrix")
            winner = int(np.argmin(losses))  # deterministic EUR/USD/GBP/JPY/CHF tie order
            reference = REFERENCES[winner]
            weights = [int(i == winner) for i in range(len(REFERENCES))]
            model = family.replace("1250", "_REF_PAST_BEST")
            choices.append({"model": model, "fit_year": year, "cutoff": str(pd.Timestamp(cutoff).date()),
                "selection_quantile": 0.95, "selection_tasks": sorted(expected_tasks), "selection_dates": len(train),
                "training_first_origin": str(pd.Timestamp(dates[train[0]]).date()),
                "training_last_origin": str(pd.Timestamp(dates[train[-1]]).date()),
                "training_last_endpoint": str(pd.Timestamp(targets[train[-1]]).date()),
                "references": REFERENCES, "prior_date_and_task_mean_scores": losses.tolist(),
                "selected_reference": reference, "weights": weights,
                "reuse": "same scenario member for all four tasks and q95/q99"})
            selected = member[(member.origin_date.dt.year == year) & (member.train_reference == reference)].copy()
            if selected.groupby("origin_date").size().ne(8).any():
                raise ValueError("Selected reference does not supply all four tasks/two quantiles")
            for _, row in selected.iterrows():
                tau = float(row.tau)
                yy, q, ee = float(row.actual_loss), float(row["var"]), float(row.es)
                predictions.append({"origin_date": str(row.origin_date.date()), "start_date": str(row.start_date.date()),
                    "target_date": str(row.target_date.date()), "book": row.book, "report": row.report,
                    "tau": tau, "model": model, "train_reference": reference, "actual_loss": yy,
                    "var": q, "es": ee, "score": float(fz0(yy, q, ee, tau)), "strict_hit": int(yy > q),
                    "fit_year": year})
    return pd.DataFrame(predictions), choices, skipped


def self_check():
    rng = np.random.default_rng(2026100209)
    n, m, tau = 150, 6, 0.95
    y = rng.normal(0, 0.01, n)
    v = rng.uniform(0.012, 0.025, (n, m))
    e = v + rng.uniform(0.003, 0.02, (n, m))
    z = np.r_[rng.dirichlet(np.ones(m)), rng.dirichlet(np.ones(m))]
    value, grad = objective_gradient(z, y, v, e - v, tau, (0.01, 0.1))
    numeric = np.empty(2 * m)
    eps = 1e-6
    for k in range(2 * m):
        plus, minus = z.copy(), z.copy()
        plus[k] += eps
        minus[k] -= eps
        numeric[k] = (objective_gradient(plus, y, v, e - v, tau, (0.01, 0.1))[0] -
                      objective_gradient(minus, y, v, e - v, tau, (0.01, 0.1))[0]) / (2 * eps)
    error = float(np.max(np.abs(grad - numeric)))
    if error > 1e-6:
        raise AssertionError(f"Combination gradient error {error}")
    a, b, info = optimize_simplex(y, v, e, tau)
    q, ee = simplex_forecast(v, e, a, b)
    equal = np.mean(fz0(y, v.mean(axis=1), e.mean(axis=1), tau))
    optimized = np.mean(fz0(y, q, ee, tau))
    if optimized > equal + 1e-9:
        raise AssertionError("Optimizer worse than its feasible equal-weight start")
    shifted = fz0(y * 100, q * 100, ee * 100, tau) - fz0(y, q, ee, tau)
    if not np.allclose(shifted, np.log(100), atol=1e-12, rtol=0):
        raise AssertionError("FZ0 deterministic scale identity failure")
    return {"gradient_max_error": error, "simplex_var_sum": float(a.sum()),
            "simplex_spacing_sum": float(b.sum()), "optimized_score": float(optimized),
            "equal_score": float(equal), "optimizer": info,
            "note": "Synthetic formula checks, not empirical risk evidence"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROJECT / "results/joint_fx_20261002/scenario_v2/predictions.csv")
    parser.add_argument("--out", type=Path, default=PROJECT / "results/joint_fx_20261002/combination/full")
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--history", type=int, default=900)
    parser.add_argument("--pool", choices=("original", "reliable"), default="original")
    parser.add_argument("--additional-input", type=Path, action="append", default=[])
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), indent=2))
        return
    if args.out.exists():
        raise FileExistsError(f"New run requires a new output directory: {args.out}")
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    raw_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    inputs = [args.input, *args.additional_input]
    input_manifest = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in inputs]
    # Preserve the written binary64 values. Default parser rounding can move
    # stress-date scores by several ulps after division by a small cash ES.
    frame = pd.concat([pd.read_csv(path, float_precision="round_trip") for path in inputs], ignore_index=True)
    selectors = RELIABLE_SELECTORS if args.pool == "reliable" else None
    candidates = CANDIDATES if selectors is None else tuple(model for model, reference in selectors)
    prediction, fits, skipped = run_combinations(frame, args.start_year, args.end_year, args.history, selectors)
    families = ("IN_FHS1250", "IN_TCOP1250") if args.pool == "reliable" else ("FHS1250", "TCOP1250")
    ref_prediction, ref_choices, ref_skipped = run_reference_selection(frame, args.start_year, args.end_year, args.history, families)
    prediction = pd.concat([prediction, ref_prediction], ignore_index=True)
    prediction.to_csv(args.out / "predictions.csv", index=False)
    (args.out / "annual_weights.json").write_text(json.dumps(fits, indent=2), encoding="utf-8")
    (args.out / "annual_reference_choices.json").write_text(json.dumps(ref_choices, indent=2), encoding="utf-8")
    if len(prediction):
        summary = prediction.groupby(["model", "tau", "book", "report"]).agg(
            rows=("score", "size"), mean_score=("score", "mean"), strict_hits=("strict_hit", "sum"),
            mean_var=("var", "mean"), mean_es=("es", "mean")).reset_index()
        summary.to_csv(args.out / "summary.csv", index=False)
    manifest = {"input": str(args.input), "input_sha256": raw_hash, "all_inputs": input_manifest,
                "candidate_models": candidates, "pool": args.pool, "selectors": selectors,
                "start_year": args.start_year, "end_year": args.end_year, "prior_completed_history": args.history,
                "annual_fit_records": len(fits), "prediction_rows": len(prediction), "skipped": skipped,
                "reference_selection_records": len(ref_choices), "reference_skipped": ref_skipped,
                "runtime_seconds": time.perf_counter() - started, "formula_self_check": self_check(),
                "script_sha256_at_start": script_sha, "annual_fit_cutoff": "last common reference origin of previous year",
                "references": {"Taylor2020": "doi:10.1016/j.ijforecast.2019.05.014",
                    "TaylorWang2026": "https://arxiv.org/html/2508.16919v2"},
                "scope": "Annual prior-only adaptations to a fixed six-candidate FX pool; functional forecasts, not joint FX laws; no exact original-pool replication"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("prediction_rows", "annual_fit_records", "runtime_seconds", "skipped")}, indent=2))


if __name__ == "__main__":
    main()
