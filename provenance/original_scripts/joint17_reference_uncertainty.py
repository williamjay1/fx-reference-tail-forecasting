"""Date-clustered uncertainty of reference-basis ES disagreement medians.

Descriptive historical reanalysis; no hypothesis-selection or model refitting.
All years and predefined tasks are retained, with pointwise percentile intervals.
"""
from pathlib import Path
import hashlib
import json
import time

import fx_windows_platform_compat  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(r"D:\MLWork\FXTailRisk")
BASE = ROOT / "results/joint_fx_20261002"
SOURCE = BASE / "analysis/reference_sensitivity_daily.csv"
OUT = BASE / "reference_uncertainty"
MODELS = ["FHS1250", "IN_FHS1250", "IN_TCOP1250"]
TASKS = [(tau, book, report) for tau in [.95, .99]
         for book in ["funded_assets", "net_cashflows"] for report in ["EUR", "USD"]]
BLOCKS = [20, 60]
B = 5000
SEED = 20261002


def circular_medians(x, block, draws=B, seed=SEED):
    """Sample common origin blocks for all task columns, truncate to original n."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Finite date-by-task matrix required")
    n, p = x.shape
    if n < block:
        raise ValueError("Insufficient dates for requested block length")
    blocks = (n + block - 1) // block
    offsets = np.arange(block)
    rng = np.random.default_rng(seed)
    result = np.empty((draws, p))
    for first in range(0, draws, 100):
        batch = min(100, draws - first)
        starts = rng.integers(0, n, size=(batch, blocks))
        indices = ((starts[..., None] + offsets) % n).reshape(batch, -1)[:, :n]
        # Every currency/report/quantile column uses these SAME date indices.
        result[first:first+batch] = np.median(x[indices], axis=1)
    return result


def self_check():
    x = np.arange(1.0, 40.0)[:, None]
    paired = np.column_stack([x[:, 0], 3*x[:, 0]+2])
    block, draws, seed = 7, 101, 197
    actual = circular_medians(paired, block, draws, seed)
    rng = np.random.default_rng(seed)
    expected=[]
    for _ in range(draws):
        starts = rng.integers(0,len(x),size=(len(x)+block-1)//block)
        ids=[(int(start)+offset)%len(x) for start in starts for offset in range(block)][:len(x)]
        expected.append(np.median(paired[ids],axis=0))
    error = float(np.max(np.abs(actual-np.asarray(expected))))
    coupled_error = float(np.max(np.abs(actual[:,1]-(3*actual[:,0]+2))))
    if error != 0 or coupled_error != 0:
        raise AssertionError("Circular median or shared-date invariant failure")
    return {"explicit_index_max_error":error,"date_coupling_max_error":coupled_error,
            "note":"Synthetic arithmetic checks, not empirical risk evidence"}


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing outputs; new run needs a new directory: {OUT}")
    start = time.perf_counter()
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    d = pd.read_csv(SOURCE, float_precision="round_trip", parse_dates=["origin_date"])
    d = d[(d.metric=="es") & d.model.isin(MODELS)].copy()
    if not np.isfinite(d.range_over_EUR).all() or not d.range_over_EUR.ge(0).all():
        raise ValueError("Reference ES relative spans must be finite and nonnegative")
    if d.duplicated(["origin_date","model","tau","book","report"]).any():
        raise ValueError("Duplicate reference sensitivity keys")
    d['year'] = d.origin_date.dt.year
    if set(d.year) != set(range(2018,2026)):
        raise ValueError("All formal years required")
    expected_tasks = set(TASKS)
    for (model,year), group in d.groupby(['model','year']):
        tasks = set(zip(group.tau,group.book,group.report))
        if tasks != expected_tasks or not group.groupby('origin_date').size().eq(8).all():
            raise ValueError(f"Unequal task/date support for {model}, {year}")
    annual = d.groupby(['model','tau','book','report','year']).agg(
        n_dates=('origin_date','size'), median_relative_ES_span=('range_over_EUR','median'),
        median_absolute_ES_span=('absolute_range','median'),
        median_max_over_min_ES=('max_over_min','median')).reset_index()
    if len(annual)!=192:
        raise ValueError("Expected 3 models x 2 quantiles x 4 tasks x 8 years")
    wide = d[d.model=="IN_FHS1250"].pivot(index='origin_date',columns=['tau','book','report'],values='range_over_EUR')
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(TASKS,names=['tau','book','report'])).sort_index()
    if len(wide)!=2046 or wide.isna().any().any():
        raise ValueError("Expected complete 2,046-date internally normalized FHS matrix")
    ci=[]
    for period in ['2018–2025',*range(2018,2026)]:
        selected = wide if isinstance(period,str) else wide[wide.index.year==period]
        values=selected.to_numpy()
        median=np.median(values,axis=0)
        for block in BLOCKS:
            bootstrap=circular_medians(values,block)
            low,high=np.quantile(bootstrap,[.025,.975],axis=0)
            for j,(tau,book,report) in enumerate(TASKS):
                ci.append(dict(model='IN_FHS1250',period=period,tau=tau,book=book,report=report,
                    n_dates=len(selected),block=block,draws=B,seed=SEED,
                    median_relative_ES_span=float(median[j]),percentile_low=float(low[j]),
                    percentile_high=float(high[j])))
    frame=pd.DataFrame(ci)
    support=annual[annual.year.between(2021,2025)].groupby(['model','tau','book','report']).agg(
        minimum_annual_median_relative_ES_span=('median_relative_ES_span','min'),
        maximum_annual_median_relative_ES_span=('median_relative_ES_span','max'),
        years_retained=('year','size')).reset_index()
    checks=self_check()
    OUT.mkdir(parents=True)
    annual.to_csv(OUT/'annual_ES_reference_medians.csv',index=False)
    frame.to_csv(OUT/'IN_FHS_date_block_median_intervals.csv',index=False)
    support.to_csv(OUT/'persistence_2021_2025.csv',index=False)
    manifest={'status':'completed','source':str(SOURCE),'source_sha256':source_sha,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'annual_rows':len(annual),'confidence_interval_rows':len(frame),
        'models':MODELS,'metric':'ES (max across five training references minus min) / EUR-reference ES',
        'all_years_retained':list(range(2018,2026)),'n_full_period_dates':len(wide),
        'blocks':BLOCKS,'bootstrap_draws':B,'seed':SEED,'self_check':checks,
        'runtime_seconds':time.perf_counter()-start,
        'interpretation':'Pointwise 95% percentile intervals for descriptive medians; shared origin indices across eight task/quantile columns within each period; algorithms not refitted; historical reanalysis, not confirmatory or a test of forecast superiority.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps(manifest,indent=2),flush=True)
    print(frame[frame.period=='2018–2025'].to_string(index=False),flush=True)
    print(support.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
