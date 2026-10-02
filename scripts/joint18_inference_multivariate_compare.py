from project_config import ROOT as PUBLIC_WORK_ROOT
from project_config import recorded_script_path
"""Stored-forecast comparison with the coordinate-closed scalar BEKK control.

Owns only inference/multivariate_comparison. Earlier inference is preserved.
All contrasts share date blocks; no task is treated as an independent investor.
This is exploratory reanalysis, conditional on recorded forecast specifications.
"""
from pathlib import Path
import json, hashlib, time
import fx_windows_platform_compat
import numpy as np
import pandas as pd
import joint17_analyse as prior
from joint18_inference import circular_means, mcs_r, audit_mcs

ROOT=PUBLIC_WORK_ROOT
OUT=ROOT/'results/joint_fx_v18/inference/multivariate_comparison'
NEW=ROOT/'results/joint_fx_v18/multivariate/full16'
PP=ROOT/'results/joint_fx_v18/pit_precision/full96'
B=5000
SEED=20261002
PAIR=['IN_FHS_REF_POOL','SCALAR_BEKK_INTERNAL']
KEY=['origin_date','book','report','tau']

def write(name,df):
    df.to_csv(OUT/name,index=False,float_format='%.17g')

def fz(y,q,e,tau):
    return np.maximum(y-q,0)/((1-tau)*e)+q/e+np.log(e)-1

def load():
    _,ev,sources=prior.read_all()
    new=pd.read_csv(NEW/'predictions.csv',parse_dates=['origin_date','start_date','target_date','fit_cutoff_date'],float_precision='round_trip')
    new['model_key']=new.model
    if new.model.unique().tolist()!=[PAIR[1]]:raise ValueError('Unexpected BEKK model')
    if len(new)!=16368 or new.origin_date.nunique()!=2046:raise ValueError('Expected full common evaluation')
    if new.duplicated(KEY).any():raise ValueError('Duplicate BEKK keys')
    old=ev[ev.model_key==PAIR[0]].set_index(KEY).sort_index()
    n=new.set_index(KEY).sort_index()
    assert n.index.equals(old.index)
    assert old.start_date.equals(n.start_date) and old.target_date.equals(n.target_date)
    target_err=float(np.max(np.abs(old.actual_loss-n.actual_loss)))
    assert target_err<1e-12
    assert (new.fit_cutoff_date<new.origin_date).all()
    assert (new.scenario_n==65536).all()
    both=pd.concat([ev,new],ignore_index=True)
    recomputed=fz(new.actual_loss.to_numpy(),new['var'].to_numpy(),new.es.to_numpy(),new.tau.to_numpy())
    score_err=float(np.max(np.abs(new.score-recomputed)))
    assert score_err<1e-12
    assert (new.es>0).all() and (new.es>=new['var']-1e-12).all()
    assert np.array_equal(new.strict_hit.to_numpy(),(new.actual_loss>new['var']).astype(int).to_numpy())
    assert both.groupby(['origin_date','tau','model_key']).size().eq(4).all()
    return ev,new,both,sources,dict(target_max_abs_difference=target_err,FZ0_max_abs_recalculation_error=score_err,
        identical_start_and_target_dates=True,prior_fit_cutoffs=True,new_ties=int((new.actual_loss==new['var']).sum()))

def pair_summaries(both):
    pair=both[both.model_key.isin(PAIR)].copy()
    mean=pair.groupby(['model_key','tau']).agg(n_dates=('origin_date','nunique'),n_task_forecasts=('score','size'),
        mean_score=('score','mean'),mean_es=('es','mean'),mean_var=('var','mean'),breaches=('strict_hit','sum'),breach_rate=('strict_hit','mean')).reset_index()
    write('model_summary.csv',mean)
    component=pair.groupby(['model_key','book','report','tau']).agg(n_dates=('origin_date','nunique'),
        mean_score=('score','mean'),mean_es=('es','mean'),mean_var=('var','mean'),breaches=('strict_hit','sum'),breach_rate=('strict_hit','mean')).reset_index()
    write('task_summary.csv',component)
    annual=pair.groupby([pair.origin_date.dt.year.rename('year'),'model_key','book','report','tau']).agg(n_dates=('origin_date','nunique'),
        mean_score=('score','mean'),mean_es=('es','mean'),breaches=('strict_hit','sum'),breach_rate=('strict_hit','mean')).reset_index()
    write('annual_task_summary.csv',annual)
    records=[]
    for (model,book,report,tau),g in pair.groupby(['model_key','book','report','tau']):
        g=g.sort_values('origin_date');n=len(g);a=1-tau
        y=g.actual_loss.to_numpy();q=g['var'].to_numpy();e=g.es.to_numpy()
        hit=(y>q).astype(float);h=hit-a;u=(e-q-np.maximum(y-q,0)/a)/e
        # The future-start return is completed two eligible origins later.
        completed_lag=g.target_date.shift(2)<=g.origin_date
        assert completed_lag.iloc[2:].all()
        lag=np.r_[np.nan,np.nan,hit[:-2]]
        ins=np.column_stack([np.ones(n),lag-a,np.log(e)])
        cond=np.column_stack([h[:,None]*ins,u[:,None]*ins]);cond=cond[np.isfinite(cond).all(axis=1)]
        for lags in [20,60]:
            se=float(np.sqrt(max(prior.hac(h,lags)[0,0],0.)))
            us=float(np.sqrt(max(prior.hac(u,lags)[0,0],0.)))
            st,pv,rank=prior.wald(np.column_stack([h,u]),lags)
            cs,cp,cr=prior.wald(cond,lags)
            sparse=bool(hit.sum()<30);degenerate=bool(se==0)
            if sparse:pv=cp=np.nan
            records.append(dict(model_key=model,book=book,report=report,tau=tau,lags=lags,n_dates=n,
                breaches=int(hit.sum()),nominal_expected=n*a,breach_rate=float(hit.mean()),
                rate_HAC_low=np.nan if degenerate else float(hit.mean()-1.96*se),
                rate_HAC_high=np.nan if degenerate else float(hit.mean()+1.96*se),
                es_identification_mean=float(u.mean()),es_identification_HAC_low=float(u.mean()-1.96*us),
                es_identification_HAC_high=float(u.mean()+1.96*us),joint_statistic=st,joint_pvalue=pv,joint_rank=rank,
                conditional_joint_statistic=cs,conditional_joint_pvalue=cp,conditional_joint_rank=cr,
                conditional_joint_n=len(cond),fewer_than_30_breaches=sparse,pvalues_withheld_sparse_tail=sparse,
                empirical_rate_HAC_degenerate=degenerate,calibration_pvalues_unadjusted=True))
    calibration=pd.DataFrame(records);write('calibration.csv',calibration)
    return mean,component,calibration

def pair_contrasts(both):
    pair=both[both.model_key.isin(PAIR)]
    wide=pair.pivot(index=KEY,columns='model_key',values='score')
    delta=(wide[PAIR[0]]-wide[PAIR[1]]).unstack(['book','report','tau']).sort_index(axis=1)
    assert delta.shape==(2046,8) and not delta.isna().any().any()
    results=[];aggregates=[]
    for block in [20,60]:
        bs=circular_means(delta.to_numpy(),block,reps=B,seed=SEED)
        est=delta.mean().to_numpy();se=bs.std(axis=0,ddof=1)
        assert (se>0).all()
        crit=float(np.quantile(np.abs((bs-est)/se).max(axis=1),.95))
        for j,(book,report,tau) in enumerate(delta.columns):
            results.append(dict(book=book,report=report,tau=tau,block=block,n_dates=2046,
                difference='IN_FHS_REF_POOL_minus_SCALAR_BEKK_INTERNAL',mean_difference=float(est[j]),
                pointwise_percentile_low=float(np.quantile(bs[:,j],.025)),pointwise_percentile_high=float(np.quantile(bs[:,j],.975)),
                simultaneous_low=float(est[j]-crit*se[j]),simultaneous_high=float(est[j]+crit*se[j]),
                simultaneous_family_n=8,simultaneous_critical=crit,bootstrap_SE=float(se[j]),bootstrap_draws=B))
        ids=[np.array([j for j,c in enumerate(delta.columns) if c[2]==tau]) for tau in [.95,.99]]
        x=np.column_stack([delta.iloc[:,j].mean(axis=1) for j in ids])
        ab=np.column_stack([bs[:,j].mean(axis=1) for j in ids]);ae=x.mean(axis=0);ass=ab.std(axis=0,ddof=1)
        ac=float(np.quantile(np.abs((ab-ae)/ass).max(axis=1),.95))
        for j,tau in enumerate([.95,.99]):
            aggregates.append(dict(tau=tau,block=block,n_dates=2046,mean_difference=float(ae[j]),
                pointwise_percentile_low=float(np.quantile(ab[:,j],.025)),pointwise_percentile_high=float(np.quantile(ab[:,j],.975)),
                simultaneous_low=float(ae[j]-ac*ass[j]),simultaneous_high=float(ae[j]+ac*ass[j]),
                simultaneous_family_n=2,simultaneous_critical=ac,bootstrap_draws=B))
    write('paired_score_contrasts_eight_task_tail_family.csv',pd.DataFrame(results))
    write('paired_score_contrasts_whole_date.csv',pd.DataFrame(aggregates))
    write('paired_score_differences_daily.csv',delta.reset_index())
    return pd.DataFrame(results),pd.DataFrame(aggregates)

def expanded_mcs(both):
    daily=both.groupby(['origin_date','tau','model_key']).score.mean().unstack('model_key')
    assert len(daily.columns)==42 and not daily.isna().any().any()
    families={'all_42_with_BEKK':list(daily.columns),
        'exclude_standalone_IN_TCOP_with_BEKK':[c for c in daily.columns if not c.startswith('IN_TCOP')]}
    assert [len(v) for v in families.values()]==[42,35]
    records=[];sets=[]
    for tau in [.95,.99]:
        d=daily.xs(tau,level='tau')
        for block in [20,60]:
            bs=circular_means(d.to_numpy(),block,reps=B,seed=SEED)
            for family,names in families.items():
                ids=[d.columns.get_loc(c) for c in names]
                r=mcs_r(d[names].to_numpy(),bs[:,ids],names)
                r['tau']=tau;r['block']=block;r['family']=family;r['family_n']=len(names);records.append(r)
                keep=sorted(r.loc[r.included_95,'model_key']);r=r.set_index('model_key')
                sets.append(dict(tau=tau,block=block,family=family,family_n=len(names),included_n=len(keep),included=keep,
                    mixture_mcs_pvalue=float(r.loc[PAIR[0],'mcs_pvalue']),BEKK_mcs_pvalue=float(r.loc[PAIR[1],'mcs_pvalue'])))
    write('mcs_pvalues_35_42.csv',pd.concat(records))
    (OUT/'mcs_sets_35_42.json').write_text(json.dumps(sets,indent=2)+'\n',encoding='utf-8')
    write('whole_date_scores_42.csv',daily.reset_index())
    return sets

def artifacts_audit(new,both):
    metadata=json.loads((NEW/'manifest.json').read_text())
    ppmeta=json.loads((PP/'manifest.json').read_text())
    checks=pd.read_csv(NEW/'closure_checks.csv',float_precision='round_trip')
    fits=json.loads((NEW/'parameters.json').read_text())
    assert len(checks)==40 and checks.groupby('year').reference.nunique().eq(5).all()
    assert len(fits)==8 and min(v['converged_starts'] for v in fits)==metadata['minimum_converged_starts']==3
    assert all(a['success'] for v in fits for a in v['attempts'])
    cov=float(checks.covariance_relative_error.max());lik=float(checks.likelihood_offset_error.max())
    assert cov==metadata['max_closure_error'] and lik==metadata['max_likelihood_offset_error']
    assert hashlib.sha256(recorded_script_path('joint18_multivariate.py',metadata['source_sha256']).read_bytes()).hexdigest()==metadata['source_sha256']
    assert hashlib.sha256(recorded_script_path('joint18_pit_precision.py',ppmeta['source_sha256']).read_bytes()).hexdigest()==ppmeta['source_sha256']
    tables=[];pits=[];precisions=[]
    p1=pd.read_csv(NEW/'pit.csv',parse_dates=['origin_date'],float_precision='round_trip')
    p2=pd.read_csv(PP/'pit.csv',parse_dates=['origin_date'],float_precision='round_trip')
    assert len(p1)==8184 and len(p2)==ppmeta['pit_rows']==24552
    p=pd.concat([p1,p2],ignore_index=True)
    assert not p.duplicated(['origin_date','model','book','report']).any()
    assert p.groupby(['model','book','report']).size().eq(2046).all()
    assert p.pit.between(0,1).all()
    ties=int((p2.cdf_left!=p2.cdf_right).sum());assert ties==0
    mapping={'IN_FHS1250_EUR':'IN_FHS1250@EUR','IN_FHS_REF_POOL':'IN_FHS_REF_POOL','HS1250':'HS1250',PAIR[1]:PAIR[1]}
    pit_alignment=[]
    for (model,book,report),g in p.groupby(['model','book','report']):
        g=g.sort_values('origin_date');v=g.pit.to_numpy();ordered=np.sort(v);n=len(v)
        # Empirical distribution discrepancy only; no iid Uniform pvalue.
        ks=float(max(np.max(np.arange(1,n+1)/n-ordered),np.max(ordered-np.arange(n)/n)))
        pits.append(dict(model=model,book=book,report=report,n_dates=n,pit_mean=float(v.mean()),pit_variance=float(v.var(ddof=1)),
            pit_min=float(v.min()),pit_max=float(v.max()),descriptive_uniform_KS_distance=ks,
            below_support=int(g.below_support.sum()),above_support=int(g.above_support.sum()),
            pit_zero=int((v==0).sum()),pit_one=int((v==1).sum()),pit_above_95=int((v>.95).sum()),pit_above_99=int((v>.99).sum())))
        hist,_=np.histogram(v,bins=np.linspace(0,1,11))
        for k,c in enumerate(hist):tables.append(dict(model=model,book=book,report=report,bin_low=k/10,bin_high=(k+1)/10,count=int(c)))
        for tau in [.95,.99]:
            forecast=both[(both.model_key==mapping[model])&(both.book==book)&(both.report==report)&(both.tau==tau)].set_index('origin_date').sort_index()
            assert forecast.index.equals(pd.DatetimeIndex(g.origin_date))
            mismatch=int(np.sum((v>tau)!=forecast.strict_hit.to_numpy().astype(bool)))
            pit_alignment.append(dict(model=model,book=book,report=report,tau=tau,PIT_threshold_breach_mismatches=mismatch,
                threshold_discreteness_can_explain_nonzero_mismatches=True))
    write('pit_summary.csv',pd.DataFrame(pits));write('pit_decile_histograms.csv',pd.DataFrame(tables))
    write('pit_hit_alignment.csv',pd.DataFrame(pit_alignment))
    lo=pd.read_csv(NEW/'precision.csv',parse_dates=['origin_date'],float_precision='round_trip');lo['model']=PAIR[1];lo['reference']='ALL_EQUIVARIANT'
    high=pd.read_csv(PP/'precision.csv',parse_dates=['origin_date'],float_precision='round_trip')
    assert lo.origin_date.nunique()==24 and len(lo)==192
    assert high.origin_date.nunique()==ppmeta['precision_origins']==96 and len(high)==ppmeta['precision_rows']==3840
    assert lo.groupby(lo.origin_date.dt.year).origin_date.nunique().eq(3).all()
    assert high.groupby(high.origin_date.dt.year).origin_date.nunique().eq(12).all()
    precision=pd.concat([lo,high],ignore_index=True)
    assert not precision.duplicated(['origin_date','model','reference','book','report','tau']).any()
    calc=(precision.es_high-precision.es_low)/precision.es_high
    err=float(np.max(np.abs(calc-precision.relative_es_change)));assert err<1e-12
    assert np.isclose(np.abs(lo.relative_es_change).max(),metadata['max_precision_relative_es_change'],rtol=1e-12)
    assert np.isclose(np.abs(high.relative_es_change).max(),ppmeta['max_native_tcop_relative_es_change'],rtol=1e-12)
    # Numerical perturbation at calendar-selected origins is not process CI.
    precision['absolute_relative_es_change']=np.abs(precision.relative_es_change)
    for (model,reference,tau),g in precision.groupby(['model','reference','tau']):
        precisions.append(dict(model=model,reference=reference,tau=tau,n_calendar_origins=g.origin_date.nunique(),n_task_rows=len(g),
            median_absolute_relative_ES_change=float(g.absolute_relative_es_change.median()),
            p95_absolute_relative_ES_change=float(g.absolute_relative_es_change.quantile(.95)),
            p99_absolute_relative_ES_change=float(g.absolute_relative_es_change.quantile(.99)),
            max_absolute_relative_ES_change=float(g.absolute_relative_es_change.max())))
    write('precision_summary.csv',pd.DataFrame(precisions))
    write('precision_largest_twenty.csv',precision.nlargest(20,'absolute_relative_es_change'))
    lowcheck=lo.merge(new[KEY+['var','es','actual_loss']],on=KEY,validate='one_to_one')
    varmatch=float(np.max(np.abs(lowcheck.var_low-lowcheck['var'])));esmatch=float(np.max(np.abs(lowcheck.es_low-lowcheck.es)))
    assert varmatch<1e-12 and esmatch<1e-12
    lowcheck['score_low']=fz(lowcheck.actual_loss.to_numpy(),lowcheck.var_low.to_numpy(),lowcheck.es_low.to_numpy(),lowcheck.tau.to_numpy())
    lowcheck['score_high']=fz(lowcheck.actual_loss.to_numpy(),lowcheck.var_high.to_numpy(),lowcheck.es_high.to_numpy(),lowcheck.tau.to_numpy())
    lowcheck['score_high_minus_low']=lowcheck.score_high-lowcheck.score_low
    lowcheck['breach_changed']=(lowcheck.actual_loss>lowcheck.var_high)!=(lowcheck.actual_loss>lowcheck.var_low)
    write('BEKK_precision_score_changes.csv',lowcheck)
    leverage=pd.read_csv(PP/'tail_leverage.csv',parse_dates=['origin_date'],float_precision='round_trip')
    assert len(leverage)==49104 and leverage.origin_date.nunique()==2046
    leverage_summary=leverage.groupby(['model','tau']).agg(n_task_rows=('es','size'),
        median_largest_tail_excess_share=('largest_tail_excess_share','median'),
        p99_largest_tail_excess_share=('largest_tail_excess_share',lambda v:v.quantile(.99)),
        max_largest_tail_excess_share=('largest_tail_excess_share','max'),scenario_n_min=('scenario_n','min'),scenario_n_max=('scenario_n','max')).reset_index()
    write('exact_empirical_tail_leverage_summary.csv',leverage_summary)
    lev_errors=[]
    for model,g in leverage.groupby('model'):
        forecasts=both[both.model_key==mapping[model]][KEY+['var','es']]
        merge=g.merge(forecasts,on=KEY,validate='one_to_one',suffixes=('_audit','_stored'))
        assert len(merge)==len(g)
        qe=float(np.max(np.abs(merge.var_audit-merge.var_stored)));ee=float(np.max(np.abs(merge.es_audit-merge.es_stored)))
        assert qe<1e-12 and ee<1e-12
        lev_errors.append(dict(model=model,max_VaR_reconstruction_error=qe,max_ES_reconstruction_error=ee))
    write('exact_empirical_risk_reconstruction_audit.csv',pd.DataFrame(lev_errors))
    return dict(BEKK_closure_rows=len(checks),BEKK_annual_fits=len(fits),BEKK_attempts=sum(len(v['attempts']) for v in fits),
        BEKK_failed_starts=sum(not a['success'] for v in fits for a in v['attempts']),
        BEKK_max_covariance_closure_error=cov,BEKK_max_likelihood_offset_error=lik,
        BEKK_max_squared_radius_closure_error=float(checks.radius_squared_error.max()),
        BEKK_max_gradient_difference=float(checks.gradient_absolute_error.max()),
        BEKK_precision_origins=24,BEKK_precision_rows=192,
        BEKK_max_absolute_relative_ES_change=float(lo.relative_es_change.abs().max()),
        BEKK_precision_low_forecast_VaR_max_error=varmatch,BEKK_precision_low_forecast_ES_max_error=esmatch,
        BEKK_precision_max_abs_score_change=float(lowcheck.score_high_minus_low.abs().max()),
        BEKK_precision_breach_flips=int(lowcheck.breach_changed.sum()),
        expanded_tcop_precision_origins=96,expanded_tcop_precision_rows=3840,
        expanded_tcop_max_absolute_relative_ES_change=float(high.relative_es_change.abs().max()),
        empirical_PIT_midCDF_ties=ties,total_PIT_rows=len(p),max_PIT_threshold_hit_disagreements=max(x['PIT_threshold_breach_mismatches'] for x in pit_alignment),
        source_hashes_verified=True,exact_empirical_risk_reconstruction=lev_errors)

def main():
    if OUT.exists():raise FileExistsError('Preserve existing comparison; use a new version directory if rerunning')
    OUT.mkdir(parents=True);t0=time.perf_counter()
    ev,new,both,sources,forecast_audit=load()
    mean,component,cal=pair_summaries(both)
    task,aggregate=pair_contrasts(both)
    print('Whole-date pair contrasts',flush=True);print(aggregate.to_string(index=False),flush=True)
    mcs_audit=audit_mcs();sets=expanded_mcs(both)
    print('MCS sets',flush=True)
    print(pd.DataFrame([{k:v for k,v in x.items() if k!='included'} for x in sets]).to_string(index=False),flush=True)
    artifact_audit=artifacts_audit(new,both)
    manifest=dict(status='completed',evaluation_origins=2046,models_compared=PAIR,candidate_family_sizes=[35,42],
        bootstrap_draws=B,blocks=[20,60],seed=SEED,score_difference='mixture_minus_BEKK; negative favours mixture',
        task_tail_simultaneous_family=8,whole_date_tail_simultaneous_family=2,
        MCS_statistic='HLN2011 T_R with coherent max_i max_j studentized elimination',MCS_confidence=.95,
        conditional_on_stored_forecasts=True,refitting_bootstrap=False,historical_reanalysis=True,
        standalone_IN_TCOP_diagnostic_in_all_42=True,established_combinations_may_contain_recorded_TCOP=True,
        calibration_pvalues_unadjusted=True,sparse_tail_pvalues_withheld_below_30_breaches=True,
        empty_empirical_breach_variance_HAC_intervals_withheld=True,
        forecast_audit=forecast_audit,MCS_independent_toy_audit=mcs_audit,artifact_audit=artifact_audit,
        runtime_seconds=time.perf_counter()-t0,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources+list(NEW.glob('*'))+list(PP.glob('*')) if p.is_file()},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('Artifact audit',json.dumps(artifact_audit,indent=2),flush=True)
    print(mean.to_string(index=False),flush=True)
    print(component.to_string(index=False),flush=True)

if __name__=='__main__':main()
