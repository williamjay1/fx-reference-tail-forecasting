from project_config import ROOT as PUBLIC_WORK_ROOT
"""Exploratory revision: common-date MCS and reference-model risk inference.

No method/refit/design-selection uncertainty is manufactured by resampling
stored forecasts. MCS follows Hansen, Lunde and Nason (2011), T_R with coherent
max_i max_j elimination; both full 41 and 34 non-standalone-copula families.
"""
from pathlib import Path
import hashlib, json, time, itertools
import fx_windows_platform_compat
import numpy as np
import pandas as pd
import joint17_analyse as prior

ROOT=PUBLIC_WORK_ROOT
OUT=ROOT/'results/joint_fx_v18/inference'
OLD=ROOT/'results/joint_fx_20261002'
REPS=5000
SEED=20261002
REFERENCES=['EUR','USD','GBP','JPY','CHF']
COMPARATORS=prior.PRIMARY

def circular_means(x,block,reps=REPS,seed=SEED):
    """Exactly n entries: fixed blocks with wraparound, last block truncated."""
    x=np.asarray(x,float)
    if x.ndim==1:x=x[:,None]
    n,k=x.shape;nb=(n+block-1)//block;last=n-(nb-1)*block
    rng=np.random.default_rng(seed)
    sums=sum(np.roll(x,-j,axis=0) for j in range(block))
    partial=sum(np.roll(x,-j,axis=0) for j in range(last))
    out=np.empty((reps,k))
    for start in range(0,reps,100):
        size=min(100,reps-start);ids=rng.integers(0,n,(size,nb))
        out[start:start+size]=(sums[ids[:,:-1]].sum(axis=1)+partial[ids[:,-1]])/n
    return out

def mcs_r(losses,boot_means,names):
    """Return sequentially adjusted inclusion pvalues, not best-model odds.

    Bootstrap d_ij means are centered on observed d_ij. Their mean squared
    errors estimate variance (same convention as arch's R implementation).
    Monte Carlo pvalues use (1 + # bootstrap >= observed)/(B+1).
    """
    losses=np.asarray(losses,float);boot_means=np.asarray(boot_means,float)
    if not np.isfinite(losses).all():raise ValueError('Nonfinite score')
    mean=losses.mean(axis=0);err=boot_means-mean
    differences=mean[:,None]-mean[None,:]
    boot_d=err[:,:,None]-err[:,None,:]
    var=(boot_d**2).mean(axis=0)
    mask=~np.eye(len(names),dtype=bool)
    if np.any(var[mask]<=1e-28):raise ValueError('Degenerate pair: inspect duplicate models')
    var=var+np.eye(len(names))
    z=differences/np.sqrt(var);bz=boot_d/np.sqrt(var)
    keep=list(range(len(names)));records=[];running=0.;step=0
    while len(keep)>1:
        sub=z[np.ix_(keep,keep)];i,j=np.unravel_index(np.argmax(sub),sub.shape)
        stat=float(sub[i,j])
        sim=bz[:,keep,:][:,:,keep].max(axis=(1,2))
        p=(1+int(np.sum(sim>=stat)))/(len(sim)+1)
        running=max(running,p);eliminated=keep[i];step+=1
        records.append(dict(model_key=names[eliminated],elimination_step=step,
                            remaining_before=len(keep),test_statistic=stat,
                            raw_pvalue=p,mcs_pvalue=running,
                            mean_score=float(mean[eliminated]),included_95=running>.05,
                            witness_model=names[keep[j]]))
        keep.pop(i)
    step+=1
    records.append(dict(model_key=names[keep[0]],elimination_step=step,
                        remaining_before=1,test_statistic=0.,raw_pvalue=1.,
                        mcs_pvalue=1.,mean_score=float(mean[keep[0]]),
                        included_95=True,witness_model='none'))
    return pd.DataFrame(records)

def slow_reference_mcs(losses,boot,names):
    """Independent scalar/pair implementation for a small deterministic toy."""
    mean=[float(np.mean(losses[:,i])) for i in range(len(names))]
    sd={};vals={}
    for i in range(len(names)):
        for j in range(len(names)):
            delta=mean[i]-mean[j]
            vals[i,j]=np.array([float(v[i]-v[j]-delta) for v in boot])
            sd[i,j]=1. if i==j else np.sqrt(np.mean(vals[i,j]**2))
    keep=list(range(len(names)));result=[];adjusted=0.
    while len(keep)>1:
        stat,i,j=max(((mean[i]-mean[j])/sd[i,j],i,j) for i in keep for j in keep)
        sim=np.array([max(vals[i,j][b]/sd[i,j] for i in keep for j in keep) for b in range(len(boot))])
        p=(1+int(np.sum(sim>=stat)))/(len(sim)+1);adjusted=max(adjusted,p)
        result.append((names[i],adjusted));keep.remove(i)
    result.append((names[keep[0]],1.))
    return result

def audit_mcs():
    rng=np.random.default_rng(97421);base=rng.normal(size=80)
    toy=np.column_stack([base+rng.normal(size=80)*.1,
                         base+rng.normal(size=80)*.1,
                         base+1.+rng.normal(size=80)*.1])
    names=['good_a','good_b','bad'];boot=circular_means(toy,5,reps=399,seed=801)
    fast=mcs_r(toy,boot,names);slow=slow_reference_mcs(toy,boot,names)
    assert fast.model_key.tolist()==[v[0] for v in slow]
    discrepancy=float(np.max(np.abs(fast.mcs_pvalue.to_numpy()-np.array([v[1] for v in slow]))))
    assert discrepancy==0.
    assert not bool(fast.set_index('model_key').loc['bad','included_95'])
    # Adding a date-specific common score or positive rescaling must preserve
    # every pairwise studentized statistic and the same MCS.
    for transformed in [toy+np.arange(80)[:,None]*.2, toy*3.]:
        bs=circular_means(transformed,5,reps=399,seed=801)
        f=mcs_r(transformed,bs,names)
        assert f.model_key.tolist()==fast.model_key.tolist()
        assert np.max(np.abs(f.mcs_pvalue-fast.mcs_pvalue))==0.
    # Verify block summation against explicit resampling for a toy last block.
    n=17;block=5;x=rng.normal(size=(n,3));nb=(n+block-1)//block
    starts=np.random.default_rng(789).integers(0,n,(43,nb))
    explicit=np.array([x[((row[:,None]+np.arange(block))%n).ravel()[:n]].mean(axis=0) for row in starts])
    err=float(np.max(np.abs(explicit-circular_means(x,block,43,789))))
    assert err<1e-14
    return dict(status='PASS',scalar_mcs_max_pvalue_discrepancy=discrepancy,
                block_means_max_difference=err,bad_model_excluded=True,
                common_date_shift_and_positive_scale_invariance=True)

def score_inference(ev):
    daily=ev.groupby(['origin_date','tau','model_key']).score.mean().unstack('model_key')
    if daily.isna().any().any():raise ValueError('Unequal common dates')
    # Remove native/pool/past-best IN-TCOP standalone forecasts from restricted
    # sensitivity. Established combinations containing one recorded MC member
    # are still conditional forecasts and have that limitation explicitly.
    families={'all_41':list(daily.columns),
              'exclude_standalone_IN_TCOP':[c for c in daily.columns if not c.startswith('IN_TCOP')]}
    assert [len(families[k]) for k in families]==[41,34]
    tables=[];summary=[];margin_records=[]
    for tau in [.95,.99]:
        d=daily.xs(tau,level='tau')
        for block in [20,60]:
            bs=circular_means(d.to_numpy(),block)
            for family,names in families.items():
                ids=[d.columns.get_loc(c) for c in names]
                result=mcs_r(d[names].to_numpy(),bs[:,ids],names)
                result['tau']=tau;result['block']=block;result['family']=family
                result['family_n']=len(names);tables.append(result)
                included=sorted(result.loc[result.included_95,'model_key'])
                pool_p=float(result.set_index('model_key').loc['IN_FHS_REF_POOL','mcs_pvalue'])
                summary.append(dict(tau=tau,block=block,family=family,family_n=len(names),
                                    included_n=len(included),included=included,
                                    mixture_mcs_pvalue=pool_p))
            pool=d.columns.get_loc('IN_FHS_REF_POOL')
            comp=[d.columns.get_loc(c) for c in COMPARATORS]
            diff=d.to_numpy()[:,[pool]]-d.to_numpy()[:,comp]
            bdiff=bs[:,[pool]]-bs[:,comp]
            delta=diff.mean(axis=0);se=bdiff.std(axis=0,ddof=1)
            centered=(bdiff-delta)/se
            critical_up=float(np.quantile(centered.max(axis=1),.95))
            critical_lo=float(np.quantile((-centered).max(axis=1),.95))
            upper=delta+critical_up*se;lower=delta-critical_lo*se
            for j,c in enumerate(COMPARATORS):
                for margin in [.005,.01,.02,.05]:
                    margin_records.append(dict(tau=tau,block=block,comparator=c,
                      observed_difference=float(delta[j]),lower_one_sided_simultaneous_95=float(lower[j]),
                      upper_one_sided_simultaneous_95=float(upper[j]),family_n=5,
                      exploratory_margin=margin,upper_bound_below_exploratory_margin=bool(upper[j]<margin),
                      directional_bounds_inside_exploratory_margin=bool(lower[j]>-margin and upper[j]<margin),
                      required_NI_margin=max(float(upper[j]),0.),
                      required_equivalence_margin=max(float(upper[j]),float(-lower[j]),0.),
                      margin_has_no_prespecified_economic_justification=True))
    pd.concat(tables).to_csv(OUT/'mcs_pvalues.csv',index=False,float_format='%.17g')
    (OUT/'mcs_sets.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    pd.DataFrame(margin_records).to_csv(OUT/'exploratory_margin_sensitivity.csv',index=False,float_format='%.17g')
    daily.reset_index().to_csv(OUT/'whole_date_scores.csv',index=False,float_format='%.17g')
    return summary

def reference_inference(ev):
    g=ev[ev.model=='IN_FHS1250'].copy()
    pivot=g.pivot(index=['origin_date','book','report','tau'],columns='train_reference',values='es')[REFERENCES]
    if (pivot<=0).any().any() or pivot.isna().any().any():raise ValueError('ES must be positive and balanced')
    # Every family is 4 references vs EUR; global16 contrasts at each tail
    # retains the same bootstrap date starts across all 4 tasks.
    records=[]
    for tau in [.95,.99]:
        tasks=[];logs=[];cash=[]
        for book in sorted(g.book.unique()):
            for report in ['EUR','USD']:
                a=pivot.xs((book,report,tau),level=['book','report','tau'])
                if tasks and not a.index.equals(index):raise ValueError('Reference date mismatch')
                index=a.index;tasks.append((book,report))
                logs.append(np.log(a.iloc[:,1:].to_numpy()/a.iloc[:,[0]].to_numpy()))
                cash.append(a.iloc[:,1:].to_numpy()-a.iloc[:,[0]].to_numpy())
        x=np.concatenate(logs,axis=1);raw=np.concatenate(cash,axis=1)
        for block in [20,60]:
            boot=circular_means(x,block);est=x.mean(axis=0);se=boot.std(axis=0,ddof=1)
            z=np.abs((boot-est)/se)
            cglobal=float(np.quantile(z.max(axis=1),.95))
            for task_id,(book,report) in enumerate(tasks):
                a=pivot.xs((book,report,tau),level=['book','report','tau'])
                sl=slice(4*task_id,4*task_id+4);ctask=float(np.quantile(z[:,sl].max(axis=1),.95))
                for j,ref in enumerate(REFERENCES[1:]):
                    k=4*task_id+j
                    records.append(dict(book=book,report=report,tau=tau,reference=ref,comparator='EUR',
                      block=block,n_dates=len(index),mean_log_ES_ratio=float(est[k]),
                      geometric_ES_ratio=float(np.exp(est[k])),median_ES_ratio=float(np.median(a[ref]/a.EUR)),
                      mean_normalized_cash_ES_difference=float(raw[:,k].mean()),
                      task_simultaneous_low=float(est[k]-ctask*se[k]),
                      task_simultaneous_high=float(est[k]+ctask*se[k]),task_family_n=4,
                      all_task_simultaneous_low=float(est[k]-cglobal*se[k]),
                      all_task_simultaneous_high=float(est[k]+cglobal*se[k]),all_task_family_n=16))
    pd.DataFrame(records).to_csv(OUT/'signed_reference_ES_contrasts.csv',index=False,float_format='%.17g')
    return pivot

def factorial_decomposition(pivot):
    """Balanced orthogonal decomposition of log ES; no independent-row F tests.

    At each date/book, axes R=reference, N=report and Q=tail. Date-specific
    grand means are removed. Seven main/interaction sums of squares partition
    the remaining finite array. Cash levels are never pooled as equivalent.
    """
    records=[]
    for book in sorted(pivot.index.get_level_values('book').unique()):
        z=pivot.xs(book,level='book')
        dates=sorted(z.index.get_level_values('origin_date').unique())
        x=np.empty((len(dates),5,2,2))
        for n,report in enumerate(['EUR','USD']):
            for q,tau in enumerate([.95,.99]):
                x[:,:,n,q]=np.log(z.xs((report,tau),level=['report','tau']).reindex(dates).to_numpy())
        grand=x.mean(axis=(1,2,3),keepdims=True)
        effects={():grand}
        for size in [1,2,3]:
            for keep in itertools.combinations((1,2,3),size):
                average=x.mean(axis=tuple(j for j in (1,2,3) if j not in keep),keepdims=True) if size<3 else x.copy()
                effect=average.copy()
                for sub_size in range(size):
                    for subset in itertools.combinations(keep,sub_size):effect=effect-effects[subset]
                effects[keep]=effect
        total=float(np.sum((x-grand)**2));summed=0.
        for keep,effect in effects.items():
            if not keep:continue
            ss=float(np.sum(np.broadcast_to(effect,x.shape)**2));summed+=ss
            label=' × '.join({1:'training_reference',2:'reporting_currency',3:'tail_level'}[k] for k in keep)
            records.append(dict(book=book,scope='five_refs_two_reports_two_tails',factor=label,
              n_dates=len(dates),sum_squares=ss,total_within_date_sum_squares=total,
              share=ss/total,rms_log_effect=np.sqrt(ss/x.size),pvalue='not_applicable'))
        assert abs(summed-total)<1e-10*max(1.,total)
        # Tail-specific two-factor partition avoids averaging economically
        # different tail functionals when comparing reference vs reporting.
        for qi,tau in enumerate([.95,.99]):
            a=x[:,:,:,qi];g=a.mean(axis=(1,2),keepdims=True)
            reference=a.mean(axis=2,keepdims=True)-g
            report=a.mean(axis=1,keepdims=True)-g
            interaction=a-g-reference-report
            tot=float(np.sum((a-g)**2));s=0.
            for label,term in [('training_reference',reference),('reporting_currency',report),('interaction',interaction)]:
                ss=float(np.sum(np.broadcast_to(term,a.shape)**2));s+=ss
                records.append(dict(book=book,scope=f'fixed_tail_{tau}',factor=label,n_dates=len(dates),
                    sum_squares=ss,total_within_date_sum_squares=tot,share=ss/tot,
                    rms_log_effect=np.sqrt(ss/a.size),pvalue='not_applicable'))
            assert abs(s-tot)<1e-10*max(1.,tot)
    pd.DataFrame(records).to_csv(OUT/'within_date_log_ES_variance_decomposition.csv',index=False,float_format='%.17g')

def parameter_tables():
    source=OLD/'scenario_v2/garch_fit_log.csv'
    d=pd.read_csv(source,float_precision='round_trip');d=d[d.year.between(2018,2025)].copy()
    assert len(d)==160 and d.converged_starts.eq(3).all()
    d['pair']=d.node+'/'+d.training_reference
    d['persistence']=d.alpha+d.beta
    d['unconditional_variance']=d.omega/(1-d.persistence)
    d.to_csv(OUT/'annual_garch_parameters.csv',index=False,float_format='%.17g')
    d.groupby('training_reference')[['alpha','beta','nu','persistence']].agg(['min','median','max']).to_csv(OUT/'garch_parameter_reference_summary.csv',float_format='%.17g')
    corr=[];cop=[]
    for year in range(2018,2026):
        sources=list(OLD.glob(f'internal_full_*/copula_parameters_{year}.json'))
        assert len(sources)==1
        params=json.loads(sources[0].read_text())
        for p in params:
            ref=p['reference'];nodes=[c for c in REFERENCES if c!=ref]
            cop.append(dict(year=year,reference=ref,nu=p['copula_df'],mean_pseudo_loglik=p['copula_loglik']))
            for i in range(4):
                for j in range(i+1,4):
                    corr.append(dict(year=year,training_reference=ref,node_i=nodes[i],node_j=nodes[j],
                         shrunk_t_quantile_correlation=p['copula_corr'][i][j]))
    pd.DataFrame(cop).to_csv(OUT/'annual_internal_copula_parameters.csv',index=False)
    pd.DataFrame(corr).to_csv(OUT/'annual_internal_copula_correlations.csv',index=False,float_format='%.17g')

def main():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():raise FileExistsError('Preserve completed run')
    audit=audit_mcs();(OUT/'mcs_implementation_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    _,ev,sources=prior.read_all()
    summary=score_inference(ev)
    pivot=reference_inference(ev);factorial_decomposition(pivot);parameter_tables()
    manifest=dict(status='completed',historical_exploratory_reanalysis=True,
      method='HLN2011 T_R coherent max_i max_j elimination',bootstrap='fixed circular date blocks',
      bootstrap_reps=REPS,bootstrap_blocks=[20,60],seed=SEED,n_dates=ev.origin_date.nunique(),
      configurations=ev.model_key.nunique(),tasks_per_date=4,tails=[.95,.99],
      parameter_refitting=False,research_design_selection_uncertainty_included=False,
      all_41_conditions_on_recorded_MC_values=True,
      restricted_34_excludes_standalone_IN_TCOP_but_combinations_contain_recorded_IN_TCOP_member=True,
      margin_sensitivity_exploratory_not_prespecified_economic_equivalence=True,
      mcs_theory_requires_stationary_mixing_loss_differentials_with_finite_higher_moments=True,
      mcs_pvalue_not_probability_best=True,variance_decomposition_descriptive_not_ANOVA_test=True,
      numerical_audit=audit,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      runtime_seconds=time.perf_counter()-started)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':main()
