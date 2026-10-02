from project_config import ROOT as PUBLIC_WORK_ROOT
"""Independent recurrence, cash valuation and fractional-tail reconstruction."""
from pathlib import Path
import json,hashlib,math
import fx_windows_platform_compat
import numpy as np
import pandas as pd

ROOT=PUBLIC_WORK_ROOT
OUT=ROOT/'results/joint_fx_v18/inference'
CCY=['EUR','USD','GBP','JPY','CHF']
BOOKS={'funded_assets':np.array([.25,.25,.25,.25]),'net_cashflows':np.array([.4,.1,-.3,-.2])}

def independent_forecast(r,train,ref,fits):
    idx=CCY.index(ref);allr=np.column_stack([np.zeros(len(r)),r])
    alltr=np.column_stack([np.zeros(len(train)),train]);others=[i for i in range(5) if i!=idx]
    x=100*(allr[:,others]-allr[:,[idx]])
    tr=100*(alltr[:,others]-alltr[:,[idx]])
    innovations=[];next_variance=[];bound_violation=0.
    for j,f in enumerate(fits):
        h=float(np.var(tr[:,j]));res=[]
        for observed in x[:,j]:
            eps=float(observed-f['mu'])
            post=f['omega']+f['alpha']*eps*eps+f['beta']*h
            res.append(eps/math.sqrt(post));h=max(post,1e-10)
        z=np.array(res[-1250:]);bound_violation=max(bound_violation,float(np.max(np.abs(z))-1/math.sqrt(f['alpha'])))
        innovations.append((z-z.mean())/z.std());next_variance.append(h)
    z=np.column_stack(innovations);n=len(z)-1
    path=np.zeros((n,2,5))
    for j,node in enumerate(others):
        f=fits[j];h=next_variance[j]
        x1=f['mu']+np.sqrt(h)*z[:-1,j]
        h2=f['omega']+f['alpha']*(x1-f['mu'])**2+f['beta']*h
        x2=f['mu']+np.sqrt(h2)*z[1:,j]
        path[:,0,node]=x1/100;path[:,1,node]=x2/100
    path=path-path[:,:,[0]]
    result=[]
    for book,w in BOOKS.items():
        ba=np.exp(path[:,0,1:])@w-w.sum()
        end=np.exp(path[:,0,1:]+path[:,1,1:])@w-w.sum()
        for report in ['EUR','USD']:
            n=CCY.index(report)
            loss=ba*np.exp(-path[:,0,n])-end*np.exp(-path[:,0,n]-path[:,1,n])
            ordered=np.sort(loss)
            for tau in [.95,.99]:
                q=ordered[math.ceil(len(ordered)*tau)-1]
                # Separate explicit sorted fractional-tail representation.
                boundary=math.ceil(tau*len(ordered))-1
                mass_at_boundary=(boundary+1)/len(ordered)-tau
                es=(mass_at_boundary*q+ordered[boundary+1:].sum()/len(ordered))/(1-tau)
                result.append(dict(book=book,report=report,tau=tau,var=float(q),es=float(es)))
    return result,bound_violation

def main():
    panel=pd.read_csv(ROOT/'datasets/joint_fx_20261002/ecb_joint_panel.csv',parse_dates=['date'])
    r=np.diff(np.log(panel[CCY].to_numpy()[:,1:]),axis=0)
    stored=pd.read_csv(ROOT/'results/joint_fx_20261002/internal_full/predictions.csv',float_precision='round_trip')
    stored=stored[stored.model=='IN_FHS1250']
    comparisons=[];baseline_max=0.;zero_failures=True;full_convergence=True;attempts=0;max_bound_violation=0.;manifests=[]
    for block in [20,60]:
        out=OUT/f'parameter_bootstrap/three_origins_B200_block{block}'
        manifest=json.loads((out/'manifest.json').read_text());manifests.append(manifest)
        assert manifest['status']=='completed'
        f=pd.read_csv(out/'fits.csv',float_precision='round_trip')
        p=pd.read_csv(out/'forecasts.csv',float_precision='round_trip')
        assert len(f)==12000 and len(p)==24000
        attempts+=len(f);zero_failures &= f.status.eq('converged').all()
        full_convergence &= f.converged_starts.eq(3).all()
        assert f.groupby(['year','replicate']).joint_indices_sha256.nunique().eq(1).all()
        assert (f.alpha>0).all() and (f.beta>0).all() and ((f.alpha+f.beta)<.999).all() and (f.nu>2.05).all()
        assert p.groupby(['year','replicate','book','report','tau']).size().eq(5).all()
        base=pd.read_csv(out/'baseline_forecasts.csv',float_precision='round_trip').rename(columns={'reference':'train_reference'})
        matched=base.merge(stored,on=['origin_date','train_reference','book','report','tau'])
        assert len(matched)==120
        baseline_max=max(baseline_max,float(np.max(abs(matched.var_x-matched.var_y))),float(np.max(abs(matched.es_x-matched.es_y))))
        for year in [2018,2021,2025]:
            params=json.loads((ROOT/f'results/joint_fx_20261002/scenario_v2/parameters_{year}.json').read_text())
            cutoff=int(np.flatnonzero(panel.date.to_numpy()==np.datetime64(params['cutoff']))[0])
            t=int(np.flatnonzero(panel.date.dt.year.to_numpy()==year)[0]);train=r[cutoff-1250:cutoff];known=r[:t]
            for rep in [0,57,199]:
                for ref in CCY:
                    fits=f[(f.year==year)&(f.replicate==rep)&(f.reference==ref)].to_dict('records')
                    computed,violation=independent_forecast(known,train,ref,fits)
                    max_bound_violation=max(max_bound_violation,violation)
                    obs=p[(p.year==year)&(p.replicate==rep)&(p.reference==ref)]
                    z=pd.DataFrame(computed).merge(obs,on=['book','report','tau'])
                    assert len(z)==8
                    for _,row in z.iterrows():
                        comparisons.append(dict(year=year,replicate=rep,reference=ref,block=block,
                           book=row.book,report=row.report,tau=row.tau,
                           absolute_var_difference=abs(row.var_x-row.var_y),
                           absolute_ES_difference=abs(row.es_x-row.es_y)))
        # Independently reconstruct every range/ratio record from stored ES.
        ranges=pd.read_csv(out/'range_sensitivity.csv',float_precision='round_trip')
        for _,row in ranges.iterrows():
            g=p[(p.origin_date==row.origin_date)&(p.book==row.book)&(p.report==row.report)&(p.tau==row.tau)]
            x=g.pivot(index='replicate',columns='reference',values='es')[CCY]
            span=(x.max(axis=1)-x.min(axis=1))/x.EUR
            assert abs(span.median()-row.perturbation_range_median)<1e-14
            assert abs(span.quantile(.025)-row.perturbation_range_p025)<1e-14
            assert abs(span.quantile(.975)-row.perturbation_range_p975)<1e-14
    d=pd.DataFrame(comparisons);d.to_csv(OUT/'parameter_bootstrap/independent_recurrence_check.csv',index=False,float_format='%.17g')
    maxvar=float(d.absolute_var_difference.max());maxes=float(d.absolute_ES_difference.max())
    assert maxvar<1e-13 and maxes<1e-13 and baseline_max<1e-13 and zero_failures and full_convergence
    audit=dict(status='PASS',total_coordinate_fit_attempts=attempts,
       all_three_optimizer_starts_converged=bool(full_convergence),zero_failed_coordinate_fits=bool(zero_failures),
       full_200_replicates_in_every_origin_block_task=True,
       maximum_baseline_difference_from_existing_v17=baseline_max,
       independently_reconstructed_risk_pairs=len(d),independent_maximum_VaR_difference=maxvar,
       independent_maximum_ES_difference=maxes,maximum_internal_bound_violation=max_bound_violation,
       independent_full_range_percentile_check=True,conditional_parameter_perturbation_only=True,
       parameter_percentiles_not_true_ES_confidence_interval=True,
       script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'parameter_bootstrap/independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
