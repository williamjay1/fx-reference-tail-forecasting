from project_config import ROOT as PUBLIC_WORK_ROOT
"""Independent aggregation and internal-studentization checks; owned math audit."""
from pathlib import Path
import json
import hashlib
import fx_windows_platform_compat
import numpy as np
import pandas as pd
import joint17_models as m
import joint17_internal_scenarios as internal

ROOT = PUBLIC_WORK_ROOT
REV = ROOT/'results/joint_fx_20261002'
OUT = REV/'math'
KEY = ['origin_date','book','report','tau','model','train_reference']

def hash_file(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def integration_changes(frames, low, high):
    d = frames[low].merge(frames[high],on=KEY,suffixes=('_lo','_hi'),validate='one_to_one')
    for name in ['var','es']:
        d[name+'_relative_change'] = abs(d[name+'_lo']-d[name+'_hi'])/abs(d[name+'_hi'])
    d['score_absolute_change'] = abs(d.score_lo-d.score_hi)
    d['formal_period'] = d.origin_date.str[:4].astype(int)>=2018
    return d

def group_summary(d, category):
    return d.groupby([category,'tau'],dropna=False).agg(
        comparisons=('origin_date','size'),max_q_relative=('var_relative_change','max'),
        median_q_relative=('var_relative_change','median'),
        max_es_relative=('es_relative_change','max'),median_es_relative=('es_relative_change','median'),
        max_score_change=('score_absolute_change','max'),median_score_change=('score_absolute_change','median')).reset_index()

def reference_contrasts(frames, low, high, model):
    out=[]
    k=['origin_date','book','report','tau']
    for key,g in frames[low][frames[low].model==model].groupby(k):
        gl=g.set_index('train_reference')
        gh=frames[high]
        for name,val in zip(k,key):gh=gh[gh[name]==val]
        gh=gh[gh.model==model].set_index('train_reference')
        for ref in m.CCY[1:]:
            for metric in ['var','es','score']:
                lo=float(gl.loc[ref,metric]-gl.loc['EUR',metric])
                hi=float(gh.loc[ref,metric]-gh.loc['EUR',metric])
                out.append(dict(zip(k,key),reference=ref,metric=metric,contrast_lo=lo,contrast_hi=hi,
                                contrast_change=hi-lo,sign_change=bool(lo*hi<0),
                                integration_change_exceeds_contrast=bool(abs(hi-lo)>abs(hi))))
    return pd.DataFrame(out)

def original_audit():
    r=pd.concat([pd.read_csv(p) for p in sorted((OUT/'precision').glob('risks_[0-9]*.csv'))],ignore_index=True)
    frames={p:r[r.power==p] for p in [12,14,16]}
    summaries={};contrasts={}
    for low,high in [(12,16),(14,16)]:
        d=integration_changes(frames,low,high)
        for period,sel in [('development_2015',~d.formal_period),('formal_2018_2025',d.formal_period)]:
            s=group_summary(d[sel],'model')
            s.to_csv(OUT/'precision'/f'original_summary_{period}_{low}_{high}.csv',index=False,float_format='%.17g')
            summaries[f'{period}_{low}_{high}']=s.to_dict('records')
        c=reference_contrasts(frames,low,high,'TCOP1250')
        c.to_csv(OUT/'precision'/f'original_reference_contrasts_{low}_{high}.csv',index=False,float_format='%.17g')
        for period,sel in [('development_2015',c.origin_date.str[:4].astype(int)==2015),
                          ('formal_2018_2025',c.origin_date.str[:4].astype(int)>=2018)]:
            z=c[sel]
            contrasts[f'{period}_{low}_{high}']=z.groupby(['tau','metric']).agg(comparisons=('sign_change','size'),
                sign_changes=('sign_change','sum'),numerical_change_larger_than_contrast=('integration_change_exceeds_contrast','sum'),
                max_contrast_change=('contrast_change',lambda x:float(abs(x).max()))).reset_index().to_dict('records')
    fixed=r[r.power==0].copy()
    worst=fixed.sort_values('es',ascending=False).head(12)
    worst.to_csv(OUT/'precision'/'original_exact_tail_worst.csv',index=False,float_format='%.17g')
    return dict(origins=int(r.origin_date.nunique()),risk_rows=len(r),summaries=summaries,contrasts=contrasts,
                exact_finite_worst=worst.to_dict('records'),max_absolute_identity=float(r.max_abs_cash_identity_residual.max()))

def internal_audit():
    frames={};manifests={}
    for p in [12,14,16,18]:
        base=REV/f'internal_precision_{p}'
        if not (base/'manifest.json').exists():continue
        frames[p]=pd.read_csv(base/'predictions.csv')
        manifests[p]=json.loads((base/'manifest.json').read_text())
        assert manifests[p]['prediction_sha256']==hash_file(base/'predictions.csv')
    summaries={};contrasts={}
    for low,high in [(12,16),(14,16),(16,18)]:
        if low not in frames or high not in frames:continue
        d=integration_changes(frames,low,high)
        for period,sel in [('development_2015',~d.formal_period),('formal_2018_2025',d.formal_period)]:
            s=group_summary(d[sel],'model')
            s.to_csv(OUT/'precision'/f'internal_independent_{period}_{low}_{high}.csv',index=False,float_format='%.17g')
            summaries[f'{period}_{low}_{high}']=s.to_dict('records')
        c=reference_contrasts(frames,low,high,'IN_TCOP1250')
        c.to_csv(OUT/'precision'/f'internal_independent_reference_contrasts_{low}_{high}.csv',index=False,float_format='%.17g')
        for period,sel in [('development_2015',c.origin_date.str[:4].astype(int)==2015),
                          ('formal_2018_2025',c.origin_date.str[:4].astype(int)>=2018)]:
            z=c[sel]
            contrasts[f'{period}_{low}_{high}']=z.groupby(['tau','metric']).agg(comparisons=('sign_change','size'),
                sign_changes=('sign_change','sum'),numerical_change_larger_than_contrast=('integration_change_exceeds_contrast','sum'),
                max_contrast_change=('contrast_change',lambda x:float(abs(x).max()))).reset_index().to_dict('records')
    final=frames[max(frames)]
    return dict(powers=sorted(frames),origins=int(final.origin_date.nunique()),manifests=manifests,summaries=summaries,
                contrasts=contrasts,max_scenario_loss=float(final.maximum_scenario_loss.max()),
                max_identity_relative=float(final.cash_identity_relative_error.max()))

def independent_bound_checks():
    panel,r,ewma,radius,target=internal.prepare()
    # Reconstruct each covariance recursion explicitly, independent of inverse-based radius formula.
    max_radial_identity=0.;max_basis_radius=0.;rng=np.random.default_rng(73108)
    ids=[int(np.flatnonzero(panel.date.to_numpy()==np.datetime64(x))[0])-1 for x in internal.FIXED_DATES]
    for s in ids:
        previous=ewma[s-1];a=float(r[s]@np.linalg.solve(previous,r[s]))
        expected=a/(.94+.06*a);observed=float(r[s]@np.linalg.solve(ewma[s],r[s]))
        max_radial_identity=max(max_radial_identity,abs(observed-expected))
        for ref in m.CCY[1:]:
            A,_=m.basis_matrix(ref);rr=A@r[s];hh=A@ewma[s]@A.T
            max_basis_radius=max(max_basis_radius,abs(rr@np.linalg.solve(hh,rr)-observed))
    assert max_radial_identity<1e-9 and max_basis_radius<1e-9
    scalar=[]
    for year in [2015,2020,2022,2025]:
        cutoff,annual=internal.annual_data(panel,r,year,8)
        for ref,others,fits,h,z,cop,u in annual:
            for j,fit in enumerate(fits):
                A,_=m.basis_matrix(ref);rb=100*r@A.T;eps=rb[:,j]-fit['mu']
                post=fit['omega']+fit['alpha']*eps*eps+fit['beta']*h[:,j]
                residual=eps/np.sqrt(post)
                assert np.max(abs(residual-z[:,j]))<1e-12
                bound=1/np.sqrt(fit['alpha'])
                train=z[cutoff-1250:cutoff,j];normalized=(train-train.mean())/train.std()
                # Exact probability-law moments of discrete inverse CDF, not a Monte Carlo test.
                assert abs(normalized.mean())<1e-12 and abs(np.mean(normalized**2)-1)<1e-12
                assert abs(residual).max()<=bound+1e-7
                scalar.append(dict(year=year,reference=ref,currency=m.CCY[others[j]],alpha=fit['alpha'],
                    raw_bound=bound,raw_max=float(abs(residual).max()),prior_sd=float(train.std()),
                    normalized_prior_max=float(abs(normalized).max())))
    scalar=pd.DataFrame(scalar);scalar.to_csv(OUT/'internal_scalar_bound_checks.csv',index=False,float_format='%.17g')
    # Finite empirical radial law E[R_scaled^2]=4 is exact before random direction integration.
    max_radial_second_moment=0.
    for s in ids:
        window=radius[s+1-1250:s+1];scaled=window/np.sqrt(np.mean(window**2)/4)
        max_radial_second_moment=max(max_radial_second_moment,abs(float(np.mean(scaled**2))-4))
    # Prefix-only perturbation: future returns cannot alter known h_t or internal z_s for s<t.
    t=ids[-1]+1;changed=r.copy();changed[t:]=rng.normal(0,.4,changed[t:].shape)
    ewma_changed=m.full_ewma_states(changed)
    assert np.array_equal(ewma[:t],ewma_changed[:t])
    first_origin=min(int(np.flatnonzero(panel.date.dt.year.to_numpy()==2013)[0]),min(ids)+1)
    assert first_origin>20
    return dict(radial_global_squared_max=float(max(radius**2)),radial_squared_bound=1/.06,
        max_sherman_morrison_radius_error=max_radial_identity,max_basis_radius_error=max_basis_radius,
        max_radial_second_moment_error=max_radial_second_moment,
        minimum_scalar_alpha=float(scalar.alpha.min()),minimum_prior_internal_sd=float(scalar.prior_sd.min()),
        maximum_standardized_prior_shock=float(scalar.normalized_prior_max.max()),
        future_covariance_prefix_unchanged=True,empirical_discrete_moments_exact=True,
        ewma_twenty_observation_initialization_precedes_all_evaluation_origins=True,
        source_hashes={p.name:hash_file(p) for p in [Path(__file__),Path(m.__file__),Path(internal.__file__)]})

if __name__=='__main__':
    result=dict(original=original_audit(),internal=internal_audit(),bound_checks=independent_bound_checks())
    (OUT/'NUMERICAL_PRECISION_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(original_origins=result['original']['origins'],internal_powers=result['internal']['powers'],
                         internal_max_scenario_loss=result['internal']['max_scenario_loss'],bounds=result['bound_checks']),indent=2))
