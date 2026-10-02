"""Independent mathematical audit of joint17 pilot; no model retraining.

Owned by joint_fx_math reviewer. Reads root scripts/results and writes only
math/audit products. Synthetic checks are explicitly separate from FX evidence.
"""
from pathlib import Path
import hashlib
import json
import math
import fx_windows_platform_compat
import numpy as np
import pandas as pd
import joint17_models as m

ROOT=Path('D:/MLWork/FXTailRisk')
REV=ROOT/'results/joint_fx_20261002'
OUT=REV/'math'
SCRIPT_NAMES=('joint17_build_data.py','joint17_models.py','joint17_run_scenarios.py')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def assert_close(name, actual, expected, tol, results):
    err=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
    results[name]={'max_abs_error':err,'tolerance':tol,'passed':bool(err<=tol)}
    assert err<=tol,(name,err,tol)

def independent_es(x,tau,weights=None):
    """Integrate upper empirical quantile via descending fractional mass."""
    x=np.asarray(x,float)
    w=np.full(len(x),1/len(x)) if weights is None else np.asarray(weights,float)
    order=np.argsort(x)
    cum=0.;q=None
    for i in order:
        cum+=w[i]
        if cum+1e-13>=tau:
            q=float(x[i]);break
    remaining=1-tau;area=0.
    for i in order[::-1]:
        take=min(remaining,float(w[i]));area+=take*x[i];remaining-=take
        if remaining<=1e-14:break
    assert abs(remaining)<1e-12
    return q,area/(1-tau)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    script_hashes={n:sha(ROOT/'scripts'/n) for n in SCRIPT_NAMES}
    checks={};rng=np.random.default_rng(2026100211)
    panel=pd.read_csv(ROOT/'datasets/joint_fx_20261002/ecb_joint_panel.csv',parse_dates=['date'])
    dates=panel.date.to_numpy();prices=panel[m.CCY].to_numpy(float)
    r=np.diff(np.log(prices[:,1:]),axis=0)
    predpath=REV/'pilot_v1/predictions.csv'
    pred=pd.read_csv(predpath,parse_dates=['origin_date','start_date','target_date','fit_cutoff_date'])
    params=json.loads((REV/'pilot_v1/parameters_2017.json').read_text(encoding='utf-8'))
    origins=sorted(pred.origin_date.unique());index={pd.Timestamp(d):i for i,d in enumerate(dates)}

    # Actual loss independently from fixed real quantities and future price rows.
    targets=pred[['origin_date','start_date','target_date','book','report','actual_loss']].drop_duplicates()
    errs=[];date_errors=[]
    for row in targets.itertuples(index=False):
        t=index[row.origin_date];w=m.BOOKS[row.book]
        a=np.r_[-w.sum(),w/prices[t,1:]]
        n=m.CCY.index(row.report)
        va=float(a@prices[t+1]);vt=float(a@prices[t+2])
        expected=(va/prices[t+1,n]-vt/prices[t+2,n])*prices[t,n]
        errs.append(row.actual_loss-expected)
        date_errors.extend([row.start_date!=pd.Timestamp(dates[t+1]),row.target_date!=pd.Timestamp(dates[t+2])])
    assert not any(date_errors)
    assert_close('observed_future_start_fixed_quantity_cash',errs,0,1e-13,checks)
    checks['task_date_offsets']={'passed':True,'distinct_origin_book_report_rows':len(targets),'origins':len(origins)}

    # Independent sign transformation, rather than reusing positive-part code.
    loss=pred.actual_loss.to_numpy();q=pred['var'].to_numpy();e=pred.es.to_numpy();alpha=1-pred.tau.to_numpy()
    y=-loss;v=-q;er=-e
    lower=-(np.where(y<=v,v-y,0))/(alpha*er)+v/er+np.log(-er)-1
    assert_close('all_saved_FZ0_scores_via_lower_return_transform',pred.score,lower,1e-10,checks)
    assert_close('all_saved_tail_identification_residuals',pred.tail_residual,e-q-np.maximum(loss-q,0)/alpha,1e-12,checks)
    assert np.array_equal(pred.strict_hit.to_numpy(),(loss>q).astype(int))
    assert np.all(e>=q-1e-12) and np.all(e>0) and np.isfinite(lower).all()
    checks['saved_domain_hits_and_finite_scores']={'passed':True,'forecast_rows':len(pred)}

    # Unweighted and weighted empirical ES integral with fractional boundary mass.
    maxq=maxe=0.;riskcases=0
    for n in (5,19,499,500,1249,1250):
        x=rng.normal(0,.03,n)
        for tau in (.7,.95,.99):
            for weighted in (False,True):
                w=rng.uniform(.01,1,n) if weighted else None
                if weighted:w=w/w.sum()
                qa,ea=m.risk_pair(x,tau,w);qb,eb=independent_es(x,tau,w)
                maxq=max(maxq,abs(qa-qb));maxe=max(maxe,abs(ea-eb));riskcases+=1
    checks['fractional_tail_integral']={'passed':maxq<1e-12 and maxe<1e-11,'cases':riskcases,'max_var_error':maxq,'max_es_error':maxe}
    assert checks['fractional_tail_integral']['passed']

    # Every basis inverts both steps and recovers actual HS risk, not just covariance.
    t=index[pd.Timestamp(origins[0])]
    past=np.stack([r[t-500:t-1],r[t-499:t]],axis=1)
    canonical=np.zeros((len(past),2,5));canonical[:,:,1:]=past
    base_losses,_=m.cash_losses(canonical)
    basis_path_error=basis_cash_error=basis_q_error=basis_es_error=0.
    for reference in m.CCY:
        a,others=m.basis_matrix(reference)
        assert abs(np.linalg.det(a))>.99
        recovered=m.to_eur(past@a.T,reference,others)
        basis_path_error=max(basis_path_error,float(np.max(abs(recovered-canonical))))
        current,_=m.cash_losses(recovered)
        for key in base_losses:
            basis_cash_error=max(basis_cash_error,float(np.max(abs(current[key]-base_losses[key]))))
            for tau in (.95,.99):
                qa,ea=m.risk_pair(base_losses[key],tau);qb,eb=m.risk_pair(current[key],tau)
                basis_q_error=max(basis_q_error,abs(qa-qb));basis_es_error=max(basis_es_error,abs(ea-eb))
    checks['five_basis_HS_negative_control']={'passed':max(basis_path_error,basis_cash_error,basis_q_error,basis_es_error)<1e-12,
      'max_path_error':basis_path_error,'max_cash_error':basis_cash_error,'max_VaR_error':basis_q_error,'max_ES_error':basis_es_error}
    assert checks['five_basis_HS_negative_control']['passed']

    # Prefix initialization explicitly fixed from prior training variance.
    fit=params['fits'][1]['garch'][0]
    a,others=m.basis_matrix(params['fits'][1]['reference'])
    rb=100*(r@a.T)
    cutoff=index[pd.Timestamp(params['cutoff'])]
    tr=rb[cutoff-1250:cutoff,0];h0=float(np.var(tr))
    original=m.variance_path(rb[:,0],fit['omega'],fit['alpha'],fit['beta'],fit['mu'],h0)
    altered=rb[:,0].copy();altered[t:]=100*rng.normal(size=len(altered)-t)
    after=m.variance_path(altered,fit['omega'],fit['alpha'],fit['beta'],fit['mu'],h0)
    assert_close('GARCH_prefix_unchanged_when_future_returns_altered',original[:t+1],after[:t+1],0.,checks)
    checks['known_label_indices']={'passed':True,'origin_t':t,'annual_cutoff_t':cutoff,
      'training_last_return_index':cutoff-1,'known_last_return_index':t-1,
      'FHS_last_second_step_residual_index':t-1,'forecast_h_t_uses_last_return':t-1,
      'all_training_and_shock_labels_known':cutoff<t}

    # Full EWMA covariance and bounded radial law invariance.
    ewma=m.full_ewma_states(r);cov=ewma[t-1]
    cov0=np.cov(r[:20].T)
    for rv in r[:t]:cov0=.94*cov0+.06*np.outer(rv,rv)
    assert_close('EWMA_current_covariance_prior_prefix_recursion',cov,cov0,1e-15,checks)
    sample=r[t-1250:t]
    inv=np.linalg.inv(cov)
    rad0=np.einsum('ij,jk,ik->i',sample,inv,sample)
    radial_error=cov_error=orthogonal_error=gaussian_ridge_error=0.
    canonical_chol=np.linalg.cholesky(cov)
    for reference in m.CCY:
        a,others=m.basis_matrix(reference);nativecov=a@cov@a.T
        native=sample@a.T
        current=np.einsum('ij,jk,ik->i',native,np.linalg.inv(nativecov),native)
        radial_error=max(radial_error,float(np.max(abs(current-rad0))))
        recovered=np.linalg.inv(a)@nativecov@np.linalg.inv(a).T
        cov_error=max(cov_error,float(np.max(abs(recovered-cov))))
        equivalent_factor=np.linalg.solve(a,np.linalg.cholesky(nativecov))
        orthogonal=np.linalg.solve(canonical_chol,equivalent_factor)
        orthogonal_error=max(orthogonal_error,float(np.max(abs(orthogonal@orthogonal.T-np.eye(4)))))
        native_effective=nativecov+np.eye(4)*1e-14
        eur_effective=np.linalg.inv(a)@native_effective@np.linalg.inv(a).T
        gaussian_ridge_error=max(gaussian_ridge_error,float(np.max(abs(eur_effective-(cov+np.eye(4)*1e-14)))))
    checks['fullcov_radial_theoretical_law_closure']={'passed':radial_error<1e-8 and cov_error<1e-14 and orthogonal_error<1e-10,
      'maximum_Mahalanobis_squared_radius_error':radial_error,'maximum_covariance_recovery_error':cov_error,
      'maximum_orthogonal_coupling_identity_error':orthogonal_error,
      'interpretation':'Same covariance and radius distribution; this proves elliptical probability law closure, not equality of separately generated finite QMC paths.'}
    assert checks['fullcov_radial_theoretical_law_closure']['passed']
    checks['gaussian_isotropic_ridge_coordinate_departure']={'maximum_effective_covariance_difference':gaussian_ridge_error,
      'status':'P2 exact-closure qualification if each coordinate independently adds I*1e-14; canonical EUR-only generation does not leak.'}

    # Directly demonstrate that equal seeds do not create common physical paths.
    radial_eur=m.radial_paths(cov,np.sqrt(rad0),8,2026100212)
    a,others=m.basis_matrix('USD')
    radial_usd=m.radial_paths(a@cov@a.T,np.sqrt(rad0),8,2026100212)
    mapped=radial_usd@np.linalg.inv(a).T
    checks['independent_radial_MC_is_not_identical_coupling']={'maximum_path_difference':float(np.max(abs(mapped-radial_eur))),
      'passed':bool(np.max(abs(mapped-radial_eur))>1e-5),
      'interpretation':'Expected rotation/coupling difference. Must not attribute this finite simulation noise to non-equivariant probability law.'}
    assert checks['independent_radial_MC_is_not_identical_coupling']['passed']
    assert all(sha(ROOT/'scripts'/n)==h for n,h in script_hashes.items()),'Root scripts changed during audit; rerun to resolve audited version.'
    payload={'status':'passed_key_math_checks_with_P2_qualifications','no_model_retraining':True,
      'script_hashes':script_hashes,'pilot_prediction_sha256':sha(predpath),'checks':checks,
      'issues':[{'severity':'P2','issue':'Clock communicated as 18 Berlin but executed and manifest state planned18 UTC; synchronize protocol/text.'},
        {'severity':'P2','issue':'Gaussian isotropic ridge means exact independent-coordinate law closure is approximate; remove ridge on SPD or transform same perturbation.'}],
      'scope':'Read-only root script review plus output/sign/formula/prefix/control checks; no independent likelihood optimization, copula fitting or full retraining.'}
    serialized=json.dumps(payload,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))
    (OUT/'pilot_independent_checks.json').write_text(serialized+'\n',encoding='utf-8')
    print(serialized)

if __name__=='__main__':main()
