"""
Statistics of the SF-fit diagnostics. Every fit goes through fit_sfs.build_model and
fit_sfs.fit (the production likelihood, Barlow-Beeston-lite included); this module only
prepares the inputs, varies the configuration and summarizes the outputs.

Conventions
-----------
  T_ki, V_ki   sum of weights / of squared weights of component k = (process, era) in bin i
  N_eff        T^2 / V  (effective number of unweighted MC events)
  sigma_rel    error / value of a scale factor
  "Asimov"     the expected data nu(theta=1) fitted with the same model: its curvature is
               the expected information of the fit, free of the observed fluctuations
"""

import contextlib
import io
import zlib
from collections import OrderedDict

import numpy as np
from scipy import stats as sps

import fit_sfs
from src.yields import normalize_label

# Screening thresholds of the classifications (also written to the provenance block). They are
# analysis heuristics chosen for this diagnostic, not universal statistical criteria.
THRESHOLDS = OrderedDict([
    ('ident_rel', 0.25),      # sigma_rel below which a parameter counts as identified
    ('weak_rel', 0.5),        # sigma_rel above which a parameter counts as weakly constrained
    ('major_frac', 0.05),     # processes with at least this MC fraction are "major"
    ('high_corr', 0.9),       # |rho| above which two parameters are called degenerate
    ('neff_low', 10.0),       # N_eff below which an MC bin is called sparse
    ('relerr_high', 0.3),     # relative MC stat error above which a bin is called poorly modelled
    ('loo_shift', 1.0),       # leave-one-interval-out shift (in nominal sigma) flagged as unstable
    ('gof_p', 0.01),          # GoF / LRT p-value threshold
    ('prior_sigma', 0.5),     # log-normal width of the weak-process prior variant
    ('toy_bias', 0.3),        # |mean pull| flagged as biased
    ('toy_width', (0.8, 1.25)),  # pull-width range counted as correct coverage
    ('cond_max', 1e3),        # Asimov correlation-matrix condition number above which templates are near-degenerate
    ('gauss_tol', 0.2),       # max relative difference of the Gaussian-information and Hessian errors
    ('support_relerr_frac', 0.25),  # fraction of fitted bins with MC rel. error > relerr_high tolerated
    ('support_data_frac', 0.02),    # fraction of data in bins without MC tolerated
])


# --------------------------------------------------------------------------- #
# Templates                                                                   #
# --------------------------------------------------------------------------- #
def in_range(h):
    return h['w'][1:-1], h['w2'][1:-1]


def entry_templates(entry, eras=None):
    """{(proc, era): (T, V)} and {era: n} of the in-range bins."""
    T = OrderedDict(((p, e), in_range(h)) for (p, e), h in entry['bkg'].items() if not eras or e in eras)
    D = OrderedDict((e, in_range(h)[0]) for e, h in entry['data'].items() if not eras or e in eras)
    return T, D


def neff(t, v):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(v > 0, t ** 2 / np.where(v > 0, v, 1.0), 0.0)


def composition(entry, processes, eras):
    """Process yields, fractions, MC statistics and template support per era and summed."""
    T, D = entry_templates(entry)
    out = OrderedDict()
    for era in list(eras) + ['total']:
        sel = [k for k in T if era == 'total' or k[1] == era]
        if not sel:
            continue
        nb = len(next(iter(T.values()))[0])
        tot = np.zeros(nb)
        tot2 = np.zeros(nb)
        procs = OrderedDict()
        for p in processes:
            ks = [k for k in sel if k[0] == p]
            if not ks:
                continue
            t = sum(T[k][0] for k in ks)
            v = sum(T[k][1] for k in ks)
            procs[p] = (t, v)
            tot += t
            tot2 += v
        ytot = tot.sum()
        data = (sum(D.values()) if era == 'total' else D.get(era)) if D else None
        rows = OrderedDict()
        for p, (t, v) in procs.items():
            y = t.sum()
            ne = neff(t, v)
            pos = t > 0
            rows[p] = {
                'yield': float(y), 'sumw2': float(v.sum()),
                'fraction': float(y / ytot) if ytot > 0 else 0.0,
                'neff_total': float(y ** 2 / v.sum()) if v.sum() > 0 else 0.0,
                'rel_stat': float(np.sqrt(v.sum()) / abs(y)) if y != 0 else np.inf,
                'bins_filled': int(pos.sum()),
                'bins_sparse': int(((ne < THRESHOLDS['neff_low']) & pos).sum()),
                'bins_negative': int((t < 0).sum()),
                'negative_bin_sum': float(t[t < 0].sum()),
                'bin_fraction': (np.divide(t, tot, out=np.zeros_like(t), where=tot > 0)).tolist(),
                'bin_neff': ne.tolist(),
            }
        with np.errstate(divide='ignore', invalid='ignore'):
            relerr = np.where(tot > 0, np.sqrt(tot2) / tot, np.inf)
        info = {
            'processes': rows, 'mc_total': float(ytot), 'mc_rel_stat': float(np.sqrt(tot2.sum()) / ytot) if ytot > 0 else None,
            'bins_mc': int((tot > 0).sum()), 'bins_total': int(nb),
            'bins_relerr_high': int(((relerr > THRESHOLDS['relerr_high']) & (tot > 0)).sum()),
            'total_relerr': np.where(np.isfinite(relerr), relerr, -1).tolist(),
            'total_neff': neff(tot, tot2).tolist(),
            # under/overflow: not fitted (fit_sfs uses the in-range bins only)
            'mc_out_of_range': float(sum(h['w'][0] + h['w'][-1] for (p, e), h in entry['bkg'].items()
                                         if era == 'total' or e == era)),
        }
        if data is not None:
            unsup = (data > 0) & (tot <= 0)
            info.update({'data': float(data.sum()), 'data_over_mc': float(data.sum() / ytot) if ytot > 0 else None,
                         'data_bins_no_mc': int(unsup.sum()), 'data_events_no_mc': float(data[unsup].sum()),
                         'data_out_of_range': float(sum(h['w'][0] + h['w'][-1] for e, h in entry['data'].items()
                                                        if era == 'total' or e == era))})
        out[era] = info
    return out


def _norm(t, v):
    """
    (p, var_p, reason) of a template as a probability distribution, or (None, None, reason)
    when it is not one: the shape distances below assume non-negative contents with a
    positive integral. Signed (negative-weight) bins are never clipped away.

    var_p is the first-order (delta-method) variance of p_i = t_i / S with S = sum t, which
    includes the normalization correlation: (v_i (1 - 2 p_i) + p_i^2 sum_j v_j) / S^2.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    s = t.sum()
    if not np.isfinite(s) or s <= 0:
        return None, None, 'non-positive integral'
    if np.any(t < 0):
        return None, None, f"{int((t < 0).sum())} negative bin(s) (signed MC weights)"
    p = t / s
    return p, np.clip((v * (1 - 2 * p) + p ** 2 * v.sum()) / s ** 2, 0, None), None


def shape_compare(t1, v1, t2, v2):
    """
    Hellinger distance and a chi2 shape-compatibility test of two templates (MC stat only).
    Both need non-negative templates with a positive integral; otherwise status 'unavailable'
    with the reason (the signed yields stay in the composition table).
    """
    p, vp, why1 = _norm(t1, v1)
    q, vq, why2 = _norm(t2, v2)
    if p is None or q is None:
        reason = "; ".join(f"template {k}: {w}" for k, w in ((1, why1), (2, why2)) if w)
        return {'status': 'unavailable', 'reason': reason, 'hellinger': None, 'chi2': None, 'ndf': 0, 'p': None}
    h = float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(p * q)))))
    m = (p > 0) | (q > 0)
    var = vp + vq
    ok = m & (var > 0)
    chi2 = float(np.sum((p[ok] - q[ok]) ** 2 / var[ok]))
    ndf = int(ok.sum()) - 1
    return {'status': 'ok', 'reason': None, 'hellinger': h, 'chi2': chi2, 'ndf': ndf,
            'p': float(sps.chi2.sf(chi2, ndf)) if ndf > 0 else None}


def similarity(entry, processes, eras=None):
    """Pairwise shape comparison of the (era-summed) process templates (NaN = unavailable)."""
    T, _ = entry_templates(entry, eras)
    tv = OrderedDict()
    for p in processes:
        ks = [k for k in T if k[0] == p]
        if ks:
            tv[p] = (sum(T[k][0] for k in ks), sum(T[k][1] for k in ks))
    unavailable = OrderedDict()
    for p, (t, v) in tv.items():
        why = _norm(t, v)[2]
        if why:
            unavailable[p] = why
    names = [p for p, (t, _) in tv.items() if t.sum() > 0]
    H = np.full((len(names), len(names)), np.nan)
    P = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                r = {'hellinger': 0.0, 'p': 1.0} if a not in unavailable else {'hellinger': None, 'p': None}
            else:
                r = shape_compare(*tv[a], *tv[b])
            H[i, j] = r['hellinger'] if r['hellinger'] is not None else np.nan
            P[i, j] = r['p'] if r['p'] is not None else np.nan
    return {'processes': names, 'hellinger': H.tolist(), 'p_same_shape': P.tolist(), 'unavailable': unavailable}


def era_shape_compat(entry, processes):
    """Run 2 vs Run 3 shape of every process (the shared-era model assumes only a common normalization)."""
    T, D = entry_templates(entry)
    out = OrderedDict()
    for p in processes:
        if (p, 'run2') in T and (p, 'run3') in T:
            out[p] = shape_compare(*T[(p, 'run2')], *T[(p, 'run3')])
    if 'run2' in D and 'run3' in D:
        d2, d3 = D['run2'], D['run3']
        out['data'] = shape_compare(d2, d2.copy(), d3, d3.copy())
    return out


# --------------------------------------------------------------------------- #
# Fits through fit_sfs                                                         #
# --------------------------------------------------------------------------- #
class SelectivePriorModel(fit_sfs.Model):
    """fit_sfs.Model with a log-normal prior on selected parameters only (fit_sfs's --prior
    constrains every parameter). Used for the weak-process constraint variant."""

    def __init__(self, base, prior_params, sigma):
        super().__init__(base.params, base.comp_param, base.comp_fixed, base.T, base.V, base.n,
                         base.bin_channel, None, base.mc_stat)
        self.prior_idx = np.array([base.params.index(p) for p in prior_params], dtype=int)
        self.prior_sigma = float(sigma)

    def deviance(self, theta):
        d = super().deviance(theta)
        if len(self.prior_idx):
            with np.errstate(invalid='ignore', divide='ignore'):  # Nelder-Mead may probe theta <= 0
                d += np.sum(np.log(theta[self.prior_idx]) ** 2) / self.prior_sigma ** 2
        return d


_POOL = {'n': 1, 'ex': None}


def set_workers(n):
    """Worker processes for the refits of stability scans and toys (1 = in process)."""
    shutdown_workers()
    _POOL['n'] = max(1, int(n or 1))


def shutdown_workers():
    if _POOL['ex'] is not None:
        _POOL['ex'].shutdown()
        _POOL['ex'] = None


def _fit_one(model):
    """fit_sfs.fit of one model: (theta, err, singular, converged, at_bound, bin deviance)."""
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            th, cov, sing, ok, ab = fit_sfs.fit(model)
        except Exception:
            return None
    return th, np.sqrt(np.clip(np.diag(cov), 0, None)), bool(sing), bool(ok), ab, float(model.bin_deviance(th).sum())


def map_fits(models):
    """fit_sfs.fit over many models, in parallel when set_workers(n > 1). Order preserved."""
    if _POOL['n'] > 1 and len(models) > 1:
        if _POOL['ex'] is None:
            from concurrent.futures import ProcessPoolExecutor
            _POOL['ex'] = ProcessPoolExecutor(max_workers=_POOL['n'])
        return list(_POOL['ex'].map(_fit_one, models, chunksize=max(1, len(models) // (4 * _POOL['n']))))
    return [_fit_one(m) for m in models]


def copy_model(model, T=None, V=None, n=None, keep=None):
    T = model.T if T is None else T
    V = model.V if V is None else V
    n = model.n if n is None else n
    bc = model.bin_channel
    if keep is not None:
        T, V, n, bc = T[:, keep], V[:, keep], n[keep], bc[keep]
    return fit_sfs.Model(model.params, list(model.comp_param), list(model.comp_fixed), T, V, n, bc,
                         model.prior, model.mc_stat)


def _channel_sums(entry, rebin, var):
    """Template sums per (proc, era) inside the (rebinned) fit range."""
    first = next(iter(entry['data'].values())) if entry['data'] else next(iter(entry['bkg'].values()))
    M = fit_sfs.rebin_matrix(first, fit_sfs.rebin_for(rebin, var))
    out = {}
    for k, h in entry['bkg'].items():
        out[k] = float(fit_sfs.hist_arrays(h, M)[0].sum())
    return out


def param_structure(entry, processes, era_model, rebin, var, fix=(), production_per_era=None):
    """
    per_era list and auto-fixed processes for an era model.

      shared       one SF per process for both eras (the Poisson terms stay per era)
      independent  one SF per process and era for every process with MC in the fit range
      production   the per_era list of the config's sf_fit block
    Processes without MC in the fit range are fixed (they cannot be measured).
    """
    sums = _channel_sums(entry, rebin, var)
    absent = [p for p in processes if not any(v > 0 for (pp, _), v in sums.items() if pp == p)]
    if era_model == 'shared':
        per_era = []
    elif era_model == 'independent':
        per_era = [p for p in processes if p not in absent and
                   all(v > 0 for (pp, _), v in sums.items() if pp == p)]
    else:
        per_era = list(production_per_era or [])
        if any(normalize_label(x) == 'all' for x in per_era):
            per_era = [p for p in processes if p not in absent]
        per_era = [p for p in per_era if p not in absent and
                   all(v > 0 for (pp, _), v in sums.items()
                       if normalize_label(pp) == normalize_label(p))]
    fixed = OrderedDict((p, 1.0) for p in absent)
    for spec in fix:
        name, val = (spec.rsplit('=', 1) + ['1'])[:2] if '=' in spec else (spec, '1')
        fixed[fit_sfs.match_process(name, processes)] = float(val)
    return per_era, fixed, absent


def fit_channel(meta, hists, region, var, era_model='shared', fix=(), rebin=None, prior_procs=(),
                prior_sigma=None, mc_stat=True, production_per_era=None, label=None):
    """
    One production fit of one channel. Returns a result dict (plus the fit_sfs.Model under
    'model', not serialized). Failures are returned as {'status': 'failed', 'error': ...}.
    """
    entry = hists[(region, var)]
    procs = meta['processes']
    res = OrderedDict(label=label or era_model, region=region, var=var, era_model=era_model,
                      rebin=rebin, mc_stat=mc_stat, fix_requested=list(fix), prior_procs=list(prior_procs),
                      prior_sigma=prior_sigma if prior_procs else None)
    try:
        per_era, fixed, absent = param_structure(entry, procs, era_model, rebin, var, fix, production_per_era)
        args = fit_sfs.fit_options(None, per_era=list(per_era) if per_era else [],
                                   fix=[f"{p}={v}" for p, v in fixed.items()], rebin=rebin,
                                   no_mc_stat=None if mc_stat else True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            model, chan_labels, split, fixed_out, per_era_set = fit_sfs.build_model(meta, hists, [(region, var)], args)
        res['messages'] = [l for l in buf.getvalue().splitlines() if l.strip()]
        if prior_procs:
            pp = [p for p in model.params if p.partition(':')[0] in set(prior_procs) or p in set(prior_procs)]
            model = SelectivePriorModel(model, pp, prior_sigma or THRESHOLDS['prior_sigma'])
        with contextlib.redirect_stdout(io.StringIO()):
            theta, cov, singular, ok, at_bound = fit_sfs.fit(model)
    except fit_sfs.FitError as e:
        res.update(status='failed', error=str(e))
        return res
    err = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = cov / np.outer(err, err)
    ones = np.ones(len(theta))
    pre = model.bin_deviance(ones)
    post = model.bin_deviance(theta)
    dev = float(post.sum())
    prior_term = float(model.deviance(theta) - dev)
    nbins = len(model.n)
    ndf = nbins - len(theta)

    # bins and data outside the fit
    first = next(iter(entry['data'].values()))
    M = fit_sfs.rebin_matrix(first, fit_sfs.rebin_for(rebin, var))
    nb_after = len(fit_sfs.hist_arrays(first, M)[0])
    n_era_groups = len(chan_labels)
    data_total = sum(fit_sfs.hist_arrays(h)[0].sum() for e, h in entry['data'].items())
    s = model.scales(theta)
    nu = s @ model.T
    mcvar = (s ** 2) @ model.V
    with np.errstate(divide='ignore', invalid='ignore'):
        resid = (model.n - nu) / np.sqrt(np.maximum(nu + (mcvar if mc_stat else 0), 1e-12))
    res.update(
        status='ok', params=list(model.params), theta=theta, err=np.where(np.array([b is None for b in at_bound]), err, np.nan),
        cov=cov, corr=corr, at_bound=list(at_bound), singular=bool(singular), converged=bool(ok),
        fixed=OrderedDict((k, float(v)) for k, v in fixed_out.items()), absent=absent, per_era=list(per_era),
        split=bool(split), channels=chan_labels, deviance=dev, prior_term=prior_term, deviance_pre=float(pre.sum()),
        nbins=nbins, nbins_total=nb_after * n_era_groups, nbins_excluded=nb_after * n_era_groups - nbins,
        ndf=ndf, p_gof=float(sps.chi2.sf(dev, ndf)) if ndf > 0 else None,
        data_total=float(data_total), data_fitted=float(model.n.sum()),
        data_excluded=float(data_total - model.n.sum()),
        bin_dev_pre=pre, bin_dev_post=post, bin_channel=list(model.bin_channel), residual=resid,
        nu_post=nu, mc_var_post=mcvar,
        model=model,
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        res['shift_from_nominal'] = (theta - 1.0) / res['err']
    res['identifiability'] = identifiability(res)
    return res


def identifiability(res):
    """
    Per-parameter flags of an observed fit. The Hessian error is only meaningful for a
    converged fit with a regular (non-singular) Hessian and the parameter away from its
    bound; otherwise the parameter is never 'identified' (it is 'weak' instead).
    """
    out = OrderedDict()
    corr = res['corr']
    regular = not res['singular'] and res.get('converged', True)
    for i, p in enumerate(res['params']):
        v, e, b = res['theta'][i], res['err'][i], res['at_bound'][i]
        row = np.abs(np.where(np.isfinite(corr[i]), corr[i], 0.0))
        row[i] = 0.0
        j = int(np.argmax(row)) if len(row) > 1 else None
        rel = float(e / v) if np.isfinite(e) and v > 0 else np.inf
        out[p] = {'rel_err': rel, 'at_bound': b, 'max_corr': float(row[j]) if j is not None else 0.0,
                  'max_corr_with': res['params'][j] if j is not None else None,
                  'regular': bool(regular and b is None),
                  'identified': bool(regular and b is None and rel < THRESHOLDS['ident_rel']),
                  'weak': bool(not regular or b is not None or rel > THRESHOLDS['weak_rel'])}
    return out


def sf_lookup(res, processes, eras):
    """{(proc, era): SF} of a fit result (fixed / absent processes included)."""
    out = {}
    for p in processes:
        for e in eras:
            out[(p, e)] = res['fixed'].get(p, 1.0)
    for name, v in zip(res['params'], res['theta']):
        proc, _, era = name.partition(':')
        if proc not in processes:  # process label containing ':'
            proc, era = name, ''
        for e in eras:
            if not era or era == e:
                out[(proc, e)] = float(v)
    return out


def lrt(restricted, full):
    """
    Likelihood-ratio test of nested fits: delta(-2lnL) ~ chi2(delta k) under the restricted
    model (Wilks). The p-value is only interpretable ('interpretable': True) when both fits
    converged with regular Hessians, no parameter sits at an SF bound (Wilks needs interior
    points), delta(-2lnL) >= 0 (a negative value means a failed minimization of the larger
    model) and the larger model itself fits (GoF p >= gof_p: with both models misspecified
    the deviance difference is not chi2-distributed). 'caveats' lists the failed conditions.
    """
    if restricted.get('status') != 'ok' or full.get('status') != 'ok':
        return None
    d = restricted['deviance'] - full['deviance']
    k = len(full['params']) - len(restricted['params'])
    if k <= 0:
        return None
    caveats = []
    for lab, r in (('restricted', restricted), ('full', full)):
        if not r.get('converged', True):
            caveats.append(f"{lab} fit did not converge")
        if r.get('singular'):
            caveats.append(f"{lab} fit has a singular Hessian")
        bnd = [p for p, b in zip(r['params'], r['at_bound']) if b]
        if bnd:
            caveats.append(f"{lab} fit has {', '.join(bnd)} at an SF bound")
    if d < -1e-6 * max(1.0, abs(restricted['deviance'])):
        caveats.append(f"delta(-2lnL) = {d:.3g} < 0: the larger model was not minimized")
    if full.get('p_gof') is not None and full['p_gof'] < THRESHOLDS['gof_p']:
        caveats.append(f"the larger model does not fit (GoF p = {full['p_gof']:.2g}): both models misspecified")
    return {'delta_deviance': float(d), 'delta_k': int(k), 'p': float(sps.chi2.sf(max(d, 0.0), k)),
            'aic_full_minus_restricted': float(-d + 2 * k),
            'interpretable': not caveats, 'caveats': caveats}


def era_ratio_tests(res):
    """Per process with two era parameters: z of SF(run3) - SF(run2) with their covariance."""
    out = OrderedDict()
    idx = {p: i for i, p in enumerate(res['params'])}
    for p in res['params']:
        proc, _, era = p.partition(':')
        if era != 'run2' or f"{proc}:run3" not in idx:
            continue
        i, j = idx[p], idx[f"{proc}:run3"]
        d = res['theta'][j] - res['theta'][i]
        var = res['cov'][i, i] + res['cov'][j, j] - 2 * res['cov'][i, j]
        z = d / np.sqrt(var) if np.isfinite(var) and var > 0 else np.nan
        out[proc] = {'run2': float(res['theta'][i]), 'run3': float(res['theta'][j]), 'diff': float(d),
                     'z': float(z), 'p': float(2 * sps.norm.sf(abs(z))) if np.isfinite(z) else None}
    return out


# --------------------------------------------------------------------------- #
# Expected information (Asimov)                                               #
# --------------------------------------------------------------------------- #
def asimov_information(model, theta0=None):
    """
    Expected information of a model at theta0 (default 1):
      cov_asimov      2 H^-1 of the production deviance on Asimov data (numeric Hessian)
      info / profiled Gaussian approximation I = G^T W G with W = 1 / (nu + sigma_MC^2);
                      'unprofiled' I_pp (information if all other SFs were known) vs.
                      'profiled' 1/(I^-1)_pp (after the others are fitted). Their ratio is the
                      fraction of the information that separates p from the other processes.
      per-bin         each bin's share of the profiled information (sums exactly to it)
    """
    P = len(model.params)
    theta0 = np.ones(P) if theta0 is None else np.asarray(theta0, dtype=float)
    s = model.scales(theta0)
    nu = s @ model.T
    am = copy_model(model, n=nu.copy())
    if isinstance(model, SelectivePriorModel):
        am = SelectivePriorModel(am, [model.params[i] for i in model.prior_idx], model.prior_sigma)
    H = fit_sfs.numeric_hessian(am.deviance, theta0)
    singular = False
    try:
        cov = 2.0 * np.linalg.inv(H)
        singular = bool(np.any(np.diag(cov) <= 0) or not np.all(np.isfinite(cov)))
    except np.linalg.LinAlgError:
        cov, singular = 2.0 * np.linalg.pinv(H), True
    sig = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = cov / np.outer(sig, sig)

    G = np.zeros((len(nu), P))
    for k, p in enumerate(model.comp_param):
        if p >= 0:
            G[:, p] += model.T[k]
    var = nu + ((s ** 2) @ model.V if model.mc_stat else 0.0)
    var = np.maximum(var, 1e-12)
    I = G.T @ (G / var[:, None])
    unprof = np.diag(I).copy()
    prof = np.zeros(P)
    per_bin = np.zeros((P, len(nu)))
    for p in range(P):
        o = [q for q in range(P) if q != p]
        if o:
            try:
                b = np.linalg.solve(I[np.ix_(o, o)], I[o, p])
            except np.linalg.LinAlgError:
                b = np.linalg.lstsq(I[np.ix_(o, o)], I[o, p], rcond=None)[0]
            r = G[:, p] - G[:, o] @ b
        else:
            r = G[:, p]
        per_bin[p] = r ** 2 / var
        prof[p] = per_bin[p].sum()

    finite = np.all(np.isfinite(corr))
    eig, vec = (np.linalg.eigh(np.where(np.isfinite(corr), corr, 0.0)) if P > 1 else (np.ones(1), np.ones((1, 1))))
    order = np.argsort(eig)
    eig, vec = eig[order], vec[:, order]
    cond = float(eig[-1] / eig[0]) if eig[0] > 1e-12 else np.inf
    with np.errstate(invalid='ignore', divide='ignore'):
        rel = sig / theta0
        # cross-check of the two approximations: Hessian of the production deviance vs I = G^T W G
        gauss_sig = np.where(prof > 0, 1.0 / np.sqrt(prof), np.inf)
        gauss_diff = np.abs(sig / gauss_sig - 1.0)
    gauss_max = float(np.nanmax(gauss_diff)) if P and np.any(np.isfinite(gauss_diff)) else np.inf
    singular = singular or not finite
    near_degenerate = bool(cond > THRESHOLDS['cond_max'])
    return OrderedDict(
        params=list(model.params), sigma=sig, rel=rel, cov=cov, corr=corr, singular=singular,
        eigenvalues=eig, weakest_direction=OrderedDict((model.params[i], float(vec[i, 0])) for i in range(P)),
        condition_number=cond, near_degenerate=near_degenerate, info_unprofiled=unprof, info_profiled=prof,
        separation_fraction=np.divide(prof, unprof, out=np.zeros(P), where=unprof > 0),
        gauss_sigma=gauss_sig, gauss_hessian_max_rel_diff=gauss_max,
        gauss_consistent=bool(gauss_max <= THRESHOLDS['gauss_tol']),
        # regular asymptotics (errors from the curvature) are only trusted when all hold
        regular=bool(not singular and not near_degenerate and gauss_max <= THRESHOLDS['gauss_tol']),
        per_bin_profiled=per_bin, per_bin_unprofiled=(G ** 2 / var[:, None]).T,
        bin_channel=list(model.bin_channel),
    )


def asimov_summary(info, processes_major):
    """
    Scalars used for the variable ranking. With a singular expected Hessian no parameter
    counts as identified (the curvature errors are undefined).
    """
    rel = np.asarray(info['rel'])
    ident = [] if info['singular'] else \
        [p for p, r in zip(info['params'], rel) if np.isfinite(r) and r < THRESHOLDS['ident_rel']]
    major = [i for i, p in enumerate(info['params']) if p.partition(':')[0] in processes_major or p in processes_major]
    corr = np.abs(np.asarray(info['corr'], dtype=float))
    if corr.size > 1:
        np.fill_diagonal(corr, 0.0)
        max_corr = float(np.nanmax(corr))
    else:
        max_corr = 0.0
    med = float(np.median(rel[major])) if major else (float(np.median(rel)) if len(rel) else np.inf)
    return {'n_identified': len(ident), 'identified': ident, 'n_params': len(rel),
            'median_rel_major': med, 'max_rel_major': float(np.max(rel[major])) if major else None,
            'max_corr': max_corr, 'condition_number': info['condition_number'], 'singular': info['singular'],
            'near_degenerate': info['near_degenerate'], 'regular': info['regular'],
            'gauss_hessian_max_rel_diff': info['gauss_hessian_max_rel_diff']}


# --------------------------------------------------------------------------- #
# Stability                                                                   #
# --------------------------------------------------------------------------- #
def bin_index_map(meta, hists, region, var, rebin=None, min_mc=0.0):
    """
    Variable-bin index and era of every bin of the fit_sfs model of one channel,
    in build_model's order (era groups in meta['eras'] order, bins without MC dropped).
    """
    entry = hists[(region, var)]
    eras = meta['eras']
    data_eras = set(entry['data'])
    split = set(eras) <= data_eras and 'all' not in eras
    first = next(iter(entry['data'].values()))
    M = fit_sfs.rebin_matrix(first, fit_sfs.rebin_for(rebin, var))
    idx, era_of, nominal = [], [], []
    for group in ([[e] for e in eras] if split else [eras]):
        if not any(e in group for e in entry['data']):
            continue
        nb = len(fit_sfs.hist_arrays(first, M)[0])
        nom = np.zeros(nb)
        for (p, e), h in entry['bkg'].items():
            if e in group:
                nom += fit_sfs.hist_arrays(h, M)[0]
        idx.extend(range(nb))
        era_of.extend([group[0] if split else 'all'] * nb)
        nominal.append(nom)
    nominal = np.concatenate(nominal)
    keep = nominal > min_mc
    return np.array(idx)[keep], np.array(era_of)[keep]


def _params_supported(model, keep):
    T = model.T[:, keep]
    for p in range(len(model.params)):
        if not any(T[k].sum() > 0 for k in range(len(model.comp_param)) if model.comp_param[k] == p):
            return False
    return True


def _submodel(model, keep=None, T=None, n=None, V=None):
    sub = copy_model(model, T=T, V=V, n=n, keep=keep)
    if isinstance(model, SelectivePriorModel):
        sub = SelectivePriorModel(sub, [model.params[i] for i in model.prior_idx], model.prior_sigma)
    return sub


def stability_scan(res, meta, hists):
    """
    Leave-one-interval-out (one variable bin, both eras), four contiguous MC-yield quarters
    left out in turn, and low- vs high-statistics halves fitted alone. Shifts are in units
    of the nominal error.
    """
    model, theta0, err0 = res['model'], res['theta'], res['err']
    vidx, _ = bin_index_map(meta, hists, res['region'], res['var'], res['rebin'])
    if len(vidx) != len(model.n):
        return {'status': 'failed', 'error': 'bin map mismatch'}
    bins = np.unique(vidx)
    mc = model.T.sum(axis=0)
    mc_per_vbin = np.array([mc[vidx == b].sum() for b in bins])
    out = OrderedDict(status='ok', params=res['params'], bins=bins.tolist(), loo=[], groups=[], halves=[])

    def shifts(theta):
        with np.errstate(invalid='ignore', divide='ignore'):
            return ((theta - theta0) / err0).tolist()

    if any(res['at_bound']) or res['singular']:
        out.update(status='skipped', reason='nominal fit has parameters at a bound or a singular Hessian: '
                                            'shifts in units of the nominal error are undefined')
        return out
    jobs = []   # (list name, entry, keep)
    for b in bins:
        keep = vidx != b
        e = {'bin': int(b)}
        out['loo'].append(e)
        if keep.sum() <= len(theta0) or not _params_supported(model, keep):
            e['status'] = 'unsupported'
        else:
            jobs.append((e, keep))
    cum = np.cumsum(mc_per_vbin) / max(mc_per_vbin.sum(), 1e-12)
    q = np.minimum((cum * 4 - 1e-9).astype(int), 3)
    for g in range(4):
        gb = bins[q == g]
        if not len(gb):
            continue
        keep = ~np.isin(vidx, gb)
        e = {'group': g, 'bins': [int(gb[0]), int(gb[-1])]}
        out['groups'].append(e)
        if keep.sum() > len(theta0) and _params_supported(model, keep):
            jobs.append((e, keep))
        else:
            e['status'] = 'unsupported'
    med = np.median(mc_per_vbin)
    for name, sel in (('low-stat bins only', bins[mc_per_vbin <= med]), ('high-stat bins only', bins[mc_per_vbin > med])):
        keep = np.isin(vidx, sel)
        e = {'subset': name, 'n_bins': int(keep.sum())}
        out['halves'].append(e)
        if keep.sum() > len(theta0) and _params_supported(model, keep):
            jobs.append((e, keep))
        else:
            e['status'] = 'unsupported'
    fitted = map_fits([_submodel(model, keep=k) for _, k in jobs])
    for (e, _), fr in zip(jobs, fitted):
        if fr is None:
            e['status'] = 'failed'
            continue
        th, er, sing, ok, ab, _dev = fr
        e.update(theta=th.tolist(), err=er.tolist(), shift=shifts(th), at_bound=ab, singular=sing)

    # summary: largest leave-one-out shift per parameter and the bin responsible
    P = len(theta0)
    mx = np.zeros(P)
    arg = [None] * P
    for e in out['loo']:
        if 'shift' not in e:
            continue
        for i, sft in enumerate(e['shift']):
            if np.isfinite(sft) and abs(sft) > mx[i]:
                mx[i], arg[i] = abs(sft), e['bin']
    out['max_loo_shift'] = mx.tolist()
    out['max_loo_bin'] = arg
    gmx = np.zeros(P)
    for e in out['groups']:
        for i, sft in enumerate(e.get('shift', [])):
            if np.isfinite(sft):
                gmx[i] = max(gmx[i], abs(sft))
    out['max_group_shift'] = gmx.tolist()
    return out


def adaptive_edges(entry, var, min_neff=20.0, max_relerr=0.2, min_mc=1.0):
    """
    Edges merging adjacent bins (left to right, eras and processes summed) until the merged MC
    bin has N_eff >= min_neff, relative stat error <= max_relerr and yield >= min_mc. The
    last merged bin absorbs an incomplete remainder. Only existing edges are used.
    """
    T, _ = entry_templates(entry)
    t = sum(v[0] for v in T.values())
    v = sum(v[1] for v in T.values())
    edges = next(iter(entry['bkg'].values()))['edges']
    out = [edges[0]]
    acc_t = acc_v = 0.0
    for i in range(len(t)):
        acc_t += t[i]
        acc_v += v[i]
        ok = acc_t >= min_mc and acc_v > 0 and acc_t ** 2 / acc_v >= min_neff and np.sqrt(acc_v) / acc_t <= max_relerr
        if ok:
            out.append(edges[i + 1])
            acc_t = acc_v = 0.0
    if out[-1] != edges[-1]:
        if len(out) > 1:
            out[-1] = edges[-1]
        else:
            out.append(edges[-1])
    return [float(e) for e in out]


def binning_variants(entry, var, nbins):
    variants = OrderedDict([('nominal', None)])
    if nbins >= 4:
        variants['merge2'] = 2
    if nbins >= 10:
        variants['merge5'] = 5
    ad = adaptive_edges(entry, var)
    if len(ad) >= 3:
        variants['adaptive'] = ad
    return variants


# --------------------------------------------------------------------------- #
# Toys                                                                        #
# --------------------------------------------------------------------------- #
TOY_ENSEMBLES = OrderedDict([
    ('conditional',
     "templates fixed at their nominal MC values (treated as the truth), data ~ Poisson(nu(theta_true)). "
     "Tests the estimator given the MC. With MC stat on, the BB-lite terms add uncertainty that is "
     "not fluctuated here: pull widths < 1 (over-coverage) are expected, and the deviance distribution "
     "is too narrow to calibrate the observed GoF."),
    ('bb_lite',
     "data ~ Poisson(nu(theta_true)) and, per bin, the total MC rescaled by one factor g_i ~ Gamma with "
     "mean 1 and relative width d_i = sqrt(sum s^2 V) / nu (all processes coherently): the generative "
     "model of the Barlow-Beeston-lite likelihood (one nuisance per bin on the total MC). Model-consistent "
     "ensemble for closure, coverage and GoF calibration of the production likelihood."),
    ('mc_template',
     "data ~ Poisson(nu(theta_true)) and every process template fluctuated independently per bin, "
     "T'_ki ~ Gamma with mean T_ki and variance V_ki (the actual sampling of independent MC samples). "
     "Checks the BB-lite approximation; bins with T_ki <= 0 or V_ki <= 0 are not fluctuated (counted)."),
])


def _gamma_scale(rng, mean, var):
    """Gamma draws with the given mean (> 0) and variance (> 0), elementwise (no clipping needed)."""
    k = mean ** 2 / var
    return rng.gamma(k, var / mean)


def toy_injection(model, theta_true):
    """
    Validation of an injection point against the production likelihood: finite values inside
    the SF bounds [fit_sfs.SF_MIN, fit_sfs.SF_MAX]. Bins with a negative expectation (signed
    templates) are generated with mean 0, as the likelihood floors nu at 1e-12; their number
    is reported.
    """
    tt = np.asarray(theta_true, dtype=float)
    out = OrderedDict(theta_true=tt.tolist(), valid=True, reasons=[])
    if tt.shape != (len(model.params),):
        out['valid'] = False
        out['reasons'].append(f"{tt.size} values for {len(model.params)} parameters")
        return out, None
    if not np.all(np.isfinite(tt)):
        out['valid'] = False
        out['reasons'].append("non-finite injected value")
    out_of_bounds = [p for p, v in zip(model.params, tt) if not (fit_sfs.SF_MIN <= v <= fit_sfs.SF_MAX)]
    if out_of_bounds:
        out['valid'] = False
        out['reasons'].append(f"outside the SF bounds [{fit_sfs.SF_MIN:g}, {fit_sfs.SF_MAX:g}]: "
                              + ", ".join(out_of_bounds))
    if not out['valid']:
        return out, None
    mu = model.scales(tt) @ model.T
    out['bins_negative_expectation'] = int((mu < 0).sum())
    out['negative_expectation_sum'] = float(mu[mu < 0].sum())
    out['near_bound'] = [p for p, v in zip(model.params, tt)
                         if v <= fit_sfs.SF_MIN * 1.01 or v >= fit_sfs.SF_MAX * 0.99]
    return out, np.maximum(mu, 0.0)


def _toy_models(model, tt, mu, n_toys, rng, ensemble):
    """Pseudo-experiment models of one ensemble (see TOY_ENSEMBLES)."""
    gen = OrderedDict()
    s = model.scales(tt)
    nu = s @ model.T
    d2 = np.where(nu > 0, (s ** 2) @ model.V / np.where(nu > 0, nu, 1.0) ** 2, 0.0)
    fluct = (d2 > 0) & (nu > 0) if model.mc_stat else np.zeros(len(nu), bool)
    if ensemble == 'bb_lite':
        gen['bins_mc_fluctuated'] = int(fluct.sum())
        gen['bins_mc_not_fluctuated'] = int((~fluct).sum())
    elif ensemble == 'mc_template':
        ok = (model.T > 0) & (model.V > 0)
        gen['cells_fluctuated'] = int(ok.sum())
        gen['cells_not_fluctuated'] = int((~ok & (model.T != 0)).sum())
    models = []
    for _ in range(n_toys):
        n = rng.poisson(mu).astype(float)
        T, V = model.T, model.V
        if ensemble == 'bb_lite' and fluct.any():
            g = np.ones(len(nu))
            g[fluct] = _gamma_scale(rng, np.ones(int(fluct.sum())), d2[fluct])
            T, V = model.T * g, model.V * g ** 2
        elif ensemble == 'mc_template' and model.mc_stat:
            ok = (model.T > 0) & (model.V > 0)
            T = model.T.copy()
            T[ok] = _gamma_scale(rng, model.T[ok], model.V[ok])
        models.append(_submodel(model, T=T, n=n, V=V))
    return models, gen


def run_toys(res, n_toys, seed, theta_true=None, ensemble='bb_lite', stress_label=None):
    """
    Pseudo-experiments refitted with the production fit (fit_sfs.fit), for one ensemble of
    TOY_ENSEMBLES. theta_true defaults to the observed best fit and is used exactly as given:
    an injection outside the SF bounds (or otherwise invalid) is refused (status 'invalid'),
    never moved. A deliberately modified injection point is a stress test: pass stress_label.

    Toy fits are classified as failed (exception), not converged, singular, or with a
    parameter at a bound. Bias / spread use the converged fits; pulls and coverage use the
    converged non-singular fits with the parameter off its bound; the GoF uses the converged
    fits. All fractions are reported.
    """
    if ensemble not in TOY_ENSEMBLES:
        raise ValueError(f"unknown toy ensemble {ensemble}")
    model = res['model']
    source = 'observed best fit' if theta_true is None else 'requested'
    inj, mu = toy_injection(model, res['theta'] if theta_true is None else theta_true)
    out = OrderedDict(status='ok', ensemble=ensemble, ensemble_definition=TOY_ENSEMBLES[ensemble],
                      n_toys=n_toys, seed=seed, params=list(model.params), injection_source=source,
                      stress_test=stress_label, theta_true=inj['theta_true'], injection=inj)
    if not inj['valid']:
        out.update(status='invalid', reason="; ".join(inj['reasons']))
        return out
    tt = np.asarray(inj['theta_true'], dtype=float)
    rng = np.random.default_rng(seed)
    models, gen = _toy_models(model, tt, mu, n_toys, rng, ensemble)
    out['generation'] = gen
    rows = map_fits(models)

    failed = [r is None for r in rows]
    conv = [r is not None and r[3] and np.all(np.isfinite(r[0])) for r in rows]
    out['n_failed'] = int(sum(failed))
    out['n_not_converged'] = int(sum(1 for r, c in zip(rows, conv) if r is not None and not c))
    good = [r for r, c in zip(rows, conv) if c]
    out['n_valid'] = len(good)
    if not good:
        out.update(status='no valid fits')
        return out
    TH = np.array([r[0] for r in good])
    ER = np.array([r[1] for r in good])
    SG = np.array([r[2] for r in good])
    AB = np.array([[b is not None for b in r[4]] for r in good])
    DV = np.array([r[5] for r in good])
    out['n_singular'] = int(SG.sum())
    out['frac_singular'] = float(SG.mean())
    out['frac_not_converged'] = float(out['n_not_converged'] / max(1, n_toys - out['n_failed']))
    out['frac_failed'] = float(out['n_failed'] / max(1, n_toys))
    out['frac_any_bound'] = float(AB.any(axis=1).mean())
    pull_ok = ~AB & ~SG[:, None] & np.isfinite(ER) & (ER > 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        pull = np.where(pull_ok, (TH - tt) / ER, np.nan)
    out['per_param'] = OrderedDict()
    for i, p in enumerate(model.params):
        pc = pull[:, i][np.isfinite(pull[:, i])]
        sd = float(TH[:, i].std(ddof=1)) if len(TH) > 1 else None
        cov = float(np.mean(np.abs(pc) < 1.0)) if len(pc) else None
        out['per_param'][p] = {
            'mean': float(TH[:, i].mean()), 'std': sd,
            'bias': float(TH[:, i].mean() - tt[i]),
            'rel_bias': float((TH[:, i].mean() - tt[i]) / tt[i]),
            'bias_over_std': float((TH[:, i].mean() - tt[i]) / sd) if sd else None,
            'median_err': float(np.median(ER[pull_ok[:, i], i])) if pull_ok[:, i].any() else None,
            'n_pulls': int(len(pc)),
            'pull_mean': float(pc.mean()) if len(pc) else None,
            'pull_mean_err': float(pc.std(ddof=1) / np.sqrt(len(pc))) if len(pc) > 1 else None,
            'pull_width': float(pc.std(ddof=1)) if len(pc) > 1 else None,
            'coverage_68': cov,
            'coverage_68_err': float(np.sqrt(cov * (1 - cov) / len(pc))) if len(pc) else None,
            'frac_at_bound': float(AB[:, i].mean()),
            'pulls': pc.tolist(),
        }
    out['deviance'] = DV.tolist()
    ndf = res.get('ndf') or 0
    gof = OrderedDict(ndf=ndf, mean_deviance=float(DV.mean()))
    if ndf > 0:
        p_asym = sps.chi2.sf(DV, ndf)
        gof['mean_over_ndf'] = float(DV.mean() / ndf)
        gof['frac_asymptotic_p_below_0p05'] = float(np.mean(p_asym < 0.05))   # 0.05 if chi2(ndf) holds
        gof['ks_p_vs_chi2_ndf'] = float(sps.kstest(DV, 'chi2', args=(ndf,)).pvalue) if len(DV) > 4 else None
    obs = res['deviance']
    n_ge = int(np.sum(DV >= obs))
    gof['p_observed'] = float((n_ge + 1) / (len(DV) + 1))        # conservative: never 0
    gof['p_observed_resolution'] = float(1.0 / (len(DV) + 1))
    gof['calibrated'] = ensemble in ('bb_lite', 'mc_template') and stress_label is None and source != 'requested'
    gof['note'] = ("toy p-value of the observed -2lnL in this ensemble" if gof['calibrated'] else
                   "NOT a calibrated GoF p-value of the observed data (ensemble or injection point not the "
                   "fitted model)")
    out['gof'] = gof
    out['p_gof_toys'] = gof['p_observed'] if gof['calibrated'] else None
    out['closure_ok'] = bool(all(
        (v['pull_mean'] is None or abs(v['pull_mean']) < THRESHOLDS['toy_bias']) and
        (v['pull_width'] is None or THRESHOLDS['toy_width'][0] <= v['pull_width'] <= THRESHOLDS['toy_width'][1]) and
        v['frac_at_bound'] < 0.05 for v in out['per_param'].values()) and out['frac_singular'] < 0.05
        and out['frac_failed'] + out['frac_not_converged'] < 0.05)
    return out


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def apply_sfs(entry, sfs, eras, chi2_fn):
    """
    Data vs MC with SFs, per era and for all eras summed ('total'): in-range yields and the
    Pearson chi2 of replot_yields.chi2_ndf. MC contents scale with SF, MC variances with SF^2.
    """
    from src.sfdiag.inputs import th1_from
    out = OrderedDict()
    groups = [(e, [e]) for e in eras] + ([('total', list(eras))] if len(eras) > 1 else [])
    for tag, group in groups:
        data = [h for e, h in entry['data'].items() if e in group]
        if not data:
            continue
        dsum = {'edges': data[0]['edges'], 'w': sum(h['w'] for h in data), 'w2': sum(h['w2'] for h in data)}
        pre, post = [], []
        for (p, e), h in entry['bkg'].items():
            if e not in group:
                continue
            pre.append(h)
            sf = sfs.get((p, e), 1.0)
            post.append({'edges': h['edges'], 'w': h['w'] * sf, 'w2': h['w2'] * sf ** 2})
        if not pre:
            continue
        hd = th1_from(dsum, "val_data")
        hpre = [th1_from(h, "val_pre") for h in pre]
        hpost = [th1_from(h, "val_post") for h in post]
        c_pre, n_pre = chi2_fn(hd, hpre)
        c_post, n_post = chi2_fn(hd, hpost)
        dtot = dsum['w'][1:-1].sum()
        mpre = sum(h['w'][1:-1].sum() for h in pre)
        mpost = sum(h['w'][1:-1].sum() for h in post)
        out[tag] = {'data': float(dtot), 'mc_pre': float(mpre), 'mc_post': float(mpost),
                    'ratio_pre': float(dtot / mpre) if mpre > 0 else None,
                    'ratio_post': float(dtot / mpost) if mpost > 0 else None,
                    'chi2_pre': float(c_pre), 'ndf_pre': int(n_pre), 'chi2_post': float(c_post), 'ndf_post': int(n_post),
                    'p_pre': float(sps.chi2.sf(c_pre, n_pre)) if n_pre > 0 else None,
                    'p_post': float(sps.chi2.sf(c_post, n_post)) if n_post > 0 else None}
    if len(eras) == 1 and eras[0] in out:
        out['total'] = out[eras[0]]
    return out


def runs_test(resid):
    """Wald-Wolfowitz runs test on residual signs (coherent residual patterns = shape mismatch)."""
    s = np.sign(resid[np.isfinite(resid) & (resid != 0)])
    n1, n2 = int((s > 0).sum()), int((s < 0).sum())
    if n1 == 0 or n2 == 0:
        return {'runs': 1, 'expected': None, 'z': None, 'p': None}
    runs = 1 + int(np.sum(s[1:] != s[:-1]))
    n = n1 + n2
    mu = 2 * n1 * n2 / n + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n ** 2 * (n - 1)) if n > 1 else 0
    z = (runs - mu) / np.sqrt(var) if var > 0 else 0.0
    return {'runs': runs, 'expected': float(mu), 'z': float(z), 'p': float(sps.norm.cdf(z))}  # one-sided: too few runs


def residual_patterns(res):
    """Runs test per era channel and the deviance share of sparse-MC bins."""
    out = OrderedDict()
    chans = np.array(res['bin_channel'])
    for ch in OrderedDict.fromkeys(res['bin_channel']):
        out[ch] = runs_test(np.asarray(res['residual'])[chans == ch])
    model = res['model']
    s = model.scales(res['theta'])
    t = s @ model.T
    v = (s ** 2) @ model.V
    ne = neff(t, v)
    post = np.asarray(res['bin_dev_post'])
    sparse = ne < THRESHOLDS['neff_low']
    return {'runs': out, 'sparse_deviance_share': float(post[sparse].sum() / post.sum()) if post.sum() > 0 else 0.0,
            'n_sparse_bins': int(sparse.sum())}


def run2_only_processes(entry, processes):
    eras_of = {}
    for (p, e) in entry['bkg']:
        eras_of.setdefault(p, set()).add(e)
    return [p for p in processes if eras_of.get(p) == {'run2'}]


def hypothetical_run3(hists, region, var, procs, run_lumis):
    """
    Copy of the channel with a HYPOTHETICAL Run 3 template for each Run-2-only process:
    its Run 2 template scaled by L(Run 3) / L(Run 2). Sensitivity study only -- not a prediction.
    """
    entry = hists[(region, var)]
    new = {'bkg': OrderedDict(entry['bkg']), 'data': entry['data']}
    ratio = run_lumis['run3'] / run_lumis['run2'] if run_lumis and run_lumis.get('run2') else 1.0
    for p in procs:
        h = entry['bkg'][(p, 'run2')].Clone(f"hypo_{region}_{var}_{zlib.crc32(p.encode())}")
        h.SetDirectory(0)
        h.Scale(ratio)
        new['bkg'][(p, 'run3')] = h
    out = OrderedDict(hists)
    out[(region, var)] = new
    return out, ratio


# --------------------------------------------------------------------------- #
# Overlaps, vetoes, weights                                                   #
# --------------------------------------------------------------------------- #
def region_overlaps(adata, regions):
    """
    Events shared by pairs of regions, from the in-pass event keys (same file, same pass), so
    no external event identifiers are needed. Data: event counts; MC: weighted sums.
    """
    from src.sfdiag.inputs import in_pass_key
    recs = adata['records']
    out = []
    for i, a in enumerate(regions):
        for b in regions[i + 1:]:
            row = {'a': a, 'b': b, 'data_events': 0, 'mc_events': 0, 'mc_weight': 0.0, 'by_era': {}}
            for kind in ('data', 'mc'):
                for f, ra in recs.get(a, {}).get(kind, {}).items():
                    rb = recs.get(b, {}).get(kind, {}).get(f)
                    if rb is None:
                        continue
                    ka, kb = in_pass_key(ra), in_pass_key(rb)
                    common = np.isin(ka, kb)
                    n = int(common.sum())
                    era = adata['files'][f]['era']
                    if kind == 'data':
                        row['data_events'] += n
                        row['by_era'].setdefault(era, {'data': 0, 'mc_weight': 0.0})['data'] += n
                    else:
                        row['mc_events'] += n
                        row['mc_weight'] += float(ra['w'][common].sum())
                        row['by_era'].setdefault(era, {'data': 0, 'mc_weight': 0.0})['mc_weight'] += float(ra['w'][common].sum())
            out.append(row)
    return out


def veto_effect(adata, kind):
    """
    Events and weights removed by nBaseLinePhotons == 0, per process and era, plus data/MC.
    Files excluded from the vetoed region because nBaseLinePhotons is missing are not
    counted as removed by the veto: they are listed separately (never passing, never removed).
    """
    from src.sfdiag.inputs import in_pass_key
    recs = adata['records']
    a, b = f"{kind}_noveto", kind
    excluded = {x['file'] for x in adata.get('excluded', []) if x['region'] == b}
    rows = OrderedDict()
    excl_rows = []
    for k in ('mc', 'data'):
        for f, ra in recs.get(a, {}).get(k, {}).items():
            fm = adata['files'][f]
            key = (fm['process'] if k == 'mc' else 'data', fm['era'])
            if f in excluded:
                excl_rows.append({'file': f, 'process': key[0], 'era': key[1], 'events_before': int(len(ra['w'])),
                                  'weight_before': float(ra['w'].sum())})
                continue
            rb = recs.get(b, {}).get(k, {}).get(f)
            kept = np.isin(in_pass_key(ra), in_pass_key(rb)) if rb is not None else np.zeros(len(ra['w']), bool)
            r = rows.setdefault(key, {'process': key[0], 'era': key[1], 'events_before': 0, 'events_removed': 0,
                                      'weight_before': 0.0, 'weight_removed': 0.0, 'sumw2_removed': 0.0,
                                      'missing_npho': 0, 'kept_with_photons': 0})
            r['events_before'] += len(ra['w'])
            r['events_removed'] += int((~kept).sum())
            r['weight_before'] += float(ra['w'].sum())
            r['weight_removed'] += float(ra['w'][~kept].sum())
            r['sumw2_removed'] += float((ra['w'][~kept] ** 2).sum())
            r['missing_npho'] += int((ra['npho'] < 0).sum())
            r['kept_with_photons'] += int((rb['npho'] != 0).sum()) if rb is not None else 0   # must be 0
    for r in rows.values():
        r['fraction_removed'] = r['weight_removed'] / r['weight_before'] if r['weight_before'] else 0.0
    eras = sorted({e for (_, e) in rows})
    dm = OrderedDict()
    for era in eras + ['total']:
        def tot(proc_is_data, removed):
            return sum((r['weight_before'] - (r['weight_removed'] if removed else 0.0))
                       for (p, e), r in rows.items()
                       if (p == 'data') == proc_is_data and (era == 'total' or e == era))
        d0, d1, m0, m1 = tot(True, False), tot(True, True), tot(False, False), tot(False, True)
        dm[era] = {'data_before': d0, 'data_after': d1, 'mc_before': m0, 'mc_after': m1,
                   'ratio_before': d0 / m0 if m0 else None, 'ratio_after': d1 / m1 if m1 else None}
    return {'rows': list(rows.values()), 'data_mc': dm, 'excluded_missing_branch': excl_rows,
            'veto_violations': int(sum(r['kept_with_photons'] for r in rows.values()))}


def weight_report(adata, region):
    """Large / negative event weights of the selected MC events per process and era."""
    out = OrderedDict()
    for f, rec in adata['records'].get(region, {}).get('mc', {}).items():
        fm = adata['files'][f]
        key = f"{fm['process']}:{fm['era']}"
        w = rec['w']
        r = out.setdefault(key, {'w': []})
        r['w'].append(w)
    for key, r in list(out.items()):
        w = np.concatenate(r['w']) if r['w'] else np.array([])
        tot = w.sum()
        ws = np.sort(np.abs(w))[::-1]
        top = max(1, int(np.ceil(0.01 * len(ws)))) if len(ws) else 0
        out[key] = {
            'events': int(len(w)), 'sumw': float(tot), 'sumw2': float((w ** 2).sum()),
            'neff': float(tot ** 2 / (w ** 2).sum()) if len(w) and (w ** 2).sum() > 0 else 0.0,
            'max_w': float(ws[0]) if len(ws) else 0.0,
            'max_w_fraction': float(ws[0] / abs(tot)) if len(ws) and tot else 0.0,
            'top1pct_fraction': float(ws[:top].sum() / np.abs(w).sum()) if len(ws) else 0.0,
            'negative_fraction': float(-w[w < 0].sum() / np.abs(w).sum()) if len(w) and np.abs(w).sum() > 0 else 0.0,
            'flag_large_weight': bool(len(ws) and tot and ws[0] / abs(tot) > 0.1),
        }
    return out


def data_duplicates(adata, region):
    """
    Duplicated (run, lumi, event) among the selected data events of the files that have event
    identifiers. Files without them cannot be checked; they are counted, and the result is
    'partial' (cross-file uniqueness of their events not established).
    """
    recs = adata['records'].get(region, {}).get('data', {})
    if not recs:
        return {'status': 'unavailable', 'reason': 'no selected data events'}
    with_ids = {f: r for f, r in recs.items() if all(k in r for k in ('run', 'lumi', 'event'))}
    without = [f for f in recs if f not in with_ids]
    out = {'files_with_ids': len(with_ids), 'files_without_ids': len(without),
           'events_without_ids': int(sum(len(recs[f]['w']) for f in without)),
           'files_without_ids_list': without}
    if not with_ids:
        out.update(status='unavailable', reason='no event-identifier branches in the data files')
        return out
    arr = np.concatenate([np.stack([r['run'], r['lumi'], r['event']], axis=1) for r in with_ids.values()])
    src = np.concatenate([np.full(len(r['w']), i) for i, r in enumerate(with_ids.values())])
    keys, inv, counts = np.unique(arr, axis=0, return_inverse=True, return_counts=True)
    dup = counts[np.ravel(inv)] > 1
    # duplicates spread over several files (overlapping inputs) vs within one file
    n_files = np.array([len(np.unique(src[np.ravel(inv) == k])) for k in np.where(counts > 1)[0]]) \
        if (counts > 1).any() else np.array([], int)
    out.update(status='ok' if not without else 'partial', events=int(len(arr)),
               duplicated_keys=int((counts > 1).sum()), extra_copies=int((counts - 1)[counts > 1].sum()),
               duplicated_keys_across_files=int((n_files > 1).sum()), duplicated_events=int(dup.sum()))
    return out
