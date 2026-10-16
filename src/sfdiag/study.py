"""
The diagnostic studies of one analysis (compressed / uncompressed), region by region,
with their canvases. Called by diagnose_sf_fit.py.

Per region (photon, hadronic SV, leptonic SV; the fits never combine regions):
  B  composition and MC statistics          (native + common selection, per era)
  C  shape similarity, Asimov information   (which SFs the variable can separate)
  D  fits: shared-era / independent-era / production per_era, weak-process variants
  E  leave-one-interval-out, MC-yield quarters, low/high-stat halves, binning variants
  F  toys (screening for every variable, extended for the recommended one), validation in
     the other observables (correlated) and on the event-level partition (independent),
     hypothetical Run 3 template for Run-2-only processes
"""

import zlib
from collections import OrderedDict

import numpy as np
import ROOT

import fit_sfs
from replot_yields import chi2_ndf
from src.sfdiag import canvases as cv
from src.sfdiag import stats as st
from src.sfdiag.inputs import as_yields, common_mask, in_pass_key, REGION_KINDS

ERA_MODELS = ('shared', 'independent')
PARTITION_META = ('event_ids', 'event_identity_safe', 'partition_key', 'id_events')   # non-variable keys
REGION_TITLE = {'photon': 'prompt #gamma CR', 'hadronic': 'hadronic SV CR', 'leptonic': 'leptonic SV CR'}


def seed_for(base, *parts):
    return (int(base) * 1000003 + zlib.crc32("|".join(map(str, parts)).encode())) % (2 ** 31)


def fit_summary(r):
    """JSON-safe subset of a fit result."""
    if r is None:
        return None
    keys = ('label', 'region', 'var', 'era_model', 'rebin', 'status', 'error', 'params', 'theta', 'err',
            'at_bound', 'singular', 'converged', 'fixed', 'absent', 'per_era', 'deviance', 'deviance_pre',
            'prior_term', 'prior_procs', 'prior_sigma', 'nbins', 'nbins_total', 'nbins_excluded', 'ndf', 'p_gof',
            'data_total', 'data_fitted', 'data_excluded', 'corr', 'identifiability', 'shift_from_nominal',
            'messages', 'channels', 'fix_requested', 'mc_stat')
    return OrderedDict((k, r[k]) for k in keys if k in r)


def asimov_summary_json(info):
    keys = ('params', 'sigma', 'rel', 'corr', 'singular', 'eigenvalues', 'weakest_direction', 'condition_number',
            'info_unprofiled', 'info_profiled', 'separation_fraction')
    return OrderedDict((k, info[k]) for k in keys)


class Ctx:
    """Shared state of one analysis."""

    def __init__(self, adata, acfg, opts, out):
        self.a = adata
        self.cfg = acfg
        self.opts = opts
        self.out = out
        self.name = adata['name']
        self.procs = adata['processes']
        self.eras = [e for e in adata['eras'] if e in ('run2', 'run3')] or adata['eras']
        self.var_meta = adata['var_meta']
        self.native_meta, self.native = as_yields(adata['native'], adata['var_meta'], self.procs, self.eras, None)
        self.common_meta, self.common = as_yields(adata['common'], adata['var_meta'], self.procs, self.eras, None)
        self.sets = {'native': (self.native_meta, self.native, adata['native']),
                     'common': (self.common_meta, self.common, adata['common'])}
        sf = fit_sfs.fit_options(acfg.get('sf_fit') or {})
        self.production = {'per_era': list(sf.per_era or []), 'fix': list(sf.fix or []), 'rebin': sf.rebin,
                           'channels': list(sf.channels or []), 'per_region': bool(sf.per_region)}
        run_lumis = adata.get('run_lumis') or {}
        self.plotters = {e: cv.styled_plotter(run_lumis.get(e, adata['lumi_total']),
                                              {'run2': '13 TeV', 'run3': '13.6 TeV'}.get(e)) for e in self.eras}
        self.plotters['total'] = cv.styled_plotter(adata['lumi_total'], adata.get('energy_label'))

    def production_var(self, kind):
        prod_name = self.a['regions'][kind].get('production_name')
        for ch in self.production['channels']:
            r, _, v = ch.partition(':')
            if r == prod_name:
                return v
        return None


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _major(comp, procs):
    tot = comp.get('total', {}).get('processes', {})
    return [p for p in procs if tot.get(p, {}).get('fraction', 0.0) >= st.THRESHOLDS['major_frac']]


def _weak_processes(info):
    """Processes with a weakly constrained or degenerate parameter in an Asimov fit."""
    weak = set()
    corr = np.abs(np.where(np.isfinite(info['corr']), info['corr'], 1.0))
    for i, p in enumerate(info['params']):
        row = corr[i].copy()
        row[i] = 0
        if not np.isfinite(info['rel'][i]) or info['rel'][i] > st.THRESHOLDS['weak_rel'] or \
                (len(row) and row.max() > st.THRESHOLDS['high_corr']):
            weak.add(p.partition(':')[0])
    return sorted(weak)


def _var_axis(ctx, var):
    m = ctx.var_meta[var]
    return m['label'], np.linspace(m['range'][0], m['range'][1], m['bins'] + 1)


def _per_vbin(values, vidx, eras_of, nb):
    out = OrderedDict()
    for e in OrderedDict.fromkeys(eras_of):
        a = np.zeros(nb)
        sel = eras_of == e
        np.add.at(a, vidx[sel], values[sel])
        out[e] = a
    return out


# --------------------------------------------------------------------------- #
# Region study                                                                #
# --------------------------------------------------------------------------- #
def study_region(ctx, kind):
    a, out, opts = ctx.a, ctx.out, ctx.opts
    base = f"{ctx.name}/{kind}"
    rlabel = f"{REGION_TITLE[kind]} ({ctx.name})"
    res = OrderedDict(kind=kind, cut=a['regions'][kind]['cut'], production_var=ctx.production_var(kind),
                      blind=bool(a['regions'][kind].get('blind')))
    cands = [v for v in a['candidates'] if (kind, v) in a['native']]
    if not cands:
        res['status'] = 'no MC in region'
        return res
    has_data = all(a['native'][(kind, v)]['data'] for v in cands)
    res['has_data'] = has_data
    for key in ('composition', 'similarity', 'era_shapes', 'fits', 'asimov', 'asimov_summary', 'lrt', 'era_ratio',
                'weak', 'residuals'):
        res[key] = OrderedDict((s, OrderedDict()) for s in ctx.sets)
    res['stability'], res['binning'], res['toys_screen'] = OrderedDict(), OrderedDict(), OrderedDict()
    res['validation'], res['partition'] = OrderedDict(), OrderedDict()
    live = {}   # (sel, var, label) -> full fit result (with model)

    for sel, (meta, hists, arr) in ctx.sets.items():
        for var in cands:
            if (kind, var) not in arr:
                continue
            entry = arr[(kind, var)]
            vlabel, edges = _var_axis(ctx, var)
            comp = st.composition(entry, ctx.procs, ctx.eras)
            res['composition'][sel][var] = comp
            res['similarity'][sel][var] = st.similarity(entry, ctx.procs)
            res['era_shapes'][sel][var] = st.era_shape_compat(entry, ctx.procs)
            _draw_inputs(ctx, base, sel, var, entry, comp, rlabel)
            if not has_data or not entry['data']:
                continue
            major = _major(comp, ctx.procs)
            fits = OrderedDict()
            for em in ERA_MODELS:
                fits[em] = st.fit_channel(meta, hists, kind, var, em, label=em)
            fits['production'] = st.fit_channel(meta, hists, kind, var, 'production', fix=ctx.production['fix'],
                                                production_per_era=ctx.production['per_era'], label='production')
            asi = OrderedDict()
            for em in ERA_MODELS:
                if fits[em]['status'] == 'ok':
                    asi[em] = st.asimov_information(fits[em]['model'])
            weak = OrderedDict((em, _weak_processes(info)) for em, info in asi.items())
            for em, wl in weak.items():
                floated = {p.partition(':')[0] for p in fits[em]['params']}
                wl = [p for p in wl if p in floated]
                if wl and len(wl) < len(floated):
                    fits[f"{em}_fixweak"] = st.fit_channel(meta, hists, kind, var, em, fix=[f"{p}=1" for p in wl],
                                                           label=f"{em}, weak fixed")
                if wl:
                    fits[f"{em}_priorweak"] = st.fit_channel(meta, hists, kind, var, em, prior_procs=wl,
                                                             prior_sigma=st.THRESHOLDS['prior_sigma'],
                                                             label=f"{em}, weak prior {st.THRESHOLDS['prior_sigma']}")
            for lab, r in fits.items():
                live[(sel, var, lab)] = r
            res['fits'][sel][var] = OrderedDict((k, fit_summary(r)) for k, r in fits.items())
            res['asimov'][sel][var] = OrderedDict((k, asimov_summary_json(i)) for k, i in asi.items())
            res['asimov_summary'][sel][var] = OrderedDict((k, st.asimov_summary(i, major)) for k, i in asi.items())
            res['weak'][sel][var] = weak
            res['lrt'][sel][var] = st.lrt(fits['shared'], fits['independent'])
            res['era_ratio'][sel][var] = st.era_ratio_tests(fits['independent']) if fits['independent']['status'] == 'ok' else {}
            res['residuals'][sel][var] = OrderedDict((k, st.residual_patterns(r)) for k, r in fits.items()
                                                     if r['status'] == 'ok')
            _draw_fits(ctx, base, sel, var, fits, asi, rlabel)

            if sel != 'native':
                continue
            # E: stability and binning
            res['stability'][var] = OrderedDict()
            res['binning'][var] = OrderedDict()
            res['toys_screen'][var] = OrderedDict()
            variants = st.binning_variants(entry, var, ctx.var_meta[var]['bins'])
            for em in ERA_MODELS:
                r = fits[em]
                if r['status'] != 'ok':
                    continue
                res['stability'][var][em] = st.stability_scan(r, meta, hists)
                bres = OrderedDict()
                for scheme, rb in variants.items():
                    bres[scheme] = r if scheme == 'nominal' else st.fit_channel(meta, hists, kind, var, em, rebin=rb,
                                                                                 label=f"{em}, {scheme}")
                res['binning'][var][em] = OrderedDict(
                    (s, dict(fit_summary(b), scheme=s, rebin_spec=variants[s], quality=_bin_quality(b)))
                    for s, b in bres.items())
                for s, b in bres.items():
                    live[('binning', var, em, s)] = b
                n_toys = opts.toys if not (any(r['at_bound']) or r['singular']) else min(opts.toys, opts.toys_degenerate)
                if n_toys > 0:
                    # model-consistent ensemble: its GoF p enters the ranking of every candidate
                    res['toys_screen'][var][em] = st.run_toys(r, n_toys, seed_for(opts.seed, ctx.name, kind, var, em),
                                                              ensemble='bb_lite')
                    res['toys_screen'][var][em]['degenerate_nominal'] = bool(any(r['at_bound']) or r['singular'])
            _draw_stability(ctx, base, var, res, rlabel)

            # F: validation in the other observables of the region (correlated: same events)
            res['validation'][var] = OrderedDict()
            for em in ERA_MODELS:
                r = fits[em]
                if r['status'] != 'ok':
                    continue
                sfs = st.sf_lookup(r, ctx.procs, ctx.eras)
                vv = OrderedDict()
                for tgt in a['region_vars'].get(kind, []):
                    if tgt == var or (kind, tgt) not in a['native'] or not a['native'][(kind, tgt)]['data']:
                        continue
                    vv[tgt] = st.apply_sfs(a['native'][(kind, tgt)], sfs, ctx.eras, chi2_ndf)
                    vv[tgt]['object_level'] = tgt not in a['candidates']
                res['validation'][var][em] = vv
            _draw_validation_summary(ctx, base, var, res['validation'][var], rlabel)

    # ranking: common-selection discrimination / identifiability among the observables whose
    # native fit is stable, MC-supported and adequate; inconclusive when there is none
    res['ranking'], res['ranking_detail'] = rank_variables(res, cands)
    rec = res['ranking_detail']['recommended']
    res['recommended_var'] = rec
    # the variable the remaining diagnostics are made for: the recommendation, else the
    # best-ranked variable with a native fit (a diagnostic focus, not a recommendation)
    focus = res['ranking_detail']['focus']
    res['focus_var'] = focus
    if focus and has_data:
        meta, hists, arr = ctx.sets['native']
        res['toys_extended'] = OrderedDict()
        for em in ERA_MODELS:
            r = live.get(('native', focus, em))
            if r and r['status'] == 'ok' and opts.toys_extended > 0:
                res['toys_extended'][em] = OrderedDict(
                    (ens, st.run_toys(r, opts.toys_extended, seed_for(opts.seed, ctx.name, kind, focus, em, 'ext', ens),
                                      ensemble=ens))
                    for ens in st.TOY_ENSEMBLES)
        _draw_toys(ctx, base, focus, res)
        # postfit stacks of the focus variable and its validation targets
        for em in ERA_MODELS:
            r = live.get(('native', focus, em))
            if not r or r['status'] != 'ok':
                continue
            sfs = st.sf_lookup(r, ctx.procs, ctx.eras)
            for tgt in a['region_vars'].get(kind, []):
                if (kind, tgt) not in a['native'] or not a['native'][(kind, tgt)]['data']:
                    continue
                for era in ctx.eras + ['total']:
                    eras = ctx.eras if era == 'total' else [era]
                    c = cv.stack_canvas(ctx.plotters[era], a['native'][(kind, tgt)], ctx.var_meta[tgt], eras, sfs,
                                        region_label=f"{rlabel}: SF({em}) from {focus}", tag=f"{kind}_{tgt}_{em}_{era}")
                    out.write(f"{base}/{era if era != 'total' else 'comparison'}/validation",
                              c, f"postfit_{em}_from_{focus}_{tgt}_{era}")
        # hypothetical Run 3 templates of Run-2-only processes
        res['missing_run3'] = missing_run3_study(ctx, kind, focus, live)
        _draw_missing_run3(ctx, base, res['missing_run3'])
    # independent validation on the event-level partition (common selection)
    if has_data:
        res['partition'] = partition_study(ctx, kind, cands)
        _draw_partition(ctx, base, res['partition'])

    # photon veto, weights, duplicates
    if kind in ('hadronic', 'leptonic'):
        res['veto'] = st.veto_effect(a, kind)
        _draw_veto(ctx, base, kind, res['veto'], cands, rlabel)
    res['weights'] = st.weight_report(a, kind)
    res['duplicates'] = st.data_duplicates(a, kind)
    res['objects'] = a['objects'].get(kind, {})
    res['assessment'] = assess(ctx, kind, res)
    return res


def _bin_quality(r):
    if r.get('status') != 'ok':
        return None
    m = r['model']
    t = m.T.sum(axis=0)
    v = m.V.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(t > 0, np.sqrt(v) / t, np.inf)
    ne = st.neff(t, v)
    return {'n_bins': int(len(t)), 'frac_relerr_high': float(np.mean(rel > st.THRESHOLDS['relerr_high'])),
            'frac_sparse': float(np.mean(ne < st.THRESHOLDS['neff_low'])), 'min_neff': float(ne.min()) if len(ne) else None}


# --------------------------------------------------------------------------- #
# Ranking, partition, missing Run 3                                            #
# --------------------------------------------------------------------------- #
def _min_major_hellinger(res, var, sel):
    """Smallest Hellinger distance between two major processes (the least separated pair)."""
    sim = res['similarity'][sel].get(var)
    comp = res['composition'][sel].get(var)
    if not sim or not comp:
        return None, None
    major = set(_major(comp, sim['processes']))
    H = np.asarray(sim['hellinger'], dtype=float)
    best = None
    for i, a in enumerate(sim['processes']):
        for j, b in enumerate(sim['processes']):
            if j > i and a in major and b in major and np.isfinite(H[i, j]) and (best is None or H[i, j] < best[0]):
                best = (float(H[i, j]), f"{a} / {b}")
    return best if best else (None, None)


def _common_metrics(res, var):
    """Discrimination and expected identifiability of a variable on the common event sample."""
    fits = res['fits']['common'].get(var) or {}
    sh = (res['asimov_summary']['common'].get(var) or {}).get('shared', {})
    ind = (res['asimov_summary']['common'].get(var) or {}).get('independent', {})
    hmin, hpair = _min_major_hellinger(res, var, 'common')
    f = fits.get('shared') or {}
    return OrderedDict(
        common_fit_ok=f.get('status') == 'ok',
        n_identified_shared_common=sh.get('n_identified'), n_params_shared_common=sh.get('n_params'),
        median_rel_major_shared_common=sh.get('median_rel_major'), max_corr_shared_common=sh.get('max_corr'),
        condition_shared_common=sh.get('condition_number'), regular_shared_common=sh.get('regular'),
        n_identified_indep_common=ind.get('n_identified'), n_params_indep_common=ind.get('n_params'),
        min_major_hellinger_common=hmin, least_separated_pair_common=hpair,
        p_gof_shared_common=f.get('p_gof'))


def _native_metrics(res, var):
    """Fit stability, MC support and goodness of fit of a variable in its production selection."""
    T = st.THRESHOLDS
    fits = res['fits']['native'].get(var) or {}
    sh = (res['asimov_summary']['native'].get(var) or {}).get('shared', {})
    ind = (res['asimov_summary']['native'].get(var) or {}).get('independent', {})
    path = []
    for em in ERA_MODELS:
        f = fits.get(em)
        if not f or f['status'] != 'ok':
            path.append(f"{em}: fit failed")
            continue
        nb = [p for p, b in zip(f['params'], f['at_bound']) if b]
        if nb:
            path.append(f"{em}: at bound {', '.join(nb)}")
        if f['singular']:
            path.append(f"{em}: singular Hessian")
        if not f['converged']:
            path.append(f"{em}: minimizer did not report convergence")
    unstable = []
    for em, sc in res['stability'].get(var, {}).items():
        if sc.get('status') != 'ok':
            continue
        ident = fits[em]['identifiability']
        for p, s in zip(sc['params'], sc['max_loo_shift']):
            if ident.get(p, {}).get('identified') and s > T['loo_shift']:
                unstable.append(f"{em}:{p} ({s:.1f} sigma, bin {sc['max_loo_bin'][sc['params'].index(p)]})")
    # MC support of the fitted bins (shared-era fit, nominal binning)
    q = ((res['binning'].get(var) or {}).get('shared') or {}).get('nominal', {}).get('quality') or {}
    f = fits.get('shared') or {}
    data_nomc = (f['data_excluded'] / f['data_total']) if f.get('status') == 'ok' and f.get('data_total') else None
    support = []
    if q.get('frac_relerr_high') is not None and q['frac_relerr_high'] > T['support_relerr_frac']:
        support.append(f"{100 * q['frac_relerr_high']:.0f}% of fitted bins with MC rel. error > {T['relerr_high']}")
    if data_nomc is not None and data_nomc > T['support_data_frac']:
        support.append(f"{100 * data_nomc:.1f}% of the data in bins without MC")
    # goodness of fit: asymptotic chi2 p and the approximate BB-lite screening-toy p (stats.gof_verdict)
    ty = (res['toys_screen'].get(var) or {}).get('shared') or {}
    p_toy = ty.get('p_gof_toys')
    p_asym = f.get('p_gof')
    gv = st.gof_verdict(p_asym, ty) if f.get('status') == 'ok' else None
    # binning dependence of the shared-era SFs (in units of the nominal error)
    bins = (res['binning'].get(var) or {}).get('shared') or {}
    spread = 0.0
    if f.get('status') == 'ok':
        for i, p in enumerate(f['params']):
            vals = [b['theta'][b['params'].index(p)] for b in bins.values() if b.get('status') == 'ok' and p in b['params']]
            if len(vals) > 1 and np.isfinite(f['err'][i]) and f['err'][i] > 0:
                spread = max(spread, float((max(vals) - min(vals)) / f['err'][i]))
    return OrderedDict(
        native_fit_ok=f.get('status') == 'ok', pathologies=path, unstable=unstable, mc_support_issues=support,
        frac_bins_relerr_high=q.get('frac_relerr_high'), data_frac_no_mc=data_nomc,
        n_identified_shared=sh.get('n_identified'), n_params_shared=sh.get('n_params'),
        median_rel_major_shared=sh.get('median_rel_major'), max_corr_shared=sh.get('max_corr'),
        condition_shared=sh.get('condition_number'), regular_shared=sh.get('regular'),
        n_identified_indep=ind.get('n_identified'), n_params_indep=ind.get('n_params'),
        median_rel_major_indep=ind.get('median_rel_major'), max_corr_indep=ind.get('max_corr'),
        p_gof_shared=p_asym, p_gof_toys_shared=p_toy,
        p_gof_indep=(fits.get('independent') or {}).get('p_gof'),
        gof_verdict=gv['verdict'] if gv else None, gof_reasons=gv['reasons'] if gv else [],
        n_toys_gof_shared=gv['n_toys_valid'] if gv else None,
        toy_resolution_shared=gv['toy_resolution'] if gv else None,
        gof_ok=bool(gv and gv['verdict'] == 'adequate'),
        dev_ndf_shared=(f['deviance'] / f['ndf'] if f.get('status') == 'ok' and f.get('ndf') else None),
        binning_spread_shared=spread)


def _num(x, default=np.inf):
    return default if x is None or (isinstance(x, float) and not np.isfinite(x)) else x


def rank_variables(res, cands):
    """
    Two rankings of the candidate observables, and the recommendation.

      common  discrimination and expected identifiability on the SAME events for every
              observable (common selection): regular expected Hessian first, then more
              identified SFs (Asimov sigma_rel < ident_rel, shared-era model), smaller median
              sigma_rel of the major processes, smaller max |rho|, larger separation of the
              least-separated major pair. This is the like-for-like comparison of observables.
      native  the production selection of each observable: fits without pathologies (bound,
              singular, not converged), stable (leave-one-bin-out), MC-supported, adequate GoF
              first, then the same identifiability order on the native events.

    An observable is eligible when its native shared-era fit is free of pathologies, stable,
    MC-supported and fits (GoF), and its common-selection expected Hessian is regular with at
    least one identified SF. The recommendation is the eligible observable ranked best on the
    common selection; with no eligible observable it is None (inconclusive). Small nominal
    errors never make an observable eligible on their own, and event counts or the postfit
    deviance are not ranked on. The thresholds are screening heuristics (stats.THRESHOLDS).
    Returns (rows in final order, detail).
    """
    rows = OrderedDict()
    for var in cands:
        row = OrderedDict(var=var)
        if not res['fits']['native'].get(var):
            row.update(usable=False, eligible=False, reason='no native fit (no data?)', ineligible=['no native fit'])
            rows[var] = row
            continue
        row.update(_native_metrics(res, var))
        row.update(_common_metrics(res, var))
        why = []
        if not row['native_fit_ok']:
            why.append("native shared-era fit failed")
        why += [f"native: {p}" for p in row['pathologies']]
        why += [f"native unstable: {u}" for u in row['unstable']]
        why += [f"native MC support: {s}" for s in row['mc_support_issues']]
        if row['native_fit_ok'] and not row['gof_ok']:
            pt = row['p_gof_toys_shared']
            why.append(f"native GoF {row['gof_verdict']}: asymptotic p = {_fmt(row['p_gof_shared'])}"
                       + (f", approx. toy p = {_fmt(pt)} ({row['n_toys_gof_shared']} toys)" if pt is not None else "")
                       + (f" [{'; '.join(row['gof_reasons'])}]" if row['gof_reasons'] else ""))
        if not row['common_fit_ok']:
            why.append("common-selection shared-era fit failed")
        elif not row['regular_shared_common']:
            why.append("common selection: expected Hessian singular / near-degenerate (condition "
                       f"{_fmt(row['condition_shared_common'])})")
        elif not row['n_identified_shared_common']:
            why.append("common selection: no SF identified")
        row['usable'] = row['native_fit_ok']
        row['eligible'] = not why
        row['ineligible'] = why
        rows[var] = row

    def common_key(r):
        if not r.get('common_fit_ok'):
            return (2, 0, np.inf, np.inf, 0.0)
        return (0 if r['regular_shared_common'] else 1, -(r['n_identified_shared_common'] or 0),
                _num(r['median_rel_major_shared_common']), _num(r['max_corr_shared_common'], 1.0),
                -_num(r['min_major_hellinger_common'], 0.0))

    def native_key(r):
        if not r.get('native_fit_ok'):
            return (3, 0, 0, np.inf)
        clean = not (r['pathologies'] or r['unstable'] or r['mc_support_issues'])
        return (0 if clean else 1, {'adequate': 0, 'inconclusive': 1}.get(r['gof_verdict'], 2),
                -(r['n_identified_shared'] or 0),
                _num(r['median_rel_major_shared']))

    order_common = sorted(rows, key=lambda v: common_key(rows[v]))
    order_native = sorted(rows, key=lambda v: native_key(rows[v]))
    for i, v in enumerate(order_common):
        rows[v]['rank_common'] = i + 1
    for i, v in enumerate(order_native):
        rows[v]['rank_native'] = i + 1
    final = sorted(rows, key=lambda v: (0 if rows[v]['eligible'] else 1 if rows[v].get('usable') else 2,
                                        rows[v]['rank_common'], rows[v]['rank_native']))
    for i, v in enumerate(final):
        rows[v]['rank'] = i + 1
    recommended = final[0] if final and rows[final[0]]['eligible'] else None
    focus = next((v for v in final if rows[v].get('usable')), None)

    # where the two selections lead to different conclusions
    notes = []
    if order_common and order_native and order_common[0] != order_native[0]:
        a, b = rows[order_common[0]], rows[order_native[0]]
        notes.append(
            f"common selection ranks {a['var']} first ({a.get('n_identified_shared_common')}/"
            f"{a.get('n_params_shared_common')} SFs identified, median sigma_rel "
            f"{_fmt(a.get('median_rel_major_shared_common'))}), the native selections rank {b['var']} first "
            f"({b.get('n_identified_shared')}/{b.get('n_params_shared')}, median sigma_rel "
            f"{_fmt(b.get('median_rel_major_shared'))}"
            + (f"; {a['var']} natively: {'; '.join(a['ineligible'])}" if a.get('ineligible') else "") + ")")
    for v, r in rows.items():
        nc, nn = r.get('n_identified_shared_common'), r.get('n_identified_shared')
        if nc is not None and nn is not None and nn != nc:
            notes.append(f"{v}: {nn} SFs identified in its native selection vs {nc} on the common events: "
                         f"the difference comes from the selected event sample, not from the observable")
    if recommended and order_common and recommended != order_common[0]:
        notes.append(f"{recommended} is recommended although {order_common[0]} discriminates better on the common "
                     f"events, because {order_common[0]} is not eligible: {'; '.join(rows[order_common[0]]['ineligible'])}")
    detail = OrderedDict(
        recommended=recommended, focus=focus, status='recommended' if recommended else 'inconclusive',
        order_common=order_common, order_native=order_native, order_final=final, disagreements=notes,
        criteria="eligible = native shared fit converged, regular, off bounds, stable, MC-supported, GoF adequate; "
                 "and common-selection expected Hessian regular with >= 1 identified SF. Recommendation = eligible "
                 "observable best ranked on the common selection.")
    if not recommended:
        detail['reasons'] = OrderedDict((v, rows[v].get('ineligible') or []) for v in final)
    return [rows[v] for v in final], detail


def partition_study(ctx, kind, cands):
    """
    Fit half A of the events (common selection), apply to half B in every candidate variable,
    and compare with the SFs fitted on half B alone. The halves are disjoint event sets, so
    this is an independent validation (unlike the cross-variable checks on the same events).
    """
    a = ctx.a
    ids = a['id_summary']
    ev = {k: ids[k]['events_by_region'].get(kind, {}) for k in ('mc', 'data')}
    out = OrderedDict(
        event_ids=a['event_ids'],
        event_identity_safe=a['event_ids'],
        partition_key=('(run, lumi, event) for every data file' if a['event_ids'] else
                       f"(run, lumi, event) where available ({ev['data'].get('event_id', 0)} data events), in-pass "
                       f"(file, chunk, row) otherwise ({ev['data'].get('in_pass', 0)} data events): reproducible for "
                       f"fixed inputs and loader chunking, but the same event in two files is not recognized"),
        id_events=ev)
    halves = {}
    for h in ('A', 'B'):
        halves[h] = as_yields(a['halves'][h], a['var_meta'], ctx.procs, ctx.eras, [kind])
    for var in cands:
        if (kind, var) not in halves['A'][1] or not halves['A'][1][(kind, var)]['data']:
            continue
        out[var] = OrderedDict()
        for em in ERA_MODELS:
            fa = st.fit_channel(*halves['A'], kind, var, em, label=f"{em}, half A")
            fb = st.fit_channel(*halves['B'], kind, var, em, label=f"{em}, half B")
            row = OrderedDict(fit_A=fit_summary(fa), fit_B=fit_summary(fb))
            if fa['status'] == 'ok' and fb['status'] == 'ok' and fa['params'] == fb['params']:
                with np.errstate(invalid='ignore', divide='ignore'):
                    z = (fa['theta'] - fb['theta']) / np.sqrt(fa['err'] ** 2 + fb['err'] ** 2)
                row['compat_z'] = OrderedDict(zip(fa['params'], z.tolist()))
            if fa['status'] == 'ok':
                sfs = st.sf_lookup(fa, ctx.procs, ctx.eras)
                row['apply_A_to_B'] = OrderedDict()
                for tgt in cands:
                    if (kind, tgt) in a['halves']['B'] and a['halves']['B'][(kind, tgt)]['data']:
                        row['apply_A_to_B'][tgt] = st.apply_sfs(a['halves']['B'][(kind, tgt)], sfs, ctx.eras, chi2_ndf)
            out[var][em] = row
    return out


def missing_run3_study(ctx, kind, var, live):
    """HYPOTHETICAL Run 3 templates for Run-2-only processes: how much would the other SFs move?"""
    meta, hists, arr = ctx.sets['native']
    entry = arr[(kind, var)]
    r2only = st.run2_only_processes(entry, ctx.procs)
    out = OrderedDict(label='HYPOTHETICAL sensitivity study, not a Run 3 prediction', processes=r2only)
    comp = st.composition(entry, ctx.procs, ctx.eras)
    out['run2_fraction'] = {p: comp.get('run2', {}).get('processes', {}).get(p, {}).get('fraction') for p in r2only}
    out['total_fraction'] = {p: comp.get('total', {}).get('processes', {}).get(p, {}).get('fraction') for p in r2only}
    if not r2only or not ctx.a.get('run_lumis'):
        return out
    hyp, ratio = st.hypothetical_run3(hists, kind, var, r2only, ctx.a['run_lumis'])
    out['lumi_ratio'] = ratio
    for em in ERA_MODELS:
        nom = live.get(('native', var, em))
        h = st.fit_channel(meta, hyp, kind, var, em, label=f"{em}, hypothetical Run 3 {'+'.join(r2only)}")
        row = OrderedDict(fit=fit_summary(h))
        if nom and nom['status'] == 'ok' and h['status'] == 'ok':
            sh = OrderedDict()
            for p, v, e in zip(nom['params'], nom['theta'], nom['err']):
                if p in h['params']:
                    sh[p] = float((h['theta'][h['params'].index(p)] - v) / e) if np.isfinite(e) and e > 0 else None
            row['shift_in_sigma'] = sh
        out[em] = row
    return out


# --------------------------------------------------------------------------- #
# Assessment                                                                  #
# --------------------------------------------------------------------------- #
def _fmt(x, f="{:.3g}"):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f.format(x)


def _gof_inadequate(f, ext_bb):
    """
    GoF of a fit (stats.gof_verdict on the asymptotic p and the approximate BB-lite toy p).
    Returns (not adequate, verdict dict, description). An inconclusive verdict is not adequate:
    small parameter errors are then not counted as a measurement.
    """
    gv = st.gof_verdict(f.get('p_gof'), ext_bb or None)
    desc = (f"asymptotic p = {_fmt(gv['p_asymptotic'])}"
            + (f", approx. BB-lite toy p = {_fmt(gv['p_toys'])} ({gv['n_toys_valid']} toys, resolution "
               f"{_fmt(gv['toy_resolution'])})" if gv['p_toys'] is not None else "")
            + (f"; {'; '.join(gv['reasons'])}" if gv['reasons'] else ""))
    return gv['verdict'] != 'adequate', gv, desc


def assess(ctx, kind, res):
    """Rule-based assessment of one region; every statement carries its numbers."""
    T = st.THRESHOLDS
    A = OrderedDict(supported=[], disfavored=[], inconclusive=[], limitations=[])
    rec = res.get('recommended_var')
    focus = res.get('focus_var')
    detail = res.get('ranking_detail') or {}
    A['recommended_observable'] = rec
    A['diagnostic_focus'] = focus
    A['recommendation_status'] = detail.get('status', 'inconclusive')
    A['selection_comparison'] = list(detail.get('disagreements', []))
    if not res.get('has_data'):
        A['inconclusive'].append("no data in this region: no fit performed")
        return A
    if not focus:
        A['inconclusive'].append("no candidate observable gives a usable native shared-era fit")
        return A
    rk = {r['var']: r for r in res['ranking']}
    r0 = rk[focus]

    def ident_txt(r):
        return (f"common selection {r.get('n_identified_shared_common')}/{r.get('n_params_shared_common')} SFs "
                f"identified, median sigma_rel {_fmt(r.get('median_rel_major_shared_common'))}, max |rho| "
                f"{_fmt(r.get('max_corr_shared_common'))}, least separated major pair "
                f"{r.get('least_separated_pair_common')} (Hellinger {_fmt(r.get('min_major_hellinger_common'))}); "
                f"native {r.get('n_identified_shared')}/{r.get('n_params_shared')}, median sigma_rel "
                f"{_fmt(r.get('median_rel_major_shared'))}")
    if rec:
        A['supported'].append(f"{rec} recommended (rank {r0['rank_common']} on the common events, "
                              f"{r0['rank_native']} in the native selections): {ident_txt(r0)}; native fit stable, "
                              f"MC-supported, GoF p = {_fmt(r0.get('p_gof_shared'))}"
                              + (f" (toys {_fmt(r0['p_gof_toys_shared'])})" if r0.get('p_gof_toys_shared') is not None else ""))
    else:
        A['inconclusive'].append(
            "no observable gives a stable, MC-supported and adequate normalization fit with identifiable SFs: "
            "observable recommendation inconclusive. The diagnostics below are for " + focus +
            " (best ranked) as a diagnostic focus, not a recommendation")
    for r in res['ranking']:
        if r['var'] == rec:
            continue
        A['disfavored' if rec else 'inconclusive'].append(
            f"{r['var']} (common rank {r.get('rank_common')}, native rank {r.get('rank_native')}): "
            + (ident_txt(r) if r.get('usable') else r.get('reason', '')) +
            (f"; not eligible: {'; '.join(r['ineligible'])}" if r.get('ineligible') else ""))
    var = focus

    # era model: the LRT p-value is only used when Wilks' conditions hold
    lr = res['lrt']['native'].get(var)
    fits = res['fits']['native'][var]
    A['era_model'] = 'inconclusive'
    if lr:
        txt = (f"shared vs independent Run 2/Run 3 SFs on {var}: delta(-2lnL) = {lr['delta_deviance']:.2f} "
               f"for {lr['delta_k']} extra parameters, nominal p = {_fmt(lr['p'])}")
        if not lr['interpretable']:
            A['inconclusive'].append("era dependence: " + txt + "; the p-value is not interpretable ("
                                     + "; ".join(lr['caveats']) + ")")
        elif lr['p'] < T['gof_p']:
            A['era_model'] = 'independent'
            A['supported'].append("independent Run 2/Run 3 SFs: " + txt)
            A['disfavored'].append("shared Run 2/Run 3 SFs: " + txt)
        elif lr['p'] > 0.05:
            A['era_model'] = 'shared'
            A['supported'].append("shared Run 2/Run 3 SFs (the data do not require era-specific SFs): " + txt)
        else:
            A['inconclusive'].append("era dependence: " + txt)
        ratios = res['era_ratio']['native'].get(var, {})
        sig = [f"{p} (run3 - run2 = {v['diff']:+.3f}, z = {v['z']:.1f})" for p, v in ratios.items()
               if v.get('z') is not None and np.isfinite(v['z']) and abs(v['z']) > 3]
        if sig:
            A['supported' if lr['interpretable'] else 'inconclusive'].append("era-dependent SFs for: " + ", ".join(sig))
        ia = (res['asimov_summary']['native'].get(var) or {}).get('independent', {})
        if A['era_model'] == 'independent' and not ia.get('regular', True):
            A['inconclusive'].append("the independent-era model is preferred but its expected Hessian is singular / "
                                     "near-degenerate: some era-specific SFs are not separately measurable")
    else:
        A['inconclusive'].append("shared vs independent era test not available (a fit failed or no extra parameters)")
    # the model the remaining diagnostics are evaluated for
    em = A['era_model'] if A['era_model'] != 'inconclusive' else 'shared'
    if fits.get(em, {}).get('status') != 'ok':
        em = next((m for m in ERA_MODELS if fits.get(m, {}).get('status') == 'ok'), em)
    A['era_model_used'] = em
    f = fits.get(em)
    if not f or f['status'] != 'ok':
        A['inconclusive'].append(f"the {em} fit of {var} failed: {f.get('error') if f else 'missing'}")
        return A

    # toys of the focus variable (model-consistent ensemble for closure and GoF)
    ext_all = (res.get('toys_extended') or {}).get(em) or {}
    ext = ext_all.get('bb_lite') or {}
    tsum = []
    for ens, ty in ext_all.items():
        if ty.get('status') != 'ok':
            tsum.append(f"{ens}: {ty.get('status')} ({ty.get('reason', '')})")
            continue
        pw = ", ".join(f"{p} {v['pull_mean']:+.2f}/{v['pull_width']:.2f}" for p, v in ty['per_param'].items()
                       if v.get('pull_width') is not None)
        tsum.append(f"{ens} ({ty['n_valid']}/{ty['n_toys']} valid fits, {ty['n_failed']} failed, "
                    f"{ty['n_not_converged']} not converged, singular {ty['frac_singular']:.2f}, any bound "
                    f"{ty['frac_any_bound']:.2f}): pull mean/width {pw}; closure {'ok' if ty['closure_ok'] else 'FAILS'}; "
                    + (f"approx. GoF toy p = {_fmt(ty['p_gof_toys'])} (resolution "
                       f"{_fmt(ty['gof']['p_observed_resolution'])})" if ty['gof']['estimates_observed_p'] else
                       "no GoF p of the observed data from this ensemble"))
    if tsum:
        A['toys'] = tsum
    tp = ext.get('per_param', {})

    inadequate, gv, gdesc = _gof_inadequate(f, ext)
    A['gof_verdict'] = gv['verdict']
    cons, need, precise_inadequate = [], [], []
    for p, idf in f['identifiability'].items():
        t = tp.get(p, {})
        toy_bad = (t.get('pull_mean') is not None and abs(t['pull_mean']) > T['toy_bias']) or \
                  (t.get('pull_width') is not None and not (T['toy_width'][0] <= t['pull_width'] <= T['toy_width'][1])) or \
                  t.get('frac_at_bound', 0) > 0.05
        i = f['params'].index(p)
        desc = (f"{p} = {f['theta'][i]:.3f} +- {_fmt(f['err'][i])} (sigma_rel {_fmt(idf['rel_err'])}, "
                f"max |rho| {idf['max_corr']:.2f} with {idf['max_corr_with']}"
                + (f", BB-lite toy pull {t['pull_mean']:+.2f}/{t['pull_width']:.2f}" if t.get('pull_width') is not None else "") + ")")
        if idf['identified'] and idf['max_corr'] <= T['high_corr'] and not toy_bad:
            # a small error from a model that does not describe the data is not a measurement
            (precise_inadequate if inadequate else cons).append(desc)
        elif idf['weak'] or idf['max_corr'] > T['high_corr'] or toy_bad:
            need.append(desc + (" [toy closure fails]" if toy_bad else "")
                        + ("" if idf.get('regular', True) else " [no regular error: bound / singular / not converged]"))
        else:
            A['inconclusive'].append(f"moderately constrained: {desc}")
    A['constrained_backgrounds'] = cons
    A['precise_but_model_inadequate'] = precise_inadequate
    A['backgrounds_needing_treatment'] = need + [f"{p} (no MC in the fit range; fixed)" for p in f.get('absent', [])]
    if precise_inadequate:
        A['inconclusive'].append(f"{len(precise_inadequate)} SF(s) have small nominal errors but the {em} model "
                                 + ("does not describe" if gv['verdict'] == 'inadequate' else "is not shown to describe")
                                 + f" {var} (GoF {gv['verdict']}: {gdesc}): not counted as constrained")

    # variants of the weak processes
    for lab in (f"{em}_fixweak", f"{em}_priorweak"):
        v = fits.get(lab)
        if v and v['status'] == 'ok':
            shifts = []
            for p, val in zip(v['params'], v['theta']):
                if p in f['params']:
                    j = f['params'].index(p)
                    e = f['err'][j]
                    if np.isfinite(e) and e > 0:
                        shifts.append(f"{p} {(val - f['theta'][j]) / e:+.1f} sigma")
            A.setdefault('weak_variants', []).append(
                f"{v['label']}: -2lnL {v['deviance']:.1f} / ndf {v['ndf']} (nominal {f['deviance']:.1f} / {f['ndf']}); "
                f"shifts of the common SFs: {', '.join(shifts) if shifts else 'none'}")

    # correlations
    corr = np.asarray(f['corr'], dtype=float)
    pairs = []
    for i in range(len(f['params'])):
        for j in range(i + 1, len(f['params'])):
            if np.isfinite(corr[i, j]) and abs(corr[i, j]) > 0.5:
                pairs.append((abs(corr[i, j]), f"{f['params'][i]} / {f['params'][j]}: {corr[i, j]:+.2f}"))
    A['correlations'] = [p for _, p in sorted(pairs, reverse=True)]
    asi = res['asimov']['native'][var].get(em)
    if asi:
        wd = sorted(asi['weakest_direction'].items(), key=lambda kv: -abs(kv[1]))[:3]
        A['weakest_direction'] = (f"smallest correlation eigenvalue {asi['eigenvalues'][0]:.3g} "
                                  f"(condition number {_fmt(asi['condition_number'])}): "
                                  + ", ".join(f"{p} {c:+.2f}" for p, c in wd))

    # binning
    bins = res['binning'].get(var, {}).get(em, {})
    choice = None
    for scheme in ('nominal', 'merge2', 'adaptive', 'merge5'):
        b = bins.get(scheme)
        if b and b.get('status') == 'ok' and b.get('quality') and b['quality']['frac_relerr_high'] <= 0.1:
            choice = scheme
            break
    if choice is None:
        ok = [s for s, b in bins.items() if b.get('status') == 'ok' and b.get('quality')]
        choice = min(ok, key=lambda s: bins[s]['quality']['frac_relerr_high']) if ok else 'nominal'
    A['recommended_binning'] = choice
    A['recommended_binning_spec'] = bins.get(choice, {}).get('rebin_spec')
    spread = []
    for p in f['params']:
        vals = [(s, b['theta'][b['params'].index(p)]) for s, b in bins.items()
                if b.get('status') == 'ok' and p in b['params']]
        i = f['params'].index(p)
        if len(vals) > 1 and np.isfinite(f['err'][i]) and f['err'][i] > 0:
            v = np.array([x for _, x in vals])
            spread.append((float((v.max() - v.min()) / f['err'][i]), p))
    q = bins.get(choice, {}).get('quality') or {}
    A['binning_note'] = (f"{choice}: {q.get('n_bins')} fitted bins, {100 * q.get('frac_relerr_high', 0):.0f}% with MC "
                         f"rel. error > {T['relerr_high']}; SF spread over binnings (in nominal sigma): "
                         + ", ".join(f"{p} {s:.1f}" for s, p in sorted(spread, reverse=True)))
    if spread and max(spread)[0] > 1.0:
        A['limitations'].append(f"binning dependence: {max(spread)[1]} varies by {max(spread)[0]:.1f} sigma across binnings")

    # stability
    sc = res['stability'].get(var, {}).get(em, {})
    if sc.get('status') == 'skipped':
        A['stability'] = f"not evaluated: {sc.get('reason')}"
        A['inconclusive'].append(f"stability of the {em} SFs of {var} not evaluated ({sc.get('reason')})")
    if sc.get('status') == 'ok':
        mx = sc['max_loo_shift']
        A['stability'] = (f"largest leave-one-interval-out shift: " +
                          ", ".join(f"{p} {s:.1f} sigma (bin {b})" for p, s, b in
                                    zip(sc['params'], mx, sc['max_loo_bin'])) +
                          "; largest MC-quarter removal shift: " +
                          ", ".join(f"{p} {s:.1f}" for p, s in zip(sc['params'], sc['max_group_shift'])))
        big = [p for p, s in zip(sc['params'], sc['max_group_shift']) if s > 2 * T['loo_shift']]
        if big:
            A['limitations'].append(f"SFs driven by a narrow kinematic interval (quarter removal > 2 sigma): {', '.join(big)}")

    # goodness of fit and validation
    g = ext.get('gof') or {}
    A['gof'] = (f"-2lnL = {f['deviance']:.1f} for ndf {f['ndf']} (asymptotic p = {_fmt(f['p_gof'])}"
                + (f", approximate BB-lite toy p = {_fmt(ext.get('p_gof_toys'))} from {g.get('n_effective')} valid toys "
                   f"(resolution {_fmt(g.get('p_observed_resolution'))}); "
                   f"toy -2lnL mean/ndf {_fmt(g.get('mean_over_ndf'))}, KS p vs chi2(ndf) {_fmt(g.get('ks_p_vs_chi2_ndf'))}"
                   if ext.get('p_gof_toys') is not None else "")
                + f"); prefit -2lnL {f['deviance_pre']:.1f}; "
                f"{f['nbins_excluded']} bins without MC, {f['data_excluded']:.0f} data events not fitted; "
                f"verdict {gv['verdict']}" + (f" ({'; '.join(gv['reasons'])})" if gv['reasons'] else ""))
    if g.get('ks_p_vs_chi2_ndf') is not None and g['ks_p_vs_chi2_ndf'] < T['gof_p']:
        A['limitations'].append(f"the toy -2lnL distribution of {var} is not chi2(ndf) (KS p = "
                                f"{_fmt(g['ks_p_vs_chi2_ndf'])}): the asymptotic GoF p is unreliable; the toy p is "
                                f"itself approximate (Gamma vs Gaussian MC nuisance)")
    val = res['validation'].get(var, {}).get(em, {})
    vtxt, vbad = [], []
    for tgt, v in val.items():
        t = v.get('total')
        if not t:
            continue
        vtxt.append(f"{tgt}{' (objects)' if v.get('object_level') else ''}: chi2/ndf {t['chi2_pre']:.0f}/{t['ndf_pre']} -> "
                    f"{t['chi2_post']:.0f}/{t['ndf_post']}, data/MC {_fmt(t['ratio_pre'])} -> {_fmt(t['ratio_post'])}")
        if not v.get('object_level'):
            if t['p_post'] is not None and t['p_post'] < T['gof_p'] and t['ndf_post'] and t['chi2_post'] / t['ndf_post'] > 2:
                vbad.append(tgt)
    A['validation'] = "correlated (same events): " + "; ".join(vtxt) if vtxt else "no validation targets"
    part = (res.get('partition') or {}).get(var, {}).get(em, {})
    if part.get('compat_z'):
        zs = part['compat_z']
        A['partition'] = ("independent halves: SF(A) - SF(B) z = " +
                          ", ".join(f"{p} {z:+.1f}" for p, z in zs.items() if z is not None and np.isfinite(z)))
        badz = [p for p, z in zs.items() if z is not None and np.isfinite(z) and abs(z) > 3]
        if badz:
            A['limitations'].append(f"SFs from the two event halves disagree (|z| > 3): {', '.join(badz)}")

    # model adequacy
    tags = []
    if gv['verdict'] == 'inconclusive':
        A['limitations'].append(f"goodness of fit of the {em} model on {var} inconclusive ({gdesc})")
    if gv['verdict'] == 'inadequate':
        A['limitations'].append(f"normalization-only model does not describe {var} (GoF: {gdesc})")
        if need or r0['pathologies']:
            tags.append("poorly constrained process mixture")
        rp = res['residuals']['native'][var].get(em, {})
        coh = [ch for ch, rt in rp.get('runs', {}).items() if rt.get('p') is not None and rt['p'] < T['gof_p']]
        if coh:
            tags.append(f"coherent residual pattern in {', '.join(coh)} (variable-dependent shape difference)")
        if rp.get('sparse_deviance_share', 0) > 0.25:
            tags.append(f"sparse MC templates ({100 * rp['sparse_deviance_share']:.0f}% of -2lnL from bins with "
                        f"N_eff < {T['neff_low']:.0f})")
        if f['data_excluded'] > 0.01 * max(f['data_total'], 1):
            tags.append(f"data without MC support ({f['data_excluded']:.0f} events): missing background contribution?")
        if lr and lr['p'] < T['gof_p']:
            tags.append("era-dependent normalization (reconstruction / selection-efficiency differences between runs?)"
                        + ("" if lr['interpretable'] else " [LRT p not interpretable]"))
        if not tags:
            tags.append("unresolved modeling limitation (no single diagnostic pattern)")
    if vbad:
        tags.append(f"SFs from {var} do not describe {', '.join(vbad)} (postfit chi2/ndf > 2): "
                    f"variable-dependent shape differences")
    A['model_adequacy'] = tags
    if tags:
        A['limitations'] += [t for t in tags if t not in A['limitations']]
        A['inconclusive'].append("normalization-only SFs are not sufficient here; physics / simulation studies of "
                                 "the listed limitations are needed before using these SFs")
    return A


# --------------------------------------------------------------------------- #
# Canvases                                                                    #
# --------------------------------------------------------------------------- #
def _colors(ctx, procs):
    pl = ctx.plotters['total']
    pl._ensure_mc_colors()
    return {p: cv.proc_color(pl, p) for p in procs}


def _draw_inputs(ctx, base, sel, var, entry, comp, rlabel):
    out = ctx.out
    vlabel, edges = _var_axis(ctx, var)
    col = _colors(ctx, ctx.procs)
    T, D = st.entry_templates(entry)
    for era in ctx.eras:
        if era not in comp:
            continue
        d = f"{base}/{era}/{sel}"
        c = cv.stack_canvas(ctx.plotters[era], entry, ctx.var_meta[var], [era], None,
                            region_label=f"{rlabel}, {sel}", tag=f"{sel}_{var}_{era}")
        out.write(d, c, f"prefit_{sel}_{var}_{era}")
        pr = comp[era]['processes']
        series = [(p, pr[p]['bin_fraction'], None, col[p], 1) for p in pr]
        out.write(d, cv.overlay_canvas(f"MC composition ({era}, {sel}, {rlabel})", vlabel, "fraction of MC in bin",
                                       edges, series, ymin=-0.05, ymax=1.05, draw="HIST"), f"composition_{sel}_{var}_{era}")
        series = [(p, pr[p]['bin_neff'], None, col[p], 1) for p in pr]
        out.write(d, cv.overlay_canvas(f"MC N_eff = (#sum w)^{{2}}/#sum w^{{2}} ({era}, {sel})", vlabel, "N_{eff}",
                                       edges, series, logy=True, draw="HIST",
                                       notes=[f"sparse: N_eff < {st.THRESHOLDS['neff_low']:.0f}"]),
                  f"neff_{sel}_{var}_{era}")
        re_ = np.array(comp[era]['total_relerr'], dtype=float)
        re_[re_ < 0] = np.nan
        out.write(d, cv.overlay_canvas(f"total MC relative stat. uncertainty ({era}, {sel})", vlabel,
                                       "#sigma_{MC} / MC", edges, [("total MC", re_, None, ROOT.kBlack, 1)],
                                       draw="HIST", ymin=0, notes=[f"bins > {st.THRESHOLDS['relerr_high']}: "
                                                                   f"{comp[era]['bins_relerr_high']}"]),
                  f"relerr_{sel}_{var}_{era}")
        series = []
        for p in ctx.procs:
            if (p, era) in T and T[(p, era)][0].sum() > 0:
                t, v = T[(p, era)]
                s = t.sum()
                series.append((p, t / s, np.sqrt(np.clip(v, 0, None)) / s, col[p], 1))
        if era in D and D[era].sum() > 0:
            series.append(("data", D[era] / D[era].sum(), np.sqrt(D[era]) / D[era].sum(), ROOT.kBlack, 2))
        out.write(d, cv.overlay_canvas(f"unit-normalized templates ({era}, {sel})", vlabel, "fraction / bin", edges,
                                       series), f"templates_{sel}_{var}_{era}")
    # comparison: era shapes and similarity
    d = f"{base}/comparison/templates"
    series = []
    for i, p in enumerate(ctx.procs):
        for era, ls in (('run2', 1), ('run3', 2)):
            if (p, era) in T and T[(p, era)][0].sum() > 0:
                t = T[(p, era)][0]
                series.append((f"{p} {era}", t / t.sum(), None, col[p], ls))
    out.write(d, cv.overlay_canvas(f"Run 2 (solid) vs Run 3 (dashed) shapes, {sel}", vlabel, "fraction / bin", edges,
                                   series, draw="HIST"), f"era_shapes_{sel}_{var}")
    sim = st.similarity(entry, ctx.procs)
    if len(sim['processes']) > 1:
        out.write(d, cv.matrix_canvas(f"template Hellinger distance ({sel}, {var}; 0 = identical shape)",
                                      sim['processes'], sim['processes'], sim['hellinger'], 0, 1, "4.2f"),
                  f"similarity_{sel}_{var}")


def _draw_fits(ctx, base, sel, var, fits, asi, rlabel):
    out = ctx.out
    d = f"{base}/comparison/fits"
    vlabel, edges = _var_axis(ctx, var)
    # SF overview
    labels = []
    for r in fits.values():
        if r['status'] == 'ok':
            labels += [p for p in r['params'] if p not in labels]
    series = []
    for k, (lab, r) in enumerate(fits.items()):
        if r['status'] != 'ok':
            continue
        y = [r['theta'][r['params'].index(p)] if p in r['params'] else None for p in labels]
        ey = [r['err'][r['params'].index(p)] if p in r['params'] else None for p in labels]
        series.append((r['label'], y, ey, cv.LINE_COLORS[k % len(cv.LINE_COLORS)]))
    if series:
        out.write(d, cv.graph_canvas(f"fitted SFs, {sel} {var} ({rlabel})", labels, series, "scale factor",
                                     ref_lines=[(1.0, 2)]), f"sfs_{sel}_{var}")
    for lab, r in fits.items():
        if r['status'] != 'ok':
            continue
        if len(r['params']) > 1:
            out.write(d, cv.matrix_canvas(f"postfit correlation, {r['label']} ({sel}, {var})", r['params'], r['params'],
                                          r['corr'], -1, 1, "4.2f"), f"corr_{sel}_{var}_{lab}")
        # per-bin deviance on the variable axis
        try:
            meta, hists, _ = ctx.sets[sel]
            vidx, eras_of = st.bin_index_map(meta, hists, r['region'], var, r['rebin'])
            nb = ctx.var_meta[var]['bins']
            if r['rebin'] is None and len(vidx) == len(r['bin_dev_post']):
                pre = _per_vbin(np.asarray(r['bin_dev_pre']), vidx, eras_of, nb)
                post = _per_vbin(np.asarray(r['bin_dev_post']), vidx, eras_of, nb)
                series = []
                for i, e in enumerate(pre):
                    series.append((f"prefit {e}", pre[e], None, cv.LINE_COLORS[2 * i], 2))
                    series.append((f"postfit {e}", post[e], None, cv.LINE_COLORS[2 * i + 1], 1))
                out.write(d, cv.overlay_canvas(f"per-bin -2lnL, {r['label']} ({sel})", vlabel, "-2 ln L contribution",
                                               edges, series, draw="HIST",
                                               notes=[f"postfit {r['deviance']:.1f} / ndf {r['ndf']}"]),
                          f"bin_deviance_{sel}_{var}_{lab}")
        except Exception as e:  # pragma: no cover - drawing only
            print(f"  warning: per-bin deviance canvas {sel}/{var}/{lab}: {e}")
    # postfit stacks per era
    for em in ERA_MODELS:
        r = fits.get(em)
        if not r or r['status'] != 'ok':
            continue
        sfs = st.sf_lookup(r, ctx.procs, ctx.eras)
        entry = ctx.sets[sel][2][(r['region'], var)]
        for era in ctx.eras:
            c = cv.stack_canvas(ctx.plotters[era], entry, ctx.var_meta[var], [era], sfs,
                                region_label=f"{rlabel}, postfit {em}", tag=f"post_{sel}_{var}_{em}_{era}")
            out.write(f"{base}/{era}/{sel}", c, f"postfit_{sel}_{var}_{em}_{era}")
    # information
    d = f"{base}/comparison/information"
    for em, info in asi.items():
        if len(info['params']) > 1:
            out.write(d, cv.matrix_canvas(f"expected (Asimov) correlation, {em} ({sel}, {var})", info['params'],
                                          info['params'], info['corr'], -1, 1, "4.2f",
                                          notes=[f"condition number {_fmt(info['condition_number'])}"]),
                      f"asimov_corr_{sel}_{var}_{em}")
        out.write(d, cv.bar_canvas(f"information per SF, {em} ({sel}, {var})", info['params'],
                                   [("expected #sigma_{rel}", list(info['rel']), None, ROOT.kBlue + 1),
                                    ("separating fraction of info", list(info['separation_fraction']), None, ROOT.kRed + 1)],
                                   "", notes=["separating fraction = profiled / unprofiled information"]),
                  f"asimov_info_{sel}_{var}_{em}")
        try:
            meta, hists, _ = ctx.sets[sel]
            vidx, eras_of = st.bin_index_map(meta, hists, fits[em]['region'], var, None)
            nb = ctx.var_meta[var]['bins']
            if len(vidx) == info['per_bin_profiled'].shape[1]:
                series = []
                for i, p in enumerate(info['params']):
                    tot = info['per_bin_profiled'][i].sum()
                    a_ = np.zeros(nb)
                    np.add.at(a_, vidx, info['per_bin_profiled'][i] / tot if tot > 0 else 0)
                    series.append((p, a_, None, cv.LINE_COLORS[i % len(cv.LINE_COLORS)], 1 + (i // len(cv.LINE_COLORS))))
                out.write(d, cv.overlay_canvas(f"share of the separating information per bin, {em} ({sel})", vlabel,
                                               "fraction of profiled information", edges, series, draw="HIST"),
                          f"bin_information_{sel}_{var}_{em}")
        except Exception as e:  # pragma: no cover
            print(f"  warning: information canvas {sel}/{var}/{em}: {e}")


def _draw_stability(ctx, base, var, res, rlabel):
    out = ctx.out
    d = f"{base}/comparison/stability"
    for em, sc in res['stability'].get(var, {}).items():
        if sc.get('status') != 'ok':
            continue
        xl = [str(e['bin']) for e in sc['loo']]
        series = []
        for i, p in enumerate(sc['params']):
            y = [e['shift'][i] if 'shift' in e else None for e in sc['loo']]
            series.append((p, y, None, cv.LINE_COLORS[i % len(cv.LINE_COLORS)]))
        out.write(d, cv.graph_canvas(f"leave-one-bin-out SF shift, {em} ({var}, {rlabel})", xl, series,
                                     "(SF - SF_{nom}) / #sigma_{nom}", ref_lines=[(1, 2), (-1, 2), (0, 3)],
                                     xtitle="removed bin (both eras)"), f"loo_{var}_{em}")
        xl = [f"drop bins {g['bins'][0]}-{g['bins'][1]}" for g in sc['groups']] + [h['subset'] for h in sc['halves']]
        series = []
        for i, p in enumerate(sc['params']):
            y = [g['shift'][i] if 'shift' in g else None for g in sc['groups']] + \
                [h['shift'][i] if 'shift' in h else None for h in sc['halves']]
            series.append((p, y, None, cv.LINE_COLORS[i % len(cv.LINE_COLORS)]))
        out.write(d, cv.graph_canvas(f"interval / statistics subsets, {em} ({var})", xl, series,
                                     "(SF - SF_{nom}) / #sigma_{nom}", ref_lines=[(1, 2), (-1, 2), (0, 3)]),
                  f"subsets_{var}_{em}")
    for em, bres in res['binning'].get(var, {}).items():
        schemes = list(bres)
        params = []
        for b in bres.values():
            if b.get('status') == 'ok':
                params += [p for p in b['params'] if p not in params]
        series = []
        for i, p in enumerate(params):
            y = [b['theta'][b['params'].index(p)] if b.get('status') == 'ok' and p in b['params'] else None for b in bres.values()]
            ey = [b['err'][b['params'].index(p)] if b.get('status') == 'ok' and p in b['params'] else None for b in bres.values()]
            series.append((p, y, ey, cv.LINE_COLORS[i % len(cv.LINE_COLORS)]))
        out.write(d, cv.graph_canvas(f"SFs vs binning, {em} ({var})", schemes, series, "scale factor",
                                     ref_lines=[(1, 2)]), f"binning_{var}_{em}")
    for em, ty in res['toys_screen'].get(var, {}).items():
        if not ty.get('per_param'):
            continue
        pp = ty['per_param']
        out.write(f"{base}/comparison/toys", cv.graph_canvas(
            f"screening toys ({ty['n_valid']}/{ty['n_toys']} valid, {ty['ensemble']}), {em} ({var}): pull mean #pm width",
            list(pp),
            [("pull", [v['pull_mean'] for v in pp.values()], [v['pull_width'] for v in pp.values()], ROOT.kBlue + 1)],
            "pull", ref_lines=[(0, 2), (1, 3), (-1, 3)],
            notes=[f"singular {ty['frac_singular']:.2f}, any bound {ty['frac_any_bound']:.2f}"]),
            f"toys_screen_{var}_{em}")


def _draw_validation_summary(ctx, base, var, val, rlabel):
    d = f"{base}/comparison/validation"
    for em, vv in val.items():
        tg = [t for t in vv if vv[t].get('total')]
        if not tg:
            continue
        pre = [vv[t]['total']['chi2_pre'] / max(vv[t]['total']['ndf_pre'], 1) for t in tg]
        post = [vv[t]['total']['chi2_post'] / max(vv[t]['total']['ndf_post'], 1) for t in tg]
        ctx.out.write(d, cv.graph_canvas(f"SFs from {var} ({em}) in the other observables (same events: correlated)",
                                         [t + (" (obj)" if vv[t].get('object_level') else "") for t in tg],
                                         [("prefit", pre, None, ROOT.kGray + 2), ("postfit", post, None, ROOT.kRed + 1)],
                                         "#chi^{2}/ndf", ref_lines=[(1, 2)]), f"validation_chi2_from_{var}_{em}")


def _draw_toys(ctx, base, var, res):
    d = f"{base}/comparison/toys"
    for em, ens in (res.get('toys_extended') or {}).items():
        f = res['fits']['native'][var][em]
        for name, ty in ens.items():
            if not ty.get('per_param'):
                continue
            tag = f"{var}_{em}_{name}"
            ctx.out.write(d, cv.pulls_grid_canvas(f"toy pulls ({ty['n_valid']}/{ty['n_toys']} valid), {em} ({var}), "
                                                  f"{name} ensemble",
                                                  OrderedDict((p, v['pulls']) for p, v in ty['per_param'].items())),
                          f"toys_pulls_{tag}")
            dv = ty['deviance']
            hi = max(max(dv) if dv else 1, f['deviance']) * 1.1
            g = ty['gof']
            ctx.out.write(d, cv.hist1d_canvas(
                f"toy -2lnL at the minimum, {em} ({var}), {name} ensemble", "-2 ln L", dv, 40, 0, hi,
                lines=[(f['deviance'], ROOT.kRed + 1)],
                notes=[f"observed {f['deviance']:.1f}, ndf {g['ndf']}",
                       f"approx. toy p = {g['p_observed']:.3f} ({g['n_effective']} toys)" if g['estimates_observed_p']
                       else "not a GoF p of the observed data (see report)",
                       f"KS p vs #chi^{{2}}(ndf) = {_fmt(g.get('ks_p_vs_chi2_ndf'))}"]),
                f"toys_gof_{tag}")
            pp = ty['per_param']
            ctx.out.write(d, cv.graph_canvas(
                f"toy closure, {em} ({var}), {name}: pull mean #pm width, 68% coverage", list(pp),
                [("pull mean #pm width", [v['pull_mean'] for v in pp.values()], [v['pull_width'] for v in pp.values()],
                  ROOT.kBlue + 1),
                 ("coverage(|pull|<1)", [v['coverage_68'] for v in pp.values()],
                  [v['coverage_68_err'] for v in pp.values()], ROOT.kGreen + 2),
                 ("fraction at bound", [v['frac_at_bound'] for v in pp.values()], None, ROOT.kRed + 1)],
                "", ref_lines=[(0, 2), (0.683, 3), (1, 3), (-1, 3)],
                notes=[f"injected: observed best fit; valid fits {ty['n_valid']}/{ty['n_toys']}"]),
                f"toys_closure_{tag}")


def _draw_partition(ctx, base, part):
    d = f"{base}/comparison/partition"
    for var, ems in part.items():
        if not isinstance(ems, dict) or var in PARTITION_META:
            continue
        for em, row in ems.items():
            fa, fb = row.get('fit_A'), row.get('fit_B')
            if not fa or not fb or fa.get('status') != 'ok' or fb.get('status') != 'ok':
                continue
            ctx.out.write(d, cv.graph_canvas(f"SFs on independent event halves, {em} ({var}, common selection)",
                                             fa['params'], [("half A", list(fa['theta']), list(fa['err']), ROOT.kBlue + 1),
                                                            ("half B", [fb['theta'][fb['params'].index(p)] if p in fb['params'] else None for p in fa['params']],
                                                             [fb['err'][fb['params'].index(p)] if p in fb['params'] else None for p in fa['params']], ROOT.kRed + 1)],
                                             "scale factor", ref_lines=[(1, 2)]), f"halves_{var}_{em}")
            ap = row.get('apply_A_to_B') or {}
            tg = [t for t in ap if ap[t].get('total')]
            if tg:
                ctx.out.write(d, cv.graph_canvas(f"SFs of half A applied to half B, {em} (from {var})", tg,
                                                 [("prefit", [ap[t]['total']['chi2_pre'] / max(ap[t]['total']['ndf_pre'], 1) for t in tg], None, ROOT.kGray + 2),
                                                  ("postfit", [ap[t]['total']['chi2_post'] / max(ap[t]['total']['ndf_post'], 1) for t in tg], None, ROOT.kRed + 1)],
                                                 "#chi^{2}/ndf (half B)", ref_lines=[(1, 2)]), f"apply_A_to_B_from_{var}_{em}")


def _draw_missing_run3(ctx, base, mr):
    if not mr or not mr.get('processes'):
        return
    for em in ERA_MODELS:
        row = mr.get(em) or {}
        sh = row.get('shift_in_sigma')
        if sh:
            ctx.out.write(f"{base}/comparison/missing_run3", cv.graph_canvas(
                f"HYPOTHETICAL Run 3 {'+'.join(mr['processes'])} (Run 2 x L3/L2): SF shifts, {em}", list(sh),
                [("shift", list(sh.values()), None, ROOT.kMagenta + 1)], "#Delta SF / #sigma",
                ref_lines=[(0, 2), (1, 3), (-1, 3)], notes=["sensitivity study, not a prediction"]),
                f"hypothetical_run3_{em}")


def _draw_veto(ctx, base, kind, veto, cands, rlabel):
    d = f"{base}/comparison/veto"
    rows = veto['rows']
    for era in ctx.eras:
        rr = [r for r in rows if r['era'] == era]
        if not rr:
            continue
        ctx.out.write(d, cv.bar_canvas(f"removed by nBaseLinePhotons == 0 ({era}, {rlabel})", [r['process'] for r in rr],
                                       [("weighted fraction removed", [r['fraction_removed'] for r in rr], None, ROOT.kRed + 1)],
                                       "fraction removed", ymin=0, ymax=1.05), f"veto_removed_{era}")
    for var in cands:
        key = (f"{kind}_noveto", var)
        if key not in ctx.a['native']:
            continue
        for era in ctx.eras:
            c = cv.stack_canvas(ctx.plotters[era], ctx.a['native'][key], ctx.var_meta[var], [era], None,
                                region_label=f"{rlabel}, before #gamma veto", tag=f"noveto_{var}_{era}")
            ctx.out.write(f"{base}/{era}/native", c, f"prefit_noveto_{var}_{era}")


# --------------------------------------------------------------------------- #
# Analysis-level                                                              #
# --------------------------------------------------------------------------- #
def cutflow(ctx, audit):
    """Event-level cutflow: audit-pass steps (all entries) then region steps from the records."""
    a, cfg = ctx.a, ctx.cfg
    mode = a['mode']
    acc = OrderedDict()

    def add(step, sample, era, n, sw, sw2):
        r = acc.setdefault((step, sample, era), [0, 0.0, 0.0])
        r[0] += n
        r[1] += sw
        r[2] += sw2

    for f, fm in a['files'].items():
        au = audit.get(f)
        if not au or au.get('error'):
            continue
        sample = 'data' if fm['is_data'] else fm['process']
        s = 1.0 if fm['is_data'] else fm['lumi'] * fm['scale']
        for step, (n, sw, sw2) in au['cutflow'].items():
            if step[0] in '012345' or f" {mode}" in step:
                add(step.replace(f" {mode}:", ":"), sample, fm['era'], n, sw * s, sw2 * s * s)
    order = ['photon', 'hadronic_noveto', 'hadronic', 'leptonic_noveto', 'leptonic']
    for k, r in enumerate(order):
        recs = a['records'].get(r, {})
        for kind in ('mc', 'data'):
            for f, rec in recs.get(kind, {}).items():
                fm = a['files'][f]
                sample = 'data' if fm['is_data'] else fm['process']
                add(f"8.{k}a region {r}", sample, fm['era'], len(rec['w']), float(rec['w'].sum()), float((rec['w'] ** 2).sum()))
                m = common_mask(rec, mode, a['isr_pt_cut'])
                add(f"8.{k}b region {r} + common selection", sample, fm['era'], int(m.sum()),
                    float(rec['w'][m].sum()), float((rec['w'][m] ** 2).sum()))
    rows = [{'step': s, 'sample': p, 'era': e, 'events': v[0], 'sumw': v[1], 'sumw2': v[2]}
            for (s, p, e), v in sorted(acc.items())]
    return rows


def draw_cutflow(ctx, rows):
    steps = list(OrderedDict.fromkeys(r['step'] for r in rows))
    procs = [p for p in ctx.procs]
    col = _colors(ctx, procs)
    for era in ctx.eras:
        data = [sum(r['sumw'] for r in rows if r['step'] == s and r['era'] == era and r['sample'] == 'data') for s in steps]
        mc = [sum(r['sumw'] for r in rows if r['step'] == s and r['era'] == era and r['sample'] != 'data') for s in steps]
        mce = [np.sqrt(sum(r['sumw2'] for r in rows if r['step'] == s and r['era'] == era and r['sample'] != 'data')) for s in steps]
        series = [("data", data, None, ROOT.kBlack), ("total MC", mc, mce, ROOT.kRed + 1)]
        for p in procs:
            series.append((p, [sum(r['sumw'] for r in rows if r['step'] == s and r['era'] == era and r['sample'] == p)
                               for s in steps], None, col[p]))
        ratio = [d / m if m > 0 else None for d, m in zip(data, mc)]
        rerr = [((d / m) * np.sqrt(1 / d + (e / m) ** 2)) if m > 0 and d > 0 else None for d, m, e in zip(data, mc, mce)]
        ctx.out.write(f"{ctx.name}/inputs", cv.ratio_canvas(f"cutflow {ctx.name} {era}", steps, series, ratio, rerr,
                                                            "events (MC weighted)"), f"cutflow_{era}")


def draw_overlaps(ctx, ov, names):
    M = np.full((len(names), len(names)), np.nan)
    W = np.full((len(names), len(names)), np.nan)
    for r in ov:
        i, j = names.index(r['a']), names.index(r['b'])
        M[i, j] = M[j, i] = r['data_events']
        W[i, j] = W[j, i] = r['mc_weight']
    ctx.out.write(f"{ctx.name}/inputs", cv.matrix_canvas(f"data events shared by two regions ({ctx.name})", names, names,
                                                         M, fmt="6.0f"), "overlap_data")
    ctx.out.write(f"{ctx.name}/inputs", cv.matrix_canvas(f"MC weight shared by two regions ({ctx.name})", names, names,
                                                         W, fmt="6.1f"), "overlap_mc")


def study_analysis(ctx, audit, regions=None):
    """All regions of one analysis plus the analysis-level inputs."""
    a = ctx.a
    res = OrderedDict(name=ctx.name, config=a['config'], mode=a['mode'], isr_pt_cut=a['isr_pt_cut'],
                      processes=ctx.procs, eras=ctx.eras, run_lumis=a['run_lumis'], production=ctx.production,
                      regions_def=OrderedDict((r, {k: v for k, v in rc.items()}) for r, rc in a['regions'].items()),
                      excluded=a['excluded'], event_ids=a['event_ids'], id_summary=a['id_summary'])
    print(f"[{ctx.name}] cutflow / overlaps")
    res['cutflow'] = cutflow(ctx, audit)
    draw_cutflow(ctx, res['cutflow'])
    names = ['photon', 'hadronic', 'leptonic', 'hadronic_noveto', 'leptonic_noveto']
    names = [n for n in names if n in a['records']]
    res['overlaps'] = st.region_overlaps(a, names)
    draw_overlaps(ctx, res['overlaps'], names)
    res['regions'] = OrderedDict()
    for kind in REGION_KINDS:
        if regions and kind not in regions:
            continue
        print(f"[{ctx.name}] region {kind}")
        try:
            res['regions'][kind] = study_region(ctx, kind)
        except Exception as e:   # recorded and reported as an incomplete run, never silent
            import traceback
            print(f"ERROR [{ctx.name}] region {kind}: {e}")
            res['regions'][kind] = OrderedDict(kind=kind, status=f"study failed: {e}",
                                               error=traceback.format_exc())
    return res


def completion_issues(name, res, requested=None):
    """
    Execution completeness of one analysis (not the physics verdict: a fully executed study
    may conclude that no model is adequate). Returns {'failed', 'partial', 'notes'} lists:

      failed   a required computation raised an error, or an integrity check failed
               (region study exception, validation without its combined-era result, stability
               scan error, record / loader or reference-yield mismatch, requested region absent)
      partial  a requested study could not be performed (no MC, unblinded region without data,
               production fit refused, screening / extended toys or the focus stability scan
               not run, extended toys without valid fits, no event-partition validation)
      notes    expected omissions (data blinded by configuration)
    """
    failed, partial, notes = [], [], []
    regions = res.get('regions', {})
    for kind in (requested or REGION_KINDS):
        if kind not in regions:
            failed.append(f"[{name}] {kind}: requested region was not studied")
    for kind, R in regions.items():
        tag = f"[{name}] {kind}"
        if R.get('error'):
            failed.append(f"{tag}: {R['status']}")
            continue
        if R.get('status'):
            partial.append(f"{tag}: {R['status']}: no study performed")
            continue
        if not R.get('has_data'):
            if R.get('blind'):
                notes.append(f"{tag}: data blinded by configuration: no fits (expected)")
            else:
                partial.append(f"{tag}: no data for at least one candidate observable: fits, toys and validation "
                               f"not performed")
            continue
        for var, fits in R['fits']['native'].items():
            for em in ERA_MODELS:
                f = fits.get(em) or {}
                if f.get('status') != 'ok':
                    partial.append(f"{tag}: native {em} fit of {var} not performed: {f.get('error')}")
                    continue
                if not (R.get('toys_screen') or {}).get(var, {}).get(em):
                    partial.append(f"{tag}: screening toys of {var} ({em}) not run")
        for var, ems in R.get('validation', {}).items():
            for em, vv in ems.items():
                for tgt, v in vv.items():
                    if 'total' not in v:
                        failed.append(f"{tag}: validation {var} -> {tgt} ({em}) has no combined-era result")
        focus = R.get('focus_var')
        if focus:
            fits = R['fits']['native'].get(focus) or {}
            ext = R.get('toys_extended') or {}
            for em in ERA_MODELS:
                if (fits.get(em) or {}).get('status') != 'ok':
                    continue
                if not ext.get(em):
                    partial.append(f"{tag}: extended toys of {focus} ({em}) not run")
                for e, ty in (ext.get(em) or {}).items():
                    if ty.get('status') != 'ok':
                        partial.append(f"{tag}: {e} toys of {focus} ({em}): {ty.get('status')} {ty.get('reason', '')}")
                sc = (R.get('stability') or {}).get(focus, {}).get(em) or {}
                if sc.get('status') == 'skipped':
                    partial.append(f"{tag}: stability scan of {focus} ({em}) skipped: {sc.get('reason')}")
        for var, ems in (R.get('stability') or {}).items():
            for em, sc in ems.items():
                if sc.get('status') == 'failed':
                    failed.append(f"{tag}: stability scan of {var} ({em}) failed: {sc.get('error')}")
        part = R.get('partition') or {}
        if not any(k not in PARTITION_META for k in part):
            partial.append(f"{tag}: no event-partition validation")
    if res.get('record_check'):
        failed.append(f"[{name}] {len(res['record_check'])} record / loader histogram mismatches")
    for r in res.get('reference') or []:
        if r.get('status') != 'ok':
            failed.append(f"[{name}] reference yields {r['region']}:{r['var']} {r['sample']}: {r['status']}")
    return OrderedDict(failed=failed, partial=partial, notes=notes)


def completion_status(parts):
    """'failed' / 'partial' / 'complete' from completion_issues-style lists."""
    return 'failed' if parts.get('failed') else 'partial' if parts.get('partial') else 'complete'
