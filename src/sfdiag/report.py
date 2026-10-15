"""
Machine-readable and human-readable outputs of the diagnostics:
  <out>/sf_diagnostics_results.json   everything (numbers only; no ROOT objects)
  <out>/tables/*.csv                  flat tables for review
  <out>/sf_diagnostics_summary.html   per-case assessment with the supporting numbers
"""

import csv
import html
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np


def jsonable(o):
    if isinstance(o, dict):
        return OrderedDict((str(k) if not isinstance(k, tuple) else ":".join(map(str, k)), jsonable(v))
                           for k, v in o.items() if k != 'model')
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def write_json(path, results):
    with open(path, 'w') as fh:
        json.dump(jsonable(results), fh, indent=1)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    keys = list(OrderedDict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(jsonable(v)) if isinstance(v, (list, dict, tuple, np.ndarray)) else
                            jsonable(v)) for k, v in r.items()})


# --------------------------------------------------------------------------- #
# Flat tables                                                                 #
# --------------------------------------------------------------------------- #
def tables(results):
    T = OrderedDict((k, []) for k in ('audit_files', 'cutflow', 'composition', 'fits', 'era_tests', 'ranking',
                                      'stability', 'binning', 'toys', 'validation', 'partition', 'veto', 'overlaps',
                                      'weights', 'excluded', 'record_check', 'reference'))
    T['audit_files'] = results.get('audit', {}).get('files', [])
    for an, A in results.get('analyses', {}).items():
        for r in A.get('cutflow', []):
            T['cutflow'].append(dict(analysis=an, **r))
        for r in A.get('overlaps', []):
            T['overlaps'].append({'analysis': an, 'a': r['a'], 'b': r['b'], 'data_events': r['data_events'],
                                  'mc_events': r['mc_events'], 'mc_weight': r['mc_weight']})
        for r in A.get('excluded', []):
            T['excluded'].append(dict(analysis=an, **r))
        for r in A.get('record_check', []):
            T['record_check'].append(dict(analysis=an, **r))
        for r in A.get('reference', []):
            T['reference'].append(dict(analysis=an, **r))
        for kind, R in A.get('regions', {}).items():
            base = {'analysis': an, 'region': kind}
            for sel, vv in R.get('composition', {}).items():
                for var, comp in vv.items():
                    for era, ce in comp.items():
                        for p, pr in ce['processes'].items():
                            T['composition'].append(dict(base, selection=sel, var=var, era=era, process=p,
                                                         **{k: pr[k] for k in ('yield', 'sumw2', 'fraction', 'neff_total',
                                                                               'rel_stat', 'bins_filled', 'bins_sparse',
                                                                               'bins_negative')}))
                        T['composition'].append(dict(base, selection=sel, var=var, era=era, process='DATA',
                                                     **{'yield': ce.get('data'), 'fraction': ce.get('data_over_mc'),
                                                        'bins_filled': ce.get('bins_mc'),
                                                        'bins_sparse': ce.get('data_bins_no_mc')}))
            for sel, vv in R.get('fits', {}).items():
                for var, fits in vv.items():
                    for lab, f in fits.items():
                        if f.get('status') != 'ok':
                            T['fits'].append(dict(base, selection=sel, var=var, fit=lab, status=f.get('status'),
                                                  error=f.get('error')))
                            continue
                        for i, p in enumerate(f['params']):
                            idf = f['identifiability'][p]
                            T['fits'].append(dict(base, selection=sel, var=var, fit=lab, status='ok', param=p,
                                                  sf=f['theta'][i], err=f['err'][i], at_bound=f['at_bound'][i],
                                                  rel_err=idf['rel_err'], max_corr=idf['max_corr'],
                                                  max_corr_with=idf['max_corr_with'], identified=idf['identified'],
                                                  weak=idf['weak'], singular=f['singular'], converged=f['converged'],
                                                  deviance=f['deviance'], ndf=f['ndf'], p_gof=f['p_gof'],
                                                  nbins=f['nbins'], nbins_excluded=f['nbins_excluded'],
                                                  data_excluded=f['data_excluded']))
            for sel, vv in R.get('lrt', {}).items():
                for var, l in vv.items():
                    if l:
                        T['era_tests'].append(dict(base, selection=sel, var=var, test='LRT shared vs independent', **l))
                    for p, e in R.get('era_ratio', {}).get(sel, {}).get(var, {}).items():
                        T['era_tests'].append(dict(base, selection=sel, var=var, test=f'run3-run2 {p}', **e))
            for r in R.get('ranking', []):
                T['ranking'].append(dict(base, **r))
            for var, ems in R.get('stability', {}).items():
                for em, sc in ems.items():
                    if sc.get('status') != 'ok':
                        continue
                    for i, p in enumerate(sc['params']):
                        T['stability'].append(dict(base, var=var, era_model=em, param=p,
                                                   max_loo_shift=sc['max_loo_shift'][i], max_loo_bin=sc['max_loo_bin'][i],
                                                   max_group_shift=sc['max_group_shift'][i]))
            for var, ems in R.get('binning', {}).items():
                for em, bs in ems.items():
                    for s, b in bs.items():
                        q = b.get('quality') or {}
                        for i, p in enumerate(b.get('params', [])):
                            T['binning'].append(dict(base, var=var, era_model=em, scheme=s, param=p, sf=b['theta'][i],
                                                     err=b['err'][i], deviance=b.get('deviance'), ndf=b.get('ndf'), **q))
            for tag, src in (('screen', R.get('toys_screen', {})), ('extended', {R.get('recommended_var'): R.get('toys_extended') or {}})):
                for var, ems in src.items():
                    for em, ty in (ems or {}).items():
                        for p, v in (ty.get('per_param') or {}).items():
                            T['toys'].append(dict(base, kind=tag, var=var, era_model=em, param=p, n_toys=ty['n_toys'],
                                                  seed=ty['seed'], theta_true=ty['theta_true'][ty['params'].index(p)],
                                                  **{k: v[k] for k in ('mean', 'std', 'bias', 'pull_mean', 'pull_width',
                                                                       'coverage_68', 'frac_at_bound')},
                                                  p_gof_toys=ty.get('p_gof_toys'), frac_singular=ty.get('frac_singular')))
            for var, ems in R.get('validation', {}).items():
                for em, vv in ems.items():
                    for tgt, v in vv.items():
                        for era, x in v.items():
                            if isinstance(x, dict):
                                T['validation'].append(dict(base, source_var=var, era_model=em, target=tgt, era=era,
                                                            object_level=v.get('object_level'), **x))
            for var, ems in (R.get('partition') or {}).items():
                if not isinstance(ems, dict) or var in ('event_ids', 'partition_key'):
                    continue
                for em, row in ems.items():
                    for p, z in (row.get('compat_z') or {}).items():
                        T['partition'].append(dict(base, var=var, era_model=em, param=p, z_A_minus_B=z))
            for r in (R.get('veto') or {}).get('rows', []):
                T['veto'].append(dict(base, **r))
            for k, w in (R.get('weights') or {}).items():
                T['weights'].append(dict(base, sample=k, **w))
    return T


# --------------------------------------------------------------------------- #
# HTML                                                                        #
# --------------------------------------------------------------------------- #
CSS = """
:root { --bg:#fff; --fg:#1d1d1f; --muted:#6b6b70; --line:#e2e2e6; --ok:#1f7a3f; --bad:#b3261e; --warn:#8a5a00; --card:#f7f7f9; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#141416; --fg:#ececf0; --muted:#a0a0a8;
  --line:#2c2c31; --ok:#5fcf8a; --bad:#ff7b72; --warn:#e3b341; --card:#1c1c20; } }
body { background:var(--bg); color:var(--fg); font:14px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;
  margin:0 auto; max-width:1180px; padding:16px; }
h1 { font-size:22px; } h2 { font-size:19px; margin-top:34px; border-bottom:1px solid var(--line); }
h3 { font-size:16px; margin-top:24px; } .muted { color:var(--muted); }
table { border-collapse:collapse; font-size:12.5px; margin:8px 0; display:block; overflow-x:auto; max-width:100%; }
th, td { border:1px solid var(--line); padding:3px 7px; text-align:left; vertical-align:top; }
th { background:var(--card); } .card { background:var(--card); border:1px solid var(--line); border-radius:6px; padding:10px 14px; margin:10px 0; }
.sup { color:var(--ok); } .dis { color:var(--bad); } .inc { color:var(--warn); } ul { margin:4px 0 4px 18px; padding:0; }
code { font-size:12px; }
"""


def _e(x):
    return html.escape(str(x))


def _f(x, fmt="{:.3g}"):
    if x is None:
        return "–"
    try:
        return fmt.format(x) if np.isfinite(x) else "–"
    except (TypeError, ValueError):
        return _e(x)


def _table(head, rows):
    out = ["<table><tr>" + "".join(f"<th>{_e(h)}</th>" for h in head) + "</tr>"]
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def _list(items, cls=""):
    if not items:
        return "<p class='muted'>none</p>"
    return f"<ul class='{cls}'>" + "".join(f"<li>{_e(i)}</li>" for i in items) + "</ul>"


def write_html(path, results):
    P = results.get('provenance', {})
    h = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>SF Fit Diagnostics</title><style>{CSS}</style></head><body>",
         "<h1>MC scale-factor likelihood diagnostics</h1>",
         f"<p class='muted'>git {_e(P.get('git_commit'))}{' (dirty)' if P.get('git_dirty') else ''} · "
         f"{_e(P.get('timestamp'))} · configs: {_e(', '.join(P.get('configs', [])))}</p>",
         "<p>All plots are TCanvas objects in <code>sf_diagnostics.root</code>; numbers in "
         "<code>sf_diagnostics_results.json</code> and <code>tables/*.csv</code>. Classification thresholds are listed at the end. "
         "Statements are generated by fixed rules from the diagnostics; "
         "<span class='sup'>supported</span>, <span class='dis'>disfavored</span> and "
         "<span class='inc'>inconclusive</span> are kept apart.</p>"]

    # overview table
    rows = []
    for an, A in results.get('analyses', {}).items():
        for kind, R in A.get('regions', {}).items():
            a = R.get('assessment', {})
            rows.append([_e(an), _e(kind), _e(a.get('recommended_observable')), _e(a.get('recommended_binning')),
                         _e(a.get('era_model')), _e(len(a.get('constrained_backgrounds', []))),
                         _e(len(a.get('backgrounds_needing_treatment', []))),
                         _e("; ".join(a.get('model_adequacy', [])) or "no inadequacy flagged")])
    h.append("<h2>Overview</h2>")
    h.append(_table(["analysis", "region", "observable", "binning", "Run 2 / Run 3 SFs", "# constrained",
                     "# need treatment", "model adequacy"], rows))

    # input audit
    au = results.get('audit', {})
    h.append("<h2>Input audit</h2>")
    issues = au.get('issues', [])
    h.append(_list(issues, 'inc') if issues else "<p>No input issues flagged.</p>")
    if au.get('not_executed'):
        h.append("<div class='card'><b>Not executed</b>" + _list(au['not_executed']) + "</div>")

    for an, A in results.get('analyses', {}).items():
        h.append(f"<h2>{_e(an)} analysis</h2><p class='muted'>{_e(A.get('config'))}; MC luminosity per run "
                 f"{_e(A.get('run_lumis'))}; event identifiers: {'yes' if A.get('event_ids') else 'no (in-pass keys)'}</p>")
        if A.get('record_check'):
            h.append("<p class='dis'>Record / loader mismatch:</p>" + _list([str(x) for x in A['record_check']]))
        if A.get('excluded'):
            h.append("<p class='inc'>Region x file combinations excluded for missing branches:</p>" +
                     _list([f"{x['region']}: {Path(x['file']).name} (missing {', '.join(x['missing'])})"
                            for x in A['excluded']]))
        ov = A.get('overlaps', [])
        if ov:
            h.append("<h3>Region overlaps</h3>" + _table(["region A", "region B", "data events", "MC events", "MC weight"],
                                                         [[_e(r['a']), _e(r['b']), _e(r['data_events']), _e(r['mc_events']),
                                                           _f(r['mc_weight'])] for r in ov]))
        for kind, R in A.get('regions', {}).items():
            h.append(region_html(an, kind, R))

    h.append("<h2>Thresholds and settings</h2>")
    h.append(_table(["name", "value"], [[_e(k), _e(v)] for k, v in P.get('thresholds', {}).items()]))
    h.append(_table(["setting", "value"], [[_e(k), _e(v)] for k, v in P.get('settings', {}).items()]))
    h.append("</body></html>")
    Path(path).write_text("\n".join(h))


def region_html(an, kind, R):
    a = R.get('assessment', {})
    h = [f"<h3>{_e(an)} · {_e(kind)}</h3>"]
    if R.get('status'):
        return h[0] + f"<p class='inc'>{_e(R['status'])}</p>"
    h.append("<div class='card'>")
    h.append(f"<p><b>Recommended observable:</b> {_e(a.get('recommended_observable'))} · "
             f"<b>binning:</b> {_e(a.get('recommended_binning'))} {_e(a.get('recommended_binning_spec') or '')} · "
             f"<b>Run 2 / Run 3:</b> {_e(a.get('era_model'))}</p>")
    h.append("<b class='sup'>Supported</b>" + _list(a.get('supported'), 'sup'))
    h.append("<b class='dis'>Disfavored</b>" + _list(a.get('disfavored'), 'dis'))
    h.append("<b class='inc'>Inconclusive</b>" + _list(a.get('inconclusive'), 'inc'))
    h.append("<b>Independently constrained backgrounds</b>" + _list(a.get('constrained_backgrounds')))
    h.append("<b>Backgrounds needing fixing, constraints or investigation</b>" + _list(a.get('backgrounds_needing_treatment')))
    if a.get('weak_variants'):
        h.append("<b>Weak-process variants</b>" + _list(a['weak_variants']))
    h.append("<b>Correlations (|rho| &gt; 0.5)</b>" + _list(a.get('correlations')))
    for k, lab in (('weakest_direction', 'Principal degeneracy'), ('binning_note', 'Binning'), ('stability', 'Stability'),
                   ('gof', 'Goodness of fit'), ('validation', 'Validation'), ('partition', 'Independent halves')):
        if a.get(k):
            h.append(f"<p><b>{lab}:</b> {_e(a[k])}</p>")
    h.append("<b>Remaining modeling limitations</b>" + _list(a.get('limitations'), 'inc'))
    h.append("</div>")

    # ranking
    rk = R.get('ranking', [])
    if rk:
        h.append(_table(["rank", "var", "identified (shared)", "median σrel major", "max |ρ|", "identified (indep)",
                         "GoF p shared", "GoF p indep", "pathologies / instability"],
                        [[_e(r.get('rank')), _e(r['var']),
                          f"{_e(r.get('n_identified_shared'))}/{_e(r.get('n_params_shared'))}",
                          _f(r.get('median_rel_major_shared')), _f(r.get('max_corr_shared')),
                          f"{_e(r.get('n_identified_indep'))}/{_e(r.get('n_params_indep'))}",
                          _f(r.get('p_gof_shared')), _f(r.get('p_gof_indep')),
                          _e("; ".join((r.get('pathologies') or []) + (r.get('unstable') or [])) or r.get('reason', ''))]
                         for r in rk]))
    # composition of the recommended variable
    rec = R.get('recommended_var')
    comp = (R.get('composition', {}).get('native', {}) or {}).get(rec) if rec else None
    if comp:
        rows = []
        for era, ce in comp.items():
            for p, pr in ce['processes'].items():
                rows.append([_e(era), _e(p), _f(pr['yield']), _f(pr['fraction'], "{:.3f}"), _f(pr['neff_total'], "{:.0f}"),
                             _f(pr['rel_stat'], "{:.3f}"), f"{pr['bins_filled']}/{pr['bins_sparse']}"])
            rows.append([_e(era), "<b>data / MC</b>", _f(ce.get('data')), _f(ce.get('data_over_mc'), "{:.3f}"), "",
                         _f(ce.get('mc_rel_stat'), "{:.3f}"), f"no-MC data bins: {ce.get('data_bins_no_mc', '–')}"])
        h.append(f"<p><b>Composition ({_e(rec)}, native)</b></p>" +
                 _table(["era", "process", "yield", "fraction", "N_eff", "rel. stat", "bins filled/sparse"], rows))
    # fit table for the recommended variable
    fits = (R.get('fits', {}).get('native', {}) or {}).get(rec) if rec else None
    if fits:
        rows = []
        for lab, f in fits.items():
            if f.get('status') != 'ok':
                rows.append([_e(lab), _e(f.get('error')), "", "", "", ""])
                continue
            sfs = ", ".join(f"{p} {v:.3f}±{_f(e, '{:.3f}')}{' [bound]' if b else ''}"
                            for p, v, e, b in zip(f['params'], f['theta'], f['err'], f['at_bound']))
            rows.append([_e(f['label']), _e(sfs), _f(f['deviance'], "{:.1f}"), _e(f['ndf']), _f(f['p_gof']),
                         _e(f"{'singular ' if f['singular'] else ''}{'' if f['converged'] else 'not converged'}")])
        h.append(f"<p><b>Fits ({_e(rec)}, native)</b></p>" +
                 _table(["model", "scale factors", "-2lnL", "ndf", "p", "flags"], rows))
    veto = R.get('veto')
    if veto:
        rows = [[_e(r['process']), _e(r['era']), _e(r['events_before']), _e(r['events_removed']),
                 _f(r['fraction_removed'], "{:.3f}"), _e(r['missing_npho'])] for r in veto['rows']]
        dm = veto['data_mc']
        h.append("<p><b>Photon veto (nBaseLinePhotons == 0)</b></p>" +
                 _table(["process", "era", "events before", "events removed", "weighted fraction removed",
                         "events w/o branch"], rows) +
                 _table(["era", "data/MC before", "data/MC after"],
                        [[_e(e), _f(v['ratio_before'], "{:.3f}"), _f(v['ratio_after'], "{:.3f}")] for e, v in dm.items()]))
    mr = R.get('missing_run3')
    if mr and mr.get('processes'):
        h.append(f"<p><b>Run-2-only processes</b> ({_e(', '.join(mr['processes']))}): Run 2 fraction "
                 f"{_e({k: _f(v, '{:.4f}') for k, v in mr['run2_fraction'].items()})}; "
                 f"{_e(mr['label'])}: shifts " +
                 _e({em: (mr.get(em) or {}).get('shift_in_sigma') for em in ('shared', 'independent')}) + "</p>")
    return "\n".join(h)
