#!/usr/bin/env python3
"""
Diagnostics of the MC scale-factor likelihood of fit_sfs.py.

Reads the data and MC of main.py input configs with the production loader, builds its own
histograms (production "native" selection and a common selection of all candidate
observables), and evaluates the production likelihood (fit_sfs.build_model / fit_sfs.fit):
composition and MC statistics, template separation and expected information, shared vs
independent Run 2 / Run 3 SFs, weak-process treatments, binning and interval stability,
toy closure, validation in other observables and on an independent event-level partition.

Outputs (in --output):
  sf_diagnostics.root               every plot as a TCanvas
  sf_diagnostics_results.json       all numbers
  tables/*.csv                      flat tables
  sf_diagnostics_summary.html       per-case assessment

Example
-------
  python3 diagnose_sf_fit.py \\
      --configs config/datamc_v34_run2run3_splitlumi.yaml \\
                config/datamc_v34_run2run3_splitlumi_uncompressed.yaml \\
      --output sf_diagnostics --workers 8

  # rerun the statistics only (histograms and records from the cache of the first run)
  python3 diagnose_sf_fit.py --configs ... --output sf_diagnostics --toys-extended 500
"""

import argparse
import datetime
import os
import pickle
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning
ROOT.TH1.AddDirectory(False)

from src.config import AnalysisConfig  # noqa: E402
from src.input_config import _expand_globs  # noqa: E402
from src.sfdiag import inputs as di  # noqa: E402
from src.sfdiag import report as rp  # noqa: E402
from src.sfdiag import stats as st  # noqa: E402
from src.sfdiag import study as sd  # noqa: E402
from src.sfdiag import canvases as cv  # noqa: E402


def parse_arguments(argv=None):
    p = argparse.ArgumentParser(description="Diagnostics of the MC scale-factor likelihood (fit_sfs.py)",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('--configs', nargs='+', required=True, metavar='YAML', help='main.py input configs')
    p.add_argument('--output', default='sf_diagnostics', help='Output directory (default: sf_diagnostics)')
    p.add_argument('--workers', type=int, default=1, help='Parallel worker processes for reading files')
    p.add_argument('--regions', nargs='+', choices=di.REGION_KINDS, default=None, help='Only these regions')
    p.add_argument('--toys', type=int, default=30, help='Screening toys per variable and era model (default 30)')
    p.add_argument('--toys-degenerate', type=int, default=10,
                   help='Screening toys for fits with SFs at a bound or a singular Hessian (slow, default 10)')
    p.add_argument('--toys-extended', type=int, default=200,
                   help='Toys for the recommended variable of each region (default 200)')
    p.add_argument('--seed', type=int, default=12345, help='Base random seed (toys)')
    p.add_argument('--cache', default=None, metavar='DIR',
                   help='Cache directory for the loaded histograms / records (default: <output>/cache); '
                        'reused when the inputs are unchanged')
    p.add_argument('--no-cache', action='store_true', help='Always re-read the inputs')
    p.add_argument('--reference-yields', nargs='+', default=[], metavar='CONFIG=YIELDS',
                   help='Regression: compare the native histograms with an existing main.py yields file '
                        '(config path = yields path)')
    p.add_argument('--event-id-branches', nargs=3, default=None, metavar=('RUN', 'LUMI', 'EVENT'),
                   help='Event-identifier branches (default: try common names)')
    p.add_argument('--verbose', action='store_true')
    return p.parse_args(argv)


def git_info():
    def run(*a):
        try:
            return subprocess.run(a, capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
        except Exception:
            return None
    return {'git_commit': run('git', 'rev-parse', 'HEAD'),
            'git_dirty': bool(run('git', 'status', '--porcelain', '--untracked-files=no'))}


# --------------------------------------------------------------------------- #
# Audit                                                                       #
# --------------------------------------------------------------------------- #
def run_audit(cfgs, opts, cache_dir):
    files = OrderedDict()
    region_cuts = OrderedDict()
    for c in cfgs:
        for f, m in c['files'].items():
            files.setdefault(f, m['is_data'])
        for r, rc in c['regions'].items():
            region_cuts[f"{c['name']}:{r}"] = rc['cut']
    ids = [tuple(opts.event_id_branches)] if opts.event_id_branches else di.DEFAULT_ID_BRANCHES
    key = None
    if cache_dir:
        import hashlib, json
        key = hashlib.sha1(json.dumps([sorted(files.items()), sorted(region_cuts.items()), ids,
                                       [(f, os.path.getmtime(f) if os.path.exists(f) else None) for f in files]],
                                      default=str).encode()).hexdigest()
        path = Path(cache_dir) / f"audit_{key[:16]}.pkl"
        if path.exists():
            print(f"Input audit: from cache {path}")
            with open(path, 'rb') as fh:
                return pickle.load(fh)
    print(f"Input audit: {len(files)} files")
    audit = di.scan_inputs(files, cfgs[0]['tree'], region_cuts, ids, workers=opts.workers)
    if cache_dir:
        with open(path, 'wb') as fh:
            pickle.dump(audit, fh, protocol=4)
    return audit


def audit_view(audit, name):
    """Per-analysis view: region:<r> keys without the analysis prefix."""
    out = {}
    for f, a in audit.items():
        b = dict(a)
        b['missing'] = {(k.replace(f"region:{name}:", "region:") if k.startswith(f"region:{name}:") else k): v
                        for k, v in (a.get('missing') or {}).items()}
        out[f] = b
    return out


def audit_summary(cfgs, audit, adatas):
    """File table and the list of input issues."""
    rows, issues = [], []
    for c in cfgs:
        raw = c['raw_patterns']
        for sect in ('background', 'data'):
            for g in raw.get(sect) or []:
                pats = g.get('files', []) if isinstance(g, dict) else [g]
                pats = [pats] if isinstance(pats, (str, dict)) else pats
                for pat in pats:
                    pat = pat['path'] if isinstance(pat, dict) else pat
                    n = len(_expand_globs([pat], raw.get('base_dir')))
                    if n == 0:
                        issues.append(f"[{c['name']}] {sect} pattern matches no file: {pat}")
        # files listed in more than one group / pattern
        seen = {}
        for sect, groups in (('background', c['bg_groups']), ('data', c['data_groups'])):
            for g in groups:
                for f in g['files']:
                    seen.setdefault(f, []).append(g.get('name'))
        for f, gs in seen.items():
            if len(gs) > 1:
                issues.append(f"[{c['name']}] {Path(f).name} is listed {len(gs)} times ({gs}); "
                              f"main.py loads it once (unique_files_from_groups)")
        names = {}
        for f in seen:
            names.setdefault(Path(f).name, []).append(f)
        for n, fs in names.items():
            if len(fs) > 1:
                issues.append(f"[{c['name']}] the same file name in several directories: {fs}")
        # data-taking eras covered by several patterns
        cov = {}
        for g in (c['raw_patterns'].get('data') or []):
            pats = g.get('files', []) if isinstance(g, dict) else [g]
            for pat in ([pats] if isinstance(pats, str) else pats):
                pat = pat['path'] if isinstance(pat, dict) else pat
                for f in _expand_globs([pat], c['raw_patterns'].get('base_dir')):
                    for y in c['files'].get(f, {}).get('years') or []:
                        cov.setdefault(y, {}).setdefault(pat, []).append(f)
        for y, pats in cov.items():
            if len(pats) > 1:
                issues.append(f"[{c['name']}] data year 20{y} comes from {len(pats)} patterns "
                              f"({', '.join(pats)}): verify they are distinct run eras (otherwise data events are "
                              f"double counted; event-level check needs event identifiers)")
    for f, a in audit.items():
        meta = next((c['files'][f] for c in cfgs if f in c['files']), {})
        w = a.get('weights') or {}
        sel = OrderedDict()
        for an, ad in adatas.items():
            for r, rr in ad['records'].items():
                rec = rr['data' if meta.get('is_data') else 'mc'].get(f)
                if rec is not None:
                    sel[f"{an}:{r}"] = {'events': int(len(rec['w'])), 'sumw': float(rec['w'].sum()),
                                        'sumw2': float((rec['w'] ** 2).sum())}
        missing = {k: v for k, v in (a.get('missing') or {}).items() if v}
        row = OrderedDict(file=Path(f).name, path=f, process=meta.get('process'), is_data=meta.get('is_data'),
                          era=meta.get('era'), years=meta.get('years'), campaign=meta.get('campaign'),
                          lumi=meta.get('lumi'), scale=meta.get('scale'), entries=a.get('entries'),
                          sumw_raw=w.get('sumw'), sumw2_raw=w.get('sumw2'),
                          neg_weight_events=w.get('n_neg'), neg_weight_sum=w.get('sumw_neg'),
                          w_max=w.get('wmax'), w_min=w.get('wmin'),
                          events_w_ge_cut=w.get('n_wcut'), sumw_w_ge_cut=w.get('sumw_wcut'),
                          id_branches=a.get('id_branches'), missing=missing, error=a.get('error'), selected=sel)
        rows.append(row)
        if a.get('error'):
            issues.append(f"{Path(f).name}: {a['error']}")
        if missing.get('triggers'):
            if not a.get('triggers_available'):
                issues.append(f"{Path(f).name}: NO MET trigger branch present: the loader applies no trigger requirement")
            else:
                issues.append(f"{Path(f).name}: trigger branch(es) {missing['triggers']} missing: the loader drops them "
                              f"from the trigger OR (applied: {a['triggers_available']})")
        if missing.get('baseline_flags'):
            issues.append(f"{Path(f).name}: baseline flag branch(es) missing, NOT applied by the loader: "
                          f"{missing['baseline_flags']}")
        for k, v in missing.items():
            if k.startswith('region:'):
                issues.append(f"{Path(f).name}: {k} reads missing branch(es) {v}: excluded from that region "
                              f"(never treated as passing)")
        if w and w.get('sumw') and w.get('sumw_wcut') and abs(w['sumw_wcut'] / w['sumw']) > 0.01:
            issues.append(f"{Path(f).name}: {100 * w['sumw_wcut'] / w['sumw']:.1f}% of the weight has evtFillWgt >= "
                          f"{AnalysisConfig.EVT_WGT_CUT:g} and is removed by the baseline weight cut")
        if w and w.get('n_neg'):
            issues.append(f"{Path(f).name}: {w['n_neg']} negative-weight events (sum {w['sumw_neg']:.3g})")
    return rows, issues


# --------------------------------------------------------------------------- #
# Loading with cache                                                          #
# --------------------------------------------------------------------------- #
def load_cached(acfg, audit, opts, cache_dir):
    fp = di.config_fingerprint(acfg)
    path = Path(cache_dir) / f"{acfg['name']}_{fp[:16]}.pkl" if cache_dir else None
    if path and path.exists():
        print(f"[{acfg['name']}] histograms and records from cache {path}")
        with open(path, 'rb') as fh:
            return pickle.load(fh)
    ids = [tuple(opts.event_id_branches)] if opts.event_id_branches else None
    adata = di.load_analysis(acfg, audit_view(audit, acfg['name']), workers=opts.workers, verbose=opts.verbose,
                             id_branches=ids)
    if path:
        with open(path, 'wb') as fh:
            pickle.dump(adata, fh, protocol=4)
    return adata


# --------------------------------------------------------------------------- #
# Summary canvases                                                            #
# --------------------------------------------------------------------------- #
def summary_canvases(out, results):
    labels, lr_p, gof_s, gof_i, n_id = [], [], [], [], []
    for an, A in results['analyses'].items():
        for kind, R in A.get('regions', {}).items():
            for var in (R.get('fits', {}).get('native') or {}):
                labels.append(f"{an[:4]} {kind[:3]} {var}")
                l = R['lrt']['native'].get(var)
                lr_p.append(l['p'] if l else None)
                f = R['fits']['native'][var]
                gof_s.append(f['shared'].get('p_gof') if f['shared'].get('status') == 'ok' else None)
                gof_i.append(f['independent'].get('p_gof') if f['independent'].get('status') == 'ok' else None)
                a = R['asimov_summary']['native'].get(var, {}).get('shared', {})
                n_id.append(a.get('n_identified'))
    if labels:
        out.write("summary", cv.bar_canvas("p-values: shared vs independent era SFs (LRT) and GoF", labels,
                                           [("LRT p (shared vs indep)", lr_p, None, ROOT.kBlue + 1),
                                            ("GoF p shared", gof_s, None, ROOT.kRed + 1),
                                            ("GoF p independent", gof_i, None, ROOT.kGreen + 2)],
                                           "p-value", logy=True, ymin=1e-12, ymax=10), "pvalues_overview")
        out.write("summary", cv.bar_canvas("expected number of identified SFs (shared era, Asimov)", labels,
                                           [("identified SFs", n_id, None, ROOT.kBlue + 1)], "count", ymin=0),
                  "identified_overview")
    for an, A in results['analyses'].items():
        for kind, R in A.get('regions', {}).items():
            rk = R.get('ranking') or []
            if rk:
                out.write("summary", cv.bar_canvas(
                    f"variable ranking {an} {kind} (lower = better)", [r['var'] for r in rk],
                    [("median expected #sigma_{rel} (major, shared)", [r.get('median_rel_major_shared') for r in rk],
                      None, ROOT.kBlue + 1),
                     ("max |#rho| (shared)", [r.get('max_corr_shared') for r in rk], None, ROOT.kRed + 1)], "",
                    ymin=0, notes=[f"recommended: {R.get('recommended_var')}"]), f"ranking_{an}_{kind}")


# --------------------------------------------------------------------------- #
def main(argv=None):
    opts = parse_arguments(argv)
    t0 = time.time()
    out_dir = Path(opts.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'tables').mkdir(exist_ok=True)
    cache_dir = None if opts.no_cache else Path(opts.cache or out_dir / 'cache')
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    cfgs = []
    for path in opts.configs:
        c = di.resolve_config(path)
        names = [x['name'] for x in cfgs]
        if c['name'] in names:
            c['name'] = f"{c['name']}_{len(cfgs)}"
        cfgs.append(c)
    print("Analyses: " + ", ".join(f"{c['name']} ({c['config']})" for c in cfgs))
    if not any(Path(f).exists() for c in cfgs for f in c['files']):
        sys.exit("Error: none of the configured input files exists here (EOS not mounted?)")

    st.set_workers(opts.workers)
    audit = run_audit(cfgs, opts, cache_dir)
    adatas = OrderedDict()
    for c in cfgs:
        adatas[c['name']] = load_cached(c, audit, opts, cache_dir)

    refs = dict(x.split('=', 1) for x in opts.reference_yields)
    results = OrderedDict()
    file_rows, issues = audit_summary(cfgs, audit, adatas)
    not_executed = []
    if not refs:
        not_executed.append("regression against existing yields files (pass --reference-yields CONFIG=YIELDS)")
    if not all(a['event_ids'] for a in adatas.values()):
        not_executed.append("event-identifier-based duplicate-data check: no (run, lumi, event) branches; "
                            "region overlaps and the validation partition use in-pass event keys instead")
    results['audit'] = {'files': file_rows, 'issues': issues, 'not_executed': not_executed}

    rout = cv.RootOut(out_dir / 'sf_diagnostics.root')
    results['analyses'] = OrderedDict()
    for c in cfgs:
        a = adatas[c['name']]
        ctx = sd.Ctx(a, c, opts, rout)
        res = sd.study_analysis(ctx, audit, opts.regions)
        res['record_check'] = di.check_records(a)
        if res['record_check']:
            print(f"WARNING [{c['name']}]: records disagree with the loader histograms: {res['record_check'][:3]}")
        ref = refs.get(c['config']) or refs.get(str(Path(c['config']).resolve()))
        if ref:
            res['reference'] = di.compare_reference_yields(a, ref)
        results['analyses'][c['name']] = res
    summary_canvases(rout, results)
    st.shutdown_workers()
    n_canvases = rout.n
    rout.close()

    results['provenance'] = OrderedDict(
        **git_info(), timestamp=datetime.datetime.now().isoformat(timespec='seconds'),
        command=" ".join([Path(sys.argv[0]).name] + (argv if argv is not None else sys.argv[1:])),
        configs=[c['config'] for c in cfgs],
        analyses={c['name']: {'mode': c['mode'], 'isr_pt_cut': c['isr_pt_cut'], 'years': c['years'],
                              'run_lumis': c['run_lumis'], 'lumi_total': c['lumi_total'],
                              'regions': {r: rc['cut'] for r, rc in c['regions'].items()},
                              'variables': {v: adatas[c['name']]['var_meta'][v] for v in adatas[c['name']]['candidates']},
                              'production_sf_fit': c['sf_fit'],
                              'manifest': {f: {k: m[k] for k in ('process', 'is_data', 'era', 'lumi', 'scale', 'campaign')}
                                           for f, m in c['files'].items()}}
                  for c in cfgs},
        baseline={'MET_CUT': AnalysisConfig.MET_CUT, 'EVT_WGT_CUT': AnalysisConfig.EVT_WGT_CUT,
                  'RJR_PTS_CUT': AnalysisConfig.RJR_PTS_CUT, 'ISR_RISR_CUT': AnalysisConfig.ISR_RISR_CUT},
        fit={'sf_bounds': [__import__('fit_sfs').SF_MIN, __import__('fit_sfs').SF_MAX], 'mc_stat': 'Barlow-Beeston lite',
             'min_mc': 0.0},
        thresholds=OrderedDict((k, v) for k, v in st.THRESHOLDS.items()),
        settings={'toys': opts.toys, 'toys_degenerate': opts.toys_degenerate, 'toys_extended': opts.toys_extended, 'seed': opts.seed, 'workers': opts.workers,
                  'regions': opts.regions, 'cache': str(cache_dir) if cache_dir else None,
                  'seed_rule': 'crc32(analysis|region|var|era_model[|ext]) mixed with --seed'},
        canvases=n_canvases, runtime_s=round(time.time() - t0, 1))

    rp.write_json(out_dir / 'sf_diagnostics_results.json', results)
    for name, rows in rp.tables(results).items():
        rp.write_csv(out_dir / 'tables' / f"{name}.csv", rows)
    rp.write_html(out_dir / 'sf_diagnostics_summary.html', results)

    print(f"\n{n_canvases} canvases -> {out_dir / 'sf_diagnostics.root'}")
    print(f"report -> {out_dir / 'sf_diagnostics_summary.html'}, numbers -> {out_dir / 'sf_diagnostics_results.json'}")
    for an, A in results['analyses'].items():
        for kind, R in A['regions'].items():
            a = R.get('assessment', {})
            print(f"  {an:<13} {kind:<9} observable {a.get('recommended_observable')}, binning "
                  f"{a.get('recommended_binning')}, Run2/Run3 {a.get('era_model')}; "
                  f"limitations: {'; '.join(a.get('limitations', [])) or 'none flagged'}")
    print(f"done in {time.time() - t0:.0f} s")
    return results


if __name__ == "__main__":
    main()
