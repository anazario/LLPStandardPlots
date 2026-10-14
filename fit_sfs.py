#!/usr/bin/env python3
"""
Fit MC scale factors to data with a binned Poisson likelihood over a yields file.

Model, per bin i of every fitted (region, variable[, era]) channel:

    nu_i(theta) = sum_k  s_k * T_k,i         s_k = theta_p for a floated process (or process x era),
                                             fixed value (default 1) otherwise
    -2 ln L     = 2 sum_i [ b_i nu_i - n_i + n_i ln(n_i / (b_i nu_i)) ]  +  sum_i (b_i - 1)^2 / d_i^2
                  + sum_p (ln theta_p)^2 / sigma^2                        (optional --prior)

n_i is the data, T_k,i the MC template of component k. The b_i terms are the
Barlow-Beeston-lite treatment of the MC statistical uncertainty: one nuisance per
bin scaling the total MC, constrained by its relative stat error d_i, and profiled
analytically. Without them, a few MC bins with large weights and tiny stat would
pull the fit. -2 ln L is the deviance (0 for a perfect fit), so at the minimum
it is a chi2-like goodness of fit with ndf = (bins used) - (floated parameters).

Choosing channels
-----------------
The scale factors are only constrained where the processes differ in *shape*
(or appear in different regions). Two processes with the same shape in every
fitted channel are degenerate: only their sum is measured, which shows up as a
large negative correlation and large errors.

Each event enters every variable of its region, so fitting several variables of
one region counts the same events several times (too-small errors, double
weight). Use one variable per region (the default: the first variable of each
unblinded region, e.g. RISR). Regions can overlap too -- that has the same
effect -- so prefer disjoint control regions.

With per-era data (data files named by year), every channel is split into
Run 2 and Run 3 by default; that is what makes per-era SFs (--per-era) measurable.

A simultaneous fit assumes each process is off by the same factor in every region.
Regions with very different selections (photon vs SV, ...) mismodel the MC
differently, so one SF per process becomes a compromise that fits none of them:
--per-region fits every region on its own and writes its SFs to the regions:
block of the SF file (read by replot_yields.py).

Examples
--------
  # One SF per process, RISR of every unblinded region
  python fit_sfs.py yields.root -o sfs.yaml

  # Separate Run 2 / Run 3 SFs for QCD, fix ttbar at 1, pick the channels
  python fit_sfs.py yields.root --per-era "QCD multijets" --fix "t#bar{t}+jets=1" \\
      --channels CustomRegion1:isr_risr CustomRegion2:isr_ptisr -o sfs.yaml

  # Separate SFs in every region (one fit per region)
  python fit_sfs.py yields.root --per-region --per-era all -o sfs.yaml

  # Then redraw everything with the fitted SFs
  python replot_yields.py yields.root --sf sfs.yaml
"""

import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from scipy import optimize, stats

from src.yields import read_yields, normalize_label, default_yields_path


class FitError(Exception):
    pass


# sf_fit: YAML keys -> option names (and defaults). CLI flags override the YAML.
FIT_DEFAULTS = OrderedDict([
    ('channels', None), ('float_procs', None), ('fix', []), ('per_era', []), ('era', None),
    ('no_era_split', False), ('prior', None), ('no_mc_stat', False), ('min_mc', 0.0),
    ('output', None), ('replot', False), ('replot_output', None), ('replot_format', None),
    ('rebin', None), ('per_region', False),
])

SF_MIN, SF_MAX = 1e-4, 100.0


def parse_arguments(argv=None):
    p = argparse.ArgumentParser(description="Binned likelihood fit of MC scale factors to data",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('yields', nargs='?', default=None,
                   help='Yields ROOT file written by main.py --yields (default with --config: from its output)')
    p.add_argument('--config', default=None, metavar='YAML',
                   help='main.py input config: fit options from its sf_fit: block, yields file from its output')
    p.add_argument('--channels', nargs='+', default=None, metavar='REGION:VAR',
                   help='Channels to fit (default: first variable of every unblinded region)')
    p.add_argument('--float', nargs='+', default=None, dest='float_procs', metavar='PROC',
                   help='Processes to float (default: all not given in --fix)')
    p.add_argument('--fix', nargs='+', default=None, metavar='PROC=SF',
                   help='Processes held at a fixed SF, e.g. "t#bar{t}+jets=1"')
    p.add_argument('--per-era', nargs='+', default=None, metavar='PROC',
                   help='Processes with one SF per era instead of one overall ("all" for every process)')
    p.add_argument('--per-region', action='store_const', const=True, default=None,
                   help='Fit every region on its own, with its own SFs (written to the regions: block)')
    p.add_argument('--era', nargs='+', default=None, help='Fit only these eras (MC and data)')
    p.add_argument('--no-era-split', action='store_const', const=True, default=None,
                   help='Fit the summed data instead of splitting each channel by era')
    p.add_argument('--prior', type=float, default=None, metavar='SIGMA',
                   help='Log-normal constraint of width SIGMA on every SF (e.g. 0.5); default none')
    p.add_argument('--no-mc-stat', action='store_const', const=True, default=None,
                   help='Ignore the MC statistical uncertainty')
    p.add_argument('--rebin', nargs='+', type=float, default=None, metavar='N|EDGE',
                   help='One integer N: merge every N bins (the last bin takes the remainder); several values: new bin edges (each must be '
                        'an existing edge). Applies to every channel; per variable via the YAML.')
    p.add_argument('--min-mc', type=float, default=None,
                   help='Drop bins whose nominal MC is at or below this value (default 0)')
    p.add_argument('-o', '--output', default=None,
                   help='Write the SFs to this YAML (for replot_yields.py --sf); '
                        'default with --config: <output>_sfs.yaml')
    p.add_argument('--replot', action='store_const', const=True, default=None,
                   help='Also redraw the Data/MC plots with the fitted SFs (needs an SF output file)')
    p.add_argument('--replot-output', default=None, help='Replot directory (default: <output>_sf)')
    return p.parse_args(argv)


def _fix_list(fix):
    """fix: {proc: sf} (YAML) or ["proc=sf", ...] (CLI) -> ["proc=sf", ...]"""
    if isinstance(fix, dict):
        return [f"{k}={v}" for k, v in fix.items()]
    return list(fix or [])


def _as_list(val):
    return [val] if isinstance(val, str) else val


def fit_options(sf_fit=None, **overrides):
    """
    Options namespace from a YAML sf_fit: block plus overrides (None = not given).

    YAML key spellings: channels, float, fix ({proc: sf} or list), per_era, era,
    era_split (bool), prior, mc_stat (bool), min_mc, output, replot (bool),
    replot_output, replot_format, rebin (N | [edges] | {var: N | [edges]}), per_region (bool).
    """
    block = dict(sf_fit or {})
    unknown = set(block) - {'channels', 'float', 'fix', 'per_era', 'era', 'era_split', 'prior',
                            'mc_stat', 'min_mc', 'output', 'replot', 'replot_output', 'replot_format',
                            'rebin', 'per_region'}
    if unknown:
        raise FitError(f"unknown sf_fit key(s): {sorted(unknown)}")
    from_yaml = {
        'channels': _as_list(block.get('channels')),
        'float_procs': _as_list(block.get('float')),
        'fix': _fix_list(block.get('fix')),
        'per_era': _as_list(block.get('per_era')),
        'era': _as_list(block.get('era')),
        'no_era_split': (not block['era_split']) if 'era_split' in block else None,
        'prior': block.get('prior'),
        'no_mc_stat': (not block['mc_stat']) if 'mc_stat' in block else None,
        'min_mc': block.get('min_mc'),
        'output': block.get('output'),
        'replot': block.get('replot'),
        'replot_output': block.get('replot_output'),
        'replot_format': _as_list(block.get('replot_format')),
        'rebin': block.get('rebin'),
        'per_region': block.get('per_region'),
    }
    opts = {}
    for key, default in FIT_DEFAULTS.items():
        val = overrides.get(key)
        if val is None:
            val = from_yaml.get(key)
        if val is None:
            val = default
        opts[key] = val
    opts['fix'] = _fix_list(opts['fix'])
    opts['rebin'] = _normalize_rebin(opts['rebin'])
    opts['per_era'] = opts['per_era'] or []
    return SimpleNamespace(**opts)


def config_paths(config_output, yields_setting=None):
    """Default yields / SF / replot paths of a main.py run with this output setting."""
    stem = config_output[:-5] if config_output.endswith('.root') else config_output
    yields = (yields_setting if isinstance(yields_setting, str)
              else default_yields_path(config_output))
    return yields, f"{stem}_sfs.yaml", f"{stem}_sf"


def match_process(name, processes):
    for p in processes:
        if normalize_label(p) == normalize_label(name):
            return p
    raise FitError(f"unknown process '{name}'; known: {processes}")


def default_channels(meta):
    return [(r['name'], r['vars'][0]) for r in meta['regions'] if not r.get('blind') and r['vars']]


def parse_channels(specs, meta, hists):
    out = []
    for spec in specs:
        if ':' not in spec:
            raise FitError(f"channel '{spec}' is not REGION:VAR")
        region, var = spec.split(':', 1)
        if (region, var) not in hists:
            # allow the branch name as well
            match = [v for v, c in meta['variables'].items() if c['branch'] == var]
            if match and (region, match[0]) in hists:
                var = match[0]
            else:
                avail = [f"{r}:{v}" for r, v in hists]
                raise FitError(f"channel '{spec}' not in the yields file; available:\n  " + "\n  ".join(avail))
        out.append((region, var))
    return out


def _normalize_rebin(rebin):
    """None | N | [edges] | {var: N | [edges]}; the CLI's [N.0] becomes N."""
    if isinstance(rebin, (list, tuple)) and len(rebin) == 1:
        rebin = rebin[0]
    if isinstance(rebin, float):
        if not rebin.is_integer():
            raise FitError(f"rebin {rebin}: a single value must be an integer merge factor")
        rebin = int(rebin)
    if isinstance(rebin, int) and rebin < 1:
        raise FitError(f"rebin {rebin}: the merge factor must be >= 1")
    if isinstance(rebin, dict):
        return {k: _normalize_rebin(v) for k, v in rebin.items()}
    if isinstance(rebin, (list, tuple)):
        edges = [float(e) for e in rebin]
        if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])):
            raise FitError(f"rebin edges {edges} must be at least two increasing values")
        return edges
    return rebin


def rebin_for(rebin, var):
    return rebin.get(var) if isinstance(rebin, dict) else rebin


def rebin_matrix(h, spec, label=None):
    """(new_bins, old_bins) 0/1 matrix merging the bins of h; None = keep the binning.
    Old bins outside the new edges belong to no new bin (not fitted)."""
    if spec is None or spec == 1:
        return None
    nb = h.GetNbinsX()
    ax = h.GetXaxis()
    old = np.array([ax.GetBinLowEdge(i) for i in range(1, nb + 2)])
    if isinstance(spec, int):
        groups = [range(i, min(i + spec, nb)) for i in range(0, nb, spec)]
    else:
        tol = 1e-6 * (old[-1] - old[0])
        idx = []
        for e in spec:
            j = int(np.argmin(np.abs(old - e)))
            if abs(old[j] - e) > tol:
                raise FitError(f"rebin edge {e} of {label or h.GetName()} is not a bin edge "
                               f"(edges: {', '.join(f'{x:g}' for x in old)})")
            idx.append(j)
        groups = [range(a, b) for a, b in zip(idx, idx[1:])]
    M = np.zeros((len(groups), nb))
    for k, g in enumerate(groups):
        M[k, list(g)] = 1.0
    return M


def hist_arrays(h, M=None):
    """(contents, variances) of the in-range bins, merged by the rebin matrix M."""
    n = h.GetNbinsX()
    c = np.array([h.GetBinContent(i) for i in range(1, n + 1)])
    v = np.array([h.GetBinError(i) ** 2 for i in range(1, n + 1)])
    return (c, v) if M is None else (M @ c, M @ v)


class Model:
    """All channels' bins concatenated; components k = (process, era) per channel."""

    def __init__(self, params, comp_param, comp_fixed, T, V, n, bin_channel, prior, mc_stat):
        self.params = params                    # parameter names
        self.comp_param = np.array(comp_param)  # parameter index per component, -1 = fixed
        self.comp_fixed = np.array(comp_fixed)
        self.T, self.V, self.n = T, V, n        # (K, B), (K, B), (B,)
        self.bin_channel = bin_channel
        self.prior = prior
        self.mc_stat = mc_stat

    def scales(self, theta):
        s = self.comp_fixed.copy()
        floated = self.comp_param >= 0
        s[floated] = theta[self.comp_param[floated]]
        return s

    def bin_deviance(self, theta):
        s = self.scales(theta)
        nu = np.maximum(s @ self.T, 1e-12)
        n = self.n
        if self.mc_stat:
            d2 = (s ** 2 @ self.V) / nu ** 2
            # profiled BB-lite nuisance: beta^2 + beta (nu d2 - 1) - n d2 = 0
            a = nu * d2 - 1.0
            beta = np.where(d2 > 0, 0.5 * (-a + np.sqrt(a * a + 4.0 * n * d2)), 1.0)
            beta = np.maximum(beta, 1e-12)
            con = np.where(d2 > 0, (beta - 1.0) ** 2 / np.where(d2 > 0, d2, 1.0), 0.0)
        else:
            beta, con = 1.0, 0.0
        mu = beta * nu
        log_term = np.where(n > 0, n * np.log(np.where(n > 0, n, 1.0) / mu), 0.0)
        return 2.0 * (mu - n + log_term) + con

    def deviance(self, theta):
        d = self.bin_deviance(theta).sum()
        if self.prior:
            d += np.sum(np.log(theta) ** 2) / self.prior ** 2
        return d


def build_model(meta, hists, channels, args):
    processes = meta['processes']
    fixed = {}
    for spec in args.fix:
        if '=' not in spec:
            raise FitError(f"--fix '{spec}' is not PROC=SF")
        name, val = spec.rsplit('=', 1)
        fixed[match_process(name, processes)] = float(val)
    floated = ([match_process(p, processes) for p in args.float_procs] if args.float_procs
               else [p for p in processes if p not in fixed])
    if any(normalize_label(p) == 'all' for p in args.per_era):
        per_era = set(floated)
    else:
        per_era = {match_process(p, processes) for p in args.per_era}

    eras = args.era or meta['eras']
    data_eras = {era for (r, v) in channels for era in hists[(r, v)]['data']}
    split = (not args.no_era_split and set(eras) <= data_eras and 'all' not in eras)
    if per_era - set(fixed) and not split:
        print("Warning: per-era SFs without per-era data: Run 2 and Run 3 MC have near-identical shapes, "
              "so only their sum is constrained.")

    params, comp_param, comp_fixed, T_rows, V_rows = [], [], [], [], []
    n_all, bin_channel, chan_labels = [], [], []
    region_count = OrderedDict()
    for region, var in channels:
        region_count[region] = region_count.get(region, 0) + 1
    rebin = getattr(args, 'rebin', None)
    matrices = {}
    for c in channels:
        if hists[c]['data']:
            first = next(iter(hists[c]['data'].values()))
            matrices[c] = rebin_matrix(first, rebin_for(rebin, c[1]), f"{c[0]}:{c[1]}")
    nbins_total = sum(len(hist_arrays(next(iter(hists[c]['data'].values())), matrices[c])[0])
                      * (len(eras) if split else 1) for c in matrices)
    offset = 0
    outside_data = 0.0
    for region, var in channels:
        entry = hists[(region, var)]
        if not entry['data']:
            print(f"Warning: {region}:{var} has no data (blinded?); skipped")
            continue
        M = matrices[(region, var)]
        for era_group in ([[e] for e in eras] if split else [eras]):
            data = [hist_arrays(h, M)[0] for e, h in entry['data'].items() if e in era_group]
            if M is not None:
                outside_data += sum(hist_arrays(h)[0].sum() for e, h in entry['data'].items()
                                    if e in era_group) - sum(d.sum() for d in data)
            if not data:
                continue
            n = np.sum(data, axis=0)
            nb = len(n)
            label = f"{region}:{var}" + (f":{era_group[0]}" if split else "")
            for (proc, era), h in entry['bkg'].items():
                if era not in era_group:
                    continue
                t, v = hist_arrays(h, M)
                row_t = np.zeros(nbins_total)
                row_v = np.zeros(nbins_total)
                row_t[offset:offset + nb] = t
                row_v[offset:offset + nb] = v
                if proc in floated:
                    pname = f"{proc}:{era}" if proc in per_era else proc
                    if pname not in params:
                        params.append(pname)
                    comp_param.append(params.index(pname))
                    comp_fixed.append(1.0)
                else:
                    comp_param.append(-1)
                    comp_fixed.append(fixed.get(proc, 1.0))
                T_rows.append(row_t)
                V_rows.append(row_v)
            n_all.append(n)
            bin_channel.extend([label] * nb)
            chan_labels.append(label)
            offset += nb

    if not params:
        raise FitError("nothing to fit (no floated process has MC in the fitted channels)")

    T = np.array(T_rows)[:, :offset]
    V = np.array(V_rows)[:, :offset]
    n = np.concatenate(n_all)
    bin_channel = np.array(bin_channel)

    if outside_data > 0.5:
        print(f"Note: {outside_data:.0f} data events lie outside the rebin edges and are not fitted")
    nominal = T.sum(axis=0)
    keep = nominal > args.min_mc
    dropped_data = n[~keep].sum()
    if dropped_data > 0:
        print(f"Note: {dropped_data:.0f} data events are in bins without MC and are not fitted")
    T, V, n, bin_channel = T[:, keep], V[:, keep], n[keep], bin_channel[keep]

    # Parameters with no template anywhere are unconstrained
    contrib = np.zeros(len(params))
    for k, p in enumerate(comp_param):
        if p >= 0:
            contrib[p] += T[k].sum()
    for p, c in zip(params, contrib):
        if c <= 0:
            raise FitError(f"parameter '{p}' has no MC in the fitted bins")

    for region, cnt in region_count.items():
        if cnt > 1:
            print(f"Warning: region {region} is fitted in {cnt} variables: the same events are counted "
                  f"{cnt} times (errors too small). Prefer one variable per region.")

    model = Model(params, comp_param, comp_fixed, T, V, n, bin_channel, args.prior, not args.no_mc_stat)
    return model, chan_labels, split, fixed, per_era


def numeric_hessian(f, x, rel=1e-4):
    k = len(x)
    h = rel * np.maximum(np.abs(x), 1e-2)
    H = np.zeros((k, k))
    f0 = f(x)
    for i in range(k):
        for j in range(i, k):
            if i == j:
                xp, xm = x.copy(), x.copy()
                xp[i] += h[i]
                xm[i] -= h[i]
                H[i, i] = (f(xp) - 2 * f0 + f(xm)) / h[i] ** 2
            else:
                xpp, xpm, xmp, xmm = (x.copy() for _ in range(4))
                xpp[i] += h[i]; xpp[j] += h[j]
                xpm[i] += h[i]; xpm[j] -= h[j]
                xmp[i] -= h[i]; xmp[j] += h[j]
                xmm[i] -= h[i]; xmm[j] -= h[j]
                H[i, j] = H[j, i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h[i] * h[j])
    return H


def fit(model):
    """
    Minimize -2 ln L. Returns (theta, cov, singular, converged, at_bound).

    at_bound[p] is 'lower' / 'upper' for a parameter the data push to the edge of
    [SF_MIN, SF_MAX], else None. The minimum is not a stationary point in those
    directions, so they get no error: cov is computed from the curvature in the
    free parameters with the bounded ones held at their bound (NaN rows/columns).
    """
    x0 = np.ones(len(model.params))
    bounds = [(SF_MIN, SF_MAX)] * len(x0)
    res = optimize.minimize(model.deviance, x0, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 10000, 'ftol': 1e-12, 'gtol': 1e-8})
    # polish (L-BFGS-B stops early on flat directions)
    res2 = optimize.minimize(model.deviance, res.x, method='Nelder-Mead',
                             options={'maxiter': 20000, 'xatol': 1e-7, 'fatol': 1e-9})
    best = res2.x if res2.fun < res.fun and np.all(res2.x > 0) else res.x
    best = np.clip(best, SF_MIN, SF_MAX)

    at_bound = [('lower' if v <= SF_MIN * 1.01 else 'upper' if v >= SF_MAX * 0.99 else None) for v in best]
    free = np.array([b is None for b in at_bound])
    k = len(best)
    cov = np.full((k, k), np.nan)
    singular = False
    if free.any():
        def dev_free(xf):
            x = best.copy()
            x[free] = xf
            return model.deviance(x)
        # -2 ln L -> covariance = 2 H^-1
        H = numeric_hessian(dev_free, best[free])
        try:
            cov_free = 2.0 * np.linalg.inv(H)
            singular = bool(np.any(np.diag(cov_free) <= 0) or not np.all(np.isfinite(cov_free)))
        except np.linalg.LinAlgError:
            cov_free, singular = 2.0 * np.linalg.pinv(H), True
        cov[np.ix_(free, free)] = cov_free
    return best, cov, singular, res.success, at_bound


def run_fit(yields_path, args):
    """
    Fit, print the results and write args.output (if set). Returns (params, values, errors).

    With args.per_region every region is fitted on its own (its own channels and SFs) and
    the SFs go to the regions: block of the output; params are then named "<region>/<param>".
    """
    meta, hists = read_yields(yields_path)
    channels = parse_channels(args.channels, meta, hists) if args.channels else default_channels(meta)
    if not channels:
        raise FitError("no unblinded channel to fit")

    if getattr(args, 'per_region', False):
        groups = OrderedDict()
        for c in channels:
            groups.setdefault(c[0], []).append(c)
    else:
        groups = OrderedDict([(None, channels)])

    results = OrderedDict()
    for region, chans in groups.items():
        if region is not None:
            print(f"\n{'=' * 24} {region} {'=' * 24}")
        results[region] = _fit_channels(meta, hists, chans, args)

    if args.output:
        _write_sfs(args, yields_path, meta, results)

    params, values, errors = [], [], []
    for region, r in results.items():
        params += [p if region is None else f"{region}/{p}" for p in r['model'].params]
        values += list(r['theta'])
        errors += list(r['err'])
    return params, np.array(values), np.array(errors)


def _fit_channels(meta, hists, channels, args):
    """Fit these channels together and print the results."""
    model, chan_labels, split, fixed, per_era = build_model(meta, hists, channels, args)
    theta, cov, singular, ok, at_bound = fit(model)
    err = np.sqrt(np.clip(np.diag(cov), 0, None))  # NaN for parameters at a bound
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = cov / np.outer(err, err)

    print(f"\nChannels ({'split by era' if split else 'eras summed'}): {', '.join(chan_labels)}")
    rebin = getattr(args, 'rebin', None)
    print(f"Bins fitted: {len(model.n)}   floated parameters: {len(theta)}"
          f"{'   rebin: ' + str(rebin) if rebin else ''}"
          f"{'   prior width: ' + str(args.prior) if args.prior else ''}"
          f"{'   MC stat: off' if args.no_mc_stat else ''}")
    if not ok:
        print("Warning: the minimizer did not report convergence")
    bounded = [p for p, b in zip(model.params, at_bound) if b]
    if bounded:
        print(f"Warning: {', '.join(bounded)} hit the SF limit [{SF_MIN:g}, {SF_MAX:g}]: the data want "
              f"{'it' if len(bounded) == 1 else 'them'} beyond it, usually because the floated processes "
              f"compensate a shape or normalization mismatch elsewhere. No error is quoted for "
              f"{'it' if len(bounded) == 1 else 'them'}; the other errors hold "
              f"{'it' if len(bounded) == 1 else 'them'} fixed at the limit. Consider --fix or --prior.")
    if singular:
        print("Warning: the Hessian of the free parameters is singular: some are not constrained by the data "
              "(degenerate shapes). Fix some of them (--fix) or add channels that separate them.")

    w = max(len(p) for p in model.params)
    print(f"\n{'parameter':<{w}}  {'SF':>8}  {'error':>8}")
    for p, v, e, b in zip(model.params, theta, err, at_bound):
        if b:
            print(f"{p:<{w}}  {v:8.4f}  {'--':>8}   <- at {b} bound, error undefined")
        else:
            print(f"{p:<{w}}  {v:8.4f}  {e:8.4f}")
    for p, v in fixed.items():
        print(f"{p:<{w}}  {v:8.4f}  {'fixed':>8}")

    if len(theta) > 1:
        print("\nCorrelation matrix" + (" (-- = parameter at a bound)" if bounded else ""))
        short = [p[:12] for p in model.params]
        print(" " * (w + 2) + " ".join(f"{s:>12}" for s in short))
        for p, row in zip(model.params, corr):
            print(f"{p:<{w}}  " + " ".join(f"{c:12.3f}" if np.isfinite(c) else f"{'--':>12}" for c in row))

    pre = model.bin_deviance(np.ones(len(theta)))
    post = model.bin_deviance(theta)
    print(f"\n{'channel':<36} {'bins':>5} {'-2lnL pre':>10} {'-2lnL post':>11}")
    for label in chan_labels:
        m = model.bin_channel == label
        print(f"{label:<36} {m.sum():>5} {pre[m].sum():>10.1f} {post[m].sum():>11.1f}")
    ndf = len(model.n) - len(theta)
    total = model.deviance(theta)
    print(f"{'total':<36} {len(model.n):>5} {pre.sum():>10.1f} {total:>11.1f}   "
          f"ndf {ndf}   p-value {stats.chi2.sf(total, ndf) if ndf > 0 else float('nan'):.3g}")

    return dict(model=model, theta=theta, err=err, corr=corr, at_bound=at_bound, fixed=fixed,
                chan_labels=chan_labels, total=total, ndf=ndf)


def _sf_table(meta, r):
    """{proc: sf | {era: sf}} of one fit result, fixed processes included."""
    sfs = OrderedDict()
    for p, v in zip(r['model'].params, r['theta']):
        proc, _, era = p.partition(':') if p not in meta['processes'] else (p, '', '')
        if era:
            sfs.setdefault(proc, OrderedDict())[era] = round(float(v), 5)
        else:
            sfs[proc] = round(float(v), 5)
    for p, v in r['fixed'].items():
        sfs[p] = float(v)
    return sfs


def _fit_info(r):
    model, theta, err, corr, at_bound = r['model'], r['theta'], r['err'], r['corr'], r['at_bound']
    return OrderedDict([
        ('channels', r['chan_labels']),
        ('parameters', {p: ({'value': round(float(v), 5), 'error': None, 'at_bound': b} if b else
                            {'value': round(float(v), 5), 'error': round(float(e), 5)})
                        for p, v, e, b in zip(model.params, theta, err, at_bound)}),
        ('correlation', {p: [round(float(c), 4) if np.isfinite(c) else None for c in row]
                         for p, row in zip(model.params, corr)}),
        ('deviance', round(float(r['total']), 3)),
        ('ndf', int(r['ndf'])),
    ])


def _write_sfs(args, yields_path, meta, results):
    common = OrderedDict([('yields', str(yields_path)), ('rebin', getattr(args, 'rebin', None)),
                          ('prior', args.prior), ('mc_stat', not args.no_mc_stat)])
    if None in results:
        r = results[None]
        info = _fit_info(r)
        doc = OrderedDict([('scale_factors', _sf_table(meta, r)),
                           ('fit', OrderedDict([('yields', common['yields'])] + list(info.items())
                                               + [(k, v) for k, v in common.items() if k != 'yields']))])
    else:
        # Regions not fitted keep SF 1 (empty global table)
        doc = OrderedDict([('scale_factors', OrderedDict()),
                           ('regions', OrderedDict((reg, _sf_table(meta, r)) for reg, r in results.items())),
                           ('fit', OrderedDict(list(common.items()) + [('per_region', True)] +
                                               [(reg, _fit_info(r)) for reg, r in results.items()]))])
    yaml.add_representer(OrderedDict, lambda d, data: d.represent_mapping('tag:yaml.org,2002:map', data.items()))
    with open(args.output, 'w') as fh:
        yaml.dump(doc, fh, sort_keys=False, default_flow_style=None, width=120)
    print(f"\nScale factors written to {args.output}  (use: replot_yields.py {yields_path} --sf {args.output})")


def run_fit_and_replot(yields_path, args, formats=None):
    """run_fit, then the replot with the fitted SFs when args.replot is set."""
    result = run_fit(yields_path, args)
    if args.replot:
        if not args.output:
            raise FitError("replot needs the SFs written to a file: set output (-o)")
        from replot_yields import replot
        out_dir = args.replot_output or f"{Path(args.output).with_suffix('')}_replot"
        print(f"\nRedrawing the Data/MC plots with the fitted SFs -> {out_dir}/")
        replot(yields_path, args.output, out_dir, formats=args.replot_format or formats or ['root'])
    return result


def main():
    args = parse_arguments()
    overrides = {k: getattr(args, k) for k in FIT_DEFAULTS if hasattr(args, k)}
    formats = None
    try:
        if args.config:
            with open(args.config) as fh:
                cfg = yaml.safe_load(fh) or {}
            if not cfg.get('output'):
                raise FitError(f"{args.config} has no output: setting to find the yields file from")
            yields_path, sf_path, replot_dir = config_paths(cfg['output'], cfg.get('yields'))
            opts = fit_options(cfg.get('sf_fit'), **overrides)
            opts.output = opts.output or sf_path
            opts.replot_output = opts.replot_output or replot_dir
            fmt = cfg.get('format', 'root')
            formats = [fmt] if isinstance(fmt, str) else list(fmt)
            yields_path = args.yields or yields_path
        else:
            if not args.yields:
                raise FitError("give a yields file or --config")
            yields_path, opts = args.yields, fit_options(None, **overrides)
        if not Path(yields_path).exists():
            raise FitError(f"yields file {yields_path} not found (run main.py with yields: true first)")
        run_fit_and_replot(yields_path, opts, formats)
    except FitError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
