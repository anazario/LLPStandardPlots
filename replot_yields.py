#!/usr/bin/env python3
"""
Redraw the data/MC comparison plots from a yields file with MC scale factors.

The yields file comes from main.py with --yields (or `yields: true` in the YAML
config). Nothing is re-read from the ntuples, so a full set of plots takes seconds.

Examples
--------
  # Plots as the full run made them (no SFs)
  python replot_yields.py datamc_v34_run2run3_splitlumi_yields.root -o replot_nominal

  # With scale factors (hand-written or written by fit_sfs.py)
  python replot_yields.py datamc_v34_run2run3_splitlumi_yields.root --sf sfs.yaml -o replot_sf

  # Run 3 only (Run 3 MC against Run 3 data), only RISR and pT(ISR)
  python replot_yields.py yields.root --sf sfs.yaml --era run3 --vars isr_risr isr_ptisr

SF file (process names match the stack labels, ignoring case and spaces):
  scale_factors:
    QCD multijets: 1.10
    W+jets: {run2: 1.05, run3: 0.95}
  regions:                      # optional per-region overrides
    CustomRegion2:
      QCD multijets: 0.8
"""

import argparse
import sys
from pathlib import Path

import ROOT

from src.yields import (read_yields, load_scale_factors, mc_by_process, data_total,
                        is_qcd, is_gjets)
from src.style import StyleManager
from src.plotter import PlotterDataMC
from src.utils import parse_background_name

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning


def parse_arguments():
    p = argparse.ArgumentParser(description="Redraw data/MC plots from a yields file with MC scale factors")
    p.add_argument('yields', help='Yields ROOT file written by main.py --yields')
    p.add_argument('--sf', default=None, help='YAML file with scale factors (none -> plots as produced)')
    p.add_argument('-o', '--output', default=None,
                   help='Output directory (default: <yields stem>_replot)')
    p.add_argument('--format', nargs='+', default=['root'], choices=['pdf', 'png', 'eps', 'root'])
    p.add_argument('--regions', nargs='+', default=None, help='Only these regions')
    p.add_argument('--vars', nargs='+', default=None,
                   help='Only these variables (short names, e.g. isr_risr, or branch names)')
    p.add_argument('--era', nargs='+', default=None,
                   help='Only these eras (MC and data), e.g. run2 or run3')
    p.add_argument('--normalize', action='store_true', help='Shape comparison (area-normalized)')
    p.add_argument('--no-merge-qcd-gjets', action='store_true',
                   help='Keep gamma+jets separate from QCD in SV regions')
    p.add_argument('--table-only', action='store_true', help='Print the yield table, draw nothing')
    return p.parse_args()


def chi2_ndf(data_hist, mc_hists):
    """Pearson chi2 over bins with MC > 0, using data and MC stat errors."""
    chi2, ndf = 0.0, 0
    for i in range(1, data_hist.GetNbinsX() + 1):
        n = data_hist.GetBinContent(i)
        mu = sum(h.GetBinContent(i) for h in mc_hists)
        var = n + sum(h.GetBinError(i) ** 2 for h in mc_hists)
        if mu <= 0 or var <= 0:
            continue
        chi2 += (n - mu) ** 2 / var
        ndf += 1
    return chi2, ndf


def _merge_qcd_gjets(mc):
    qcd = next((p for p in mc if is_qcd(p)), None)
    gj = [p for p in mc if is_gjets(p)]
    if qcd is None or not gj:
        return mc
    out = dict(mc)
    for p in gj:
        out[qcd].Add(out.pop(p))
    return out


def var_matches(var, meta, wanted):
    return not wanted or var in wanted or meta['variables'][var]['branch'] in wanted


def replot(yields_path, sf_path=None, out_dir=None, formats=('root',), regions=None, variables=None,
           eras=None, normalize=False, merge_qcd_gjets=True, table_only=False):
    """Redraw the Data/MC plots of a yields file with the SFs of sf_path (None = no SFs).

    Output: <out_dir>/<region>/datamc_plots/datamc_<var>_<region>.<fmt> (plots.root for 'root').
    Raises ValueError on unknown processes in the SF file or unknown eras.
    """
    meta, hists = read_yields(yields_path)
    sfs = load_scale_factors(sf_path)
    sfs.validate(meta['processes'])

    if eras:
        missing = [e for e in eras if e not in meta['eras']]
        if missing:
            raise ValueError(f"era(s) {missing} not in the yields file (has: {meta['eras']})")

    # Luminosity label for the selected eras
    run_lumis = meta.get('run_lumis') or {}
    if eras and run_lumis:
        lumi = sum(run_lumis.get(e, 0.0) for e in eras)
        energy_label = {'run2': '13 TeV', 'run3': '13.6 TeV'}.get(eras[0]) if len(eras) == 1 else meta.get('energy_label')
    else:
        lumi = meta.get('lumi') or 0.0
        energy_label = meta.get('energy_label')
    style = StyleManager(luminosity=round(lumi), energy=meta.get('energy', 13.6), energy_label=energy_label)
    style.set_style()
    plotter = PlotterDataMC(style)

    out_dir = Path(out_dir or f"{Path(yields_path).stem}_replot")
    f_root = None
    if not table_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        if 'root' in formats:
            f_root = ROOT.TFile(str(out_dir / "plots.root"), "RECREATE")

    print(f"{'region':<16} {'variable':<24} {'data':>10} {'MC nominal':>11} {'MC w/ SF':>11} "
          f"{'data/MC':>8} {'chi2/ndf nom':>13} {'chi2/ndf SF':>12}")
    for reg in meta['regions']:
        region = reg['name']
        if regions and region not in regions:
            continue
        merge = reg.get('region_type') in ('sv', 'inclusive-sv') and merge_qcd_gjets
        for var in reg['vars']:
            if not var_matches(var, meta, variables):
                continue
            entry = hists[(region, var)]
            mc_nom = mc_by_process(entry, region, None, eras)
            mc_sf = mc_by_process(entry, region, sfs, eras)
            if merge:
                mc_nom, mc_sf = _merge_qcd_gjets(mc_nom), _merge_qcd_gjets(mc_sf)
            if not mc_sf:
                continue
            data = None if reg.get('blind') else data_total(entry, eras)

            tot_nom = sum(h.Integral() for h in mc_nom.values())
            tot_sf = sum(h.Integral() for h in mc_sf.values())
            if data:
                n_data = data.Integral()
                c_nom, d_nom = chi2_ndf(data, mc_nom.values())
                c_sf, d_sf = chi2_ndf(data, mc_sf.values())
                print(f"{region:<16} {var:<24} {n_data:>10.1f} {tot_nom:>11.1f} {tot_sf:>11.1f} "
                      f"{n_data / tot_sf if tot_sf else 0:>8.3f} "
                      f"{c_nom:>7.1f}/{d_nom:<5d} {c_sf:>7.1f}/{d_sf:<5d}")
            else:
                print(f"{region:<16} {var:<24} {'blinded':>10} {tot_nom:>11.1f} {tot_sf:>11.1f}")

            if table_only:
                continue

            mc_histograms = []
            for proc, h in mc_sf.items():
                if h.Integral() <= 0:
                    continue
                h.SetLineColor(ROOT.kBlack)
                h.SetLineWidth(1)
                h.SetFillColor(plotter._get_background_color_index(proc))
                h.SetStats(0)
                mc_histograms.append((h, plotter._clean_mc_label(parse_background_name(proc))))
            if data:
                data.SetLineColor(ROOT.kBlack)
                data.SetMarkerStyle(20)
                data.SetMarkerSize(style.data_marker_size)
                data.SetLineWidth(style.data_line_width)
                data.SetStats(0)
                if data.Integral() <= 0:
                    data = None

            vconf = meta['variables'][var]
            xmin, xmax = vconf['range']
            canvas = plotter.draw_data_mc_comparison(mc_histograms, data, var, vconf['label'], xmin, xmax,
                                                     final_state_label=reg.get('label'), suffix=region,
                                                     normalized=normalize)
            sub = out_dir / region / "datamc_plots"
            for fmt in formats:
                if fmt == 'root':
                    d = f_root.GetDirectory(region) or f_root.mkdir(region)
                    d.cd()
                    canvas.Write()
                    f_root.cd()
                else:
                    sub.mkdir(parents=True, exist_ok=True)
                    canvas.SaveAs(str(sub / f"{canvas.GetName()}.{fmt}"))
            canvas.Close()

    if f_root:
        f_root.Close()
    if not table_only:
        print(f"\nPlots saved to {out_dir}/")


def main():
    args = parse_arguments()
    try:
        replot(args.yields, args.sf, args.output, formats=args.format, regions=args.regions,
               variables=args.vars, eras=args.era, normalize=args.normalize,
               merge_qcd_gjets=not args.no_merge_qcd_gjets, table_only=args.table_only)
    except ValueError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
