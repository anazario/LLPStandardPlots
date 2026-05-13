#!/usr/bin/env python3
"""
fit_plots.py – produce pre/postfit comparison plots from a Combine
FitDiagnostics output for both compressed and uncompressed analyses.

Auto-detects fit type from the config (override with --mode if needed).

Usage examples:
  # Compressed (auto-detected, 12-bin shape_transfer):
  python fit_plots.py \\
      --fit-result root/fits/compressed/fitDiagnostics.PhoSVHadronicCompressedValidation.root \\
      --fit-config root/fits/compressed/SV_ValidationRegions_Compressed_FitConfig.yaml \\
      --output plots/compressed --lumi 12 --format pdf

  # Uncompressed (auto-detected, shape_transfer + ABCD):
  python fit_plots.py \\
      --fit-result root/fits/uncompressed/fitDiagnostics_PhoDelayedPromptSVHad_ValFits.root \\
      --fit-config root/fits/uncompressed/PhoSV_ValidationRegions_NonCompressed_FitConfig.yaml \\
      --output plots/uncompressed --lumi 12 --format pdf
"""

import argparse
import os
from src.fit_plotter import FitPlotter


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot pre/postfit yields from FitDiagnostics (compressed + uncompressed)"
    )
    p.add_argument("--fit-result", required=True,
                   help="Path to fitDiagnostics ROOT file")
    p.add_argument("--fit-config", required=True,
                   help="Path to fit config YAML")
    p.add_argument("--output", default="fit",
                   help="Output path prefix, e.g. 'plots/myfit' (default: fit)")
    p.add_argument("--format", choices=["pdf", "png", "eps", "root"], default="pdf",
                   help="Output format (default: pdf)")
    p.add_argument("--lumi",   type=float, default=136.0,
                   help="Integrated luminosity in fb^-1 (default: 136)")
    p.add_argument("--energy", type=float, default=13.0,
                   help="Centre-of-mass energy in TeV (default: 13)")
    p.add_argument("--mode", choices=["compressed", "uncompressed"], default=None,
                   help="Override auto-detected fit mode")
    p.add_argument("--label-scheme", choices=["auto", "legacy", "compressed-final", "noncompressed-final"],
                   default="auto",
                   help="Axis/category label scheme (default: auto, current behavior)")
    p.add_argument("--sr-color", action="store_true", default=False,
                   help="Highlight predicted SR bin in orange on ABCD plots")
    p.add_argument("--data-mc", action="store_true", default=False,
                   help="Draw stacked per-process Data/MC plots instead of total-background plots")
    return p.parse_args()


def main():
    args = parse_args()

    # Create output directory if prefix includes one
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plotter = FitPlotter(luminosity=args.lumi, energy=args.energy)
    plotter.plot_all(
        fit_result_path=args.fit_result,
        fit_config_path=args.fit_config,
        output_prefix=args.output,
        output_format=args.format,
        mode_override=args.mode,
        label_scheme=args.label_scheme,
        show_sr=args.sr_color,
        data_mc=args.data_mc,
    )


if __name__ == "__main__":
    main()
