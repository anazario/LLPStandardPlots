#!/usr/bin/env python3
"""
Repair ROOT files whose 2D canvases crash (or show a black canvas) in ROOT 6.24.

Files written by ROOT >= 6.26 store the 2D colour palette (TPaletteAxis)
without a name; ROOT 6.24 only recognises it by the name 'palette' and
segfaults painting it. This rewrites every canvas that has an unnamed palette,
in place (or into a copy with --output).

    python tools/fix_palette_names.py 1d_risr_svonly_compressed_v54.root
    python tools/fix_palette_names.py in.root --output fixed.root
"""
import argparse
import os
import shutil
import sys

import ROOT

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import name_palettes  # noqa: E402


def fix_directory(directory):
    fixed = 0
    for key in list(directory.GetListOfKeys()):
        cls = key.GetClassName()
        if cls in ("TDirectoryFile", "TDirectory"):
            fixed += fix_directory(key.ReadObj())
        elif cls == "TCanvas":
            canvas = key.ReadObj()
            if name_palettes(canvas):
                directory.cd()
                canvas.Write(key.GetName(), ROOT.TObject.kOverwrite)
                fixed += 1
    return fixed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="ROOT file to repair")
    p.add_argument("--output", help="write a repaired copy here instead of changing the input")
    args = p.parse_args()

    ROOT.gROOT.SetBatch(True)
    path = args.input
    if args.output:
        shutil.copyfile(args.input, args.output)
        path = args.output

    f = ROOT.TFile.Open(path, "UPDATE")
    if not f or f.IsZombie():
        sys.exit(f"Cannot open {path}")
    n = fix_directory(f)
    f.Close()
    print(f"{path}: fixed {n} canvas(es)")


if __name__ == "__main__":
    main()
