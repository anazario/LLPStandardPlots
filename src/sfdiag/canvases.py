"""
ROOT output of the diagnostics: every plot is a TCanvas written into one file.

Data/MC stacks are drawn with PlotterDataMC.draw_data_mc_comparison (as replot_yields.py
does); the remaining canvas types (overlays, matrices, bar charts, graphs) are small
generic helpers. Every canvas gets a deterministic unique name; all drawn objects are
kept alive on the canvas until it is written.
"""

import re
from collections import OrderedDict

import numpy as np
import ROOT

from src.utils import parse_background_name

ROOT.gROOT.SetBatch(True)

LINE_COLORS = [ROOT.kBlue + 1, ROOT.kRed + 1, ROOT.kGreen + 2, ROOT.kMagenta + 1, ROOT.kOrange + 7,
               ROOT.kCyan + 2, ROOT.kGray + 2, ROOT.kViolet + 6, ROOT.kAzure + 7, ROOT.kSpring - 6]
_UID = [0]


def uid(prefix):
    _UID[0] += 1
    return f"{prefix}__{_UID[0]}"


def safe(s):
    return re.sub(r'[^A-Za-z0-9]+', '_', str(s)).strip('_')


class RootOut:
    """One output file; write(dir, canvas, name) creates nested directories on demand."""

    def __init__(self, path):
        self.path = str(path)
        self.f = ROOT.TFile(self.path, "RECREATE")
        self.names = set()
        self.n = 0
        from main import _write_palette_setup
        _write_palette_setup(self.f)

    def dir(self, path):
        d = self.f
        for part in [p for p in path.split('/') if p]:
            sub = d.GetDirectory(part)
            d = sub if sub else d.mkdir(part)
        return d

    def write(self, path, canvas, name):
        if canvas is None:
            return
        name = safe(name)
        full = f"{path}/{name}"
        if full in self.names:
            k = 2
            while f"{full}_{k}" in self.names:
                k += 1
            name, full = f"{name}_{k}", f"{full}_{k}"
        self.names.add(full)
        d = self.dir(path)
        d.cd()
        canvas.SetName(name)
        canvas.Write(name)
        self.f.cd()
        canvas.Close()
        self.n += 1

    def write_object(self, path, obj, name):
        d = self.dir(path)
        d.cd()
        obj.Write(safe(name))
        self.f.cd()

    def close(self):
        self.f.Close()


def _th1(edges, values, errors=None, name="h", title=""):
    edges = np.asarray(edges, dtype=float)
    h = ROOT.TH1D(uid(name), title, len(edges) - 1, edges)
    h.SetDirectory(0)
    for i, v in enumerate(values):
        h.SetBinContent(i + 1, float(v) if np.isfinite(v) else 0.0)
        if errors is not None:
            e = errors[i]
            h.SetBinError(i + 1, float(e) if np.isfinite(e) else 0.0)
    return h


def _label_th1(labels, values, errors=None, name="hl", title=""):
    h = ROOT.TH1D(uid(name), title, len(labels), 0, len(labels))
    h.SetDirectory(0)
    for i, (l, v) in enumerate(zip(labels, values)):
        h.GetXaxis().SetBinLabel(i + 1, str(l))
        h.SetBinContent(i + 1, float(v) if v is not None and np.isfinite(v) else 0.0)
        if errors is not None and errors[i] is not None and np.isfinite(errors[i]):
            h.SetBinError(i + 1, float(errors[i]))
    return h


def _text(canvas, lines, x=0.14, y=0.92, size=0.03):
    t = ROOT.TLatex()
    t.SetNDC()
    t.SetTextSize(size)
    t.SetTextFont(42)
    for i, l in enumerate(lines):
        t.DrawLatex(x, y - i * size * 1.3, l)
    canvas._keep.append(t)


def new_canvas(title, w=900, h=650):
    c = ROOT.TCanvas(uid("c"), title, w, h)
    c._keep = []
    return c


def proc_color(plotter, proc):
    return plotter._get_background_color_index(proc)


def proc_label(plotter, proc):
    return plotter._clean_mc_label(parse_background_name(proc))


# --------------------------------------------------------------------------- #
# Data / MC stack (production drawing)                                        #
# --------------------------------------------------------------------------- #
def stack_canvas(plotter, entry, var_meta, eras, sfs=None, region_label=None, tag="", normalized=False,
                 rebin_edges=None):
    """Data/MC stack + ratio for the given eras (summed), MC scaled by sfs {(proc, era): sf}."""
    from src.sfdiag.inputs import th1_from
    mc = OrderedDict()
    for (p, e), h in entry['bkg'].items():
        if e not in eras:
            continue
        sf = sfs.get((p, e), 1.0) if sfs else 1.0
        hd = {'edges': h['edges'], 'w': h['w'] * sf, 'w2': h['w2'] * sf ** 2}
        mc[p] = hd if p not in mc else {'edges': h['edges'], 'w': mc[p]['w'] + hd['w'], 'w2': mc[p]['w2'] + hd['w2']}
    data = [h for e, h in entry['data'].items() if e in eras]
    plotter._ensure_mc_colors()
    hists = []
    for p, hd in mc.items():
        h = th1_from(hd, f"stk_{safe(p)}", p)
        if rebin_edges is not None:
            h = h.Rebin(len(rebin_edges) - 1, uid("rb"), np.asarray(rebin_edges, dtype=float))
            h.SetDirectory(0)
        if h.Integral() <= 0:
            continue
        h.SetLineColor(ROOT.kBlack)
        h.SetLineWidth(1)
        h.SetFillColor(proc_color(plotter, p))
        h.SetStats(0)
        hists.append((h, proc_label(plotter, p)))
    hdata = None
    if data:
        dd = {'edges': data[0]['edges'], 'w': sum(h['w'] for h in data), 'w2': sum(h['w2'] for h in data)}
        hdata = th1_from(dd, "stk_data", "data")
        if rebin_edges is not None:
            hdata = hdata.Rebin(len(rebin_edges) - 1, uid("rbd"), np.asarray(rebin_edges, dtype=float))
            hdata.SetDirectory(0)
        hdata.SetLineColor(ROOT.kBlack)
        hdata.SetMarkerStyle(20)
        hdata.SetMarkerSize(plotter.style.data_marker_size)
        hdata.SetLineWidth(plotter.style.data_line_width)
        hdata.SetStats(0)
        if hdata.Integral() <= 0:
            hdata = None
    if not hists:
        return None
    xmin, xmax = var_meta['range']
    c = plotter.draw_data_mc_comparison(hists, hdata, uid("v"), var_meta['label'], xmin, xmax,
                                        final_state_label=region_label, suffix=tag, normalized=normalized)
    c._keep = [hists, hdata]
    return c


def styled_plotter(lumi, energy_label, energy=13.6):
    from src.style import StyleManager
    from src.plotter import PlotterDataMC
    st = StyleManager(luminosity=round(lumi or 0), energy=energy, energy_label=energy_label)
    return PlotterDataMC(st)


# --------------------------------------------------------------------------- #
# Generic canvases                                                            #
# --------------------------------------------------------------------------- #
def overlay_canvas(title, xlabel, ylabel, edges, series, logy=False, notes=(), ymin=None, ymax=None, draw="HIST E"):
    """series: list of (label, values, errors|None, color, line_style)."""
    c = new_canvas(title)
    c.SetLeftMargin(0.13)
    c.SetRightMargin(0.25)
    c.SetLogy(bool(logy))
    leg = ROOT.TLegend(0.76, 0.35, 0.995, 0.9)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.028)
    hs = []
    vmax, vmin = -np.inf, np.inf
    for lab, vals, errs, col, ls in series:
        h = _th1(edges, vals, errs, "ov")
        h.SetLineColor(col)
        h.SetMarkerColor(col)
        h.SetLineStyle(ls)
        h.SetLineWidth(2)
        h.SetStats(0)
        hs.append(h)
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if len(v):
            vmax = max(vmax, v.max())
            pos = v[v > 0]
            if len(pos):
                vmin = min(vmin, pos.min())
        leg.AddEntry(h, lab, "l")
    if not hs:
        return None
    first = hs[0]
    first.SetTitle(f"{title};{xlabel};{ylabel}")
    if ymax is None:
        ymax = vmax * (20 if logy else 1.3) if np.isfinite(vmax) and vmax > 0 else 1.0
    if ymin is None:
        ymin = (vmin * 0.3 if np.isfinite(vmin) else 1e-3) if logy else min(0.0, np.nanmin([np.nanmin(np.asarray(s[1], float)) for s in series]))
    first.SetMaximum(ymax)
    first.SetMinimum(ymin)
    first.Draw(draw)
    for h in hs[1:]:
        h.Draw(f"{draw} SAME")
    leg.Draw()
    _text(c, notes, y=0.86)
    c._keep += [hs, leg]
    return c


def matrix_canvas(title, xlabels, ylabels, M, zmin=None, zmax=None, fmt="4.2f", ztitle="", notes=()):
    M = np.asarray(M, dtype=float)
    c = new_canvas(title, 900, 750)
    c.SetLeftMargin(0.24)
    c.SetBottomMargin(0.2)
    c.SetRightMargin(0.15)
    h = ROOT.TH2D(uid("m"), f"{title};;;{ztitle}", len(xlabels), 0, len(xlabels), len(ylabels), 0, len(ylabels))
    h.SetDirectory(0)
    for i, l in enumerate(xlabels):
        h.GetXaxis().SetBinLabel(i + 1, str(l))
    for j, l in enumerate(ylabels):
        h.GetYaxis().SetBinLabel(j + 1, str(l))
    for i in range(len(xlabels)):
        for j in range(len(ylabels)):
            v = M[j, i] if M.shape == (len(ylabels), len(xlabels)) else np.nan
            h.SetBinContent(i + 1, j + 1, v if np.isfinite(v) else -999)
    if zmin is not None:
        h.SetMinimum(zmin)
    if zmax is not None:
        h.SetMaximum(zmax)
    h.SetStats(0)
    h.GetXaxis().LabelsOption("v")
    h.SetMarkerSize(1.2)
    ROOT.gStyle.SetPaintTextFormat(fmt)
    ROOT.gStyle.SetPalette(ROOT.kViridis)
    h.Draw("COLZ TEXT")
    _text(c, notes, x=0.02, y=0.97, size=0.025)
    c._keep.append(h)
    return c


def bar_canvas(title, labels, series, ylabel="", logy=False, notes=(), ymin=None, ymax=None):
    """series: list of (label, values, errors|None, color). Values per category label."""
    c = new_canvas(title, 1000, 650)
    c.SetBottomMargin(0.28)
    c.SetRightMargin(0.22)
    c.SetLeftMargin(0.12)
    c.SetLogy(bool(logy))
    leg = ROOT.TLegend(0.79, 0.5, 0.995, 0.9)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.028)
    hs = []
    allv = []
    for k, (lab, vals, errs, col) in enumerate(series):
        h = _label_th1(labels, vals, errs, "bar")
        h.SetLineColor(col)
        h.SetMarkerColor(col)
        h.SetMarkerStyle(20 + k)
        h.SetLineWidth(2)
        h.SetStats(0)
        hs.append(h)
        allv += [v for v in vals if v is not None and np.isfinite(v)]
        leg.AddEntry(h, lab, "lp")
    if not hs:
        return None
    hs[0].SetTitle(f"{title};;{ylabel}")
    hs[0].GetXaxis().LabelsOption("v")
    hs[0].GetXaxis().SetLabelSize(0.03)
    pos = [v for v in allv if v > 0]
    if ymax is None:
        ymax = (max(allv) * (20 if logy else 1.3)) if allv and max(allv) > 0 else 1.0
    if ymin is None:
        ymin = (min(pos) * 0.3 if pos else 1e-3) if logy else min(0.0, min(allv) * 1.3 if allv else 0.0)
    hs[0].SetMaximum(ymax)
    hs[0].SetMinimum(ymin)
    hs[0].Draw("E1 P")
    for h in hs[1:]:
        h.Draw("E1 P SAME")
    leg.Draw()
    _text(c, notes, y=0.86)
    c._keep += [hs, leg]
    return c


def graph_canvas(title, xlabels, series, ylabel="", ref_lines=(), notes=(), ymin=None, ymax=None, xtitle=""):
    """series: list of (label, y, ey|None, color); x = category index (+ small offset per series)."""
    c = new_canvas(title, 1000, 650)
    c.SetBottomMargin(0.25)
    c.SetRightMargin(0.22)
    leg = ROOT.TLegend(0.79, 0.5, 0.995, 0.9)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.028)
    n = len(xlabels)
    frame = ROOT.TH1D(uid("fr"), f"{title};{xtitle};{ylabel}", max(n, 1), 0, max(n, 1))
    frame.SetDirectory(0)
    frame.SetStats(0)
    for i, l in enumerate(xlabels):
        frame.GetXaxis().SetBinLabel(i + 1, str(l))
    frame.GetXaxis().LabelsOption("v")
    frame.GetXaxis().SetLabelSize(0.03 if n < 30 else 0.02)
    ys = [v for _, y, ey, _ in series for v, e in zip(y, ey if ey is not None else [0] * len(y))
          if v is not None and np.isfinite(v)]
    es = [abs(e) for _, y, ey, _ in series if ey is not None for e in ey if e is not None and np.isfinite(e)]
    lo = (min(ys) - (max(es) if es else 0)) if ys else -1
    hi = (max(ys) + (max(es) if es else 0)) if ys else 1
    pad = 0.15 * (hi - lo if hi > lo else 1)
    frame.SetMinimum(ymin if ymin is not None else lo - pad)
    frame.SetMaximum(ymax if ymax is not None else hi + pad)
    frame.Draw("AXIS")
    keep = [frame]
    off = np.linspace(-0.25, 0.25, len(series)) if len(series) > 1 else [0.0]
    for k, (lab, y, ey, col) in enumerate(series):
        g = ROOT.TGraphErrors()
        g.SetName(uid("g"))
        j = 0
        for i, v in enumerate(y):
            if v is None or not np.isfinite(v):
                continue
            g.SetPoint(j, i + 0.5 + off[k], float(v))
            e = ey[i] if ey is not None else 0.0
            g.SetPointError(j, 0.0, float(e) if e is not None and np.isfinite(e) else 0.0)
            j += 1
        g.SetMarkerStyle(20 + k % 14)
        g.SetMarkerColor(col)
        g.SetLineColor(col)
        g.SetLineWidth(2)
        if j:
            g.Draw("P SAME")
            leg.AddEntry(g, lab, "lp")
        keep.append(g)
    for yv, style in ref_lines:
        ln = ROOT.TLine(0, yv, max(n, 1), yv)
        ln.SetLineStyle(style)
        ln.SetLineColor(ROOT.kGray + 2)
        ln.Draw()
        keep.append(ln)
    leg.Draw()
    _text(c, notes, y=0.86)
    c._keep += keep + [leg]
    return c


def ratio_canvas(title, labels, top_series, ratio, ratio_err=None, ylabel="", rlabel="data / MC", logy=True, notes=()):
    """Category plot with a ratio pad (cutflows): top_series as in bar_canvas, ratio below."""
    c = new_canvas(title, 1000, 800)
    p1 = ROOT.TPad(uid("p1"), "", 0, 0.38, 1, 1)
    p2 = ROOT.TPad(uid("p2"), "", 0, 0, 1, 0.38)
    p1.SetBottomMargin(0.02)
    p2.SetTopMargin(0.02)
    p2.SetBottomMargin(0.5)
    for p in (p1, p2):
        p.SetRightMargin(0.22)
        p.SetLeftMargin(0.12)
        p.Draw()
    p1.cd()
    p1.SetLogy(bool(logy))
    leg = ROOT.TLegend(0.79, 0.3, 0.995, 0.92)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)
    hs = []
    allv = []
    for k, (lab, vals, errs, col) in enumerate(top_series):
        h = _label_th1(labels, vals, errs, "cf")
        h.SetLineColor(col)
        h.SetMarkerColor(col)
        h.SetMarkerStyle(20 + k)
        h.SetLineWidth(2)
        h.SetStats(0)
        h.GetXaxis().SetLabelSize(0)
        hs.append(h)
        allv += [v for v in vals if v is not None and np.isfinite(v) and v > 0]
        leg.AddEntry(h, lab, "lp")
    if not hs:
        return None
    hs[0].SetTitle(f"{title};;{ylabel}")
    hs[0].SetMaximum(max(allv) * 30 if allv else 10)
    hs[0].SetMinimum(min(allv) * 0.3 if allv else 0.1)
    hs[0].Draw("HIST P")
    for h in hs[1:]:
        h.Draw("HIST P SAME")
    leg.Draw()
    _text(c, notes, y=0.9)
    p2.cd()
    hr = _label_th1(labels, ratio, ratio_err, "cfr")
    hr.SetStats(0)
    hr.SetTitle(f";;{rlabel}")
    hr.SetMarkerStyle(20)
    hr.GetXaxis().LabelsOption("v")
    hr.GetXaxis().SetLabelSize(0.07)
    hr.GetYaxis().SetLabelSize(0.07)
    hr.GetYaxis().SetTitleSize(0.08)
    hr.GetYaxis().SetTitleOffset(0.6)
    rv = [r for r in ratio if r is not None and np.isfinite(r)]
    hr.SetMinimum(max(0.0, min(rv) * 0.8) if rv else 0)
    hr.SetMaximum(max(rv) * 1.2 if rv else 2)
    hr.Draw("E1 P")
    ln = ROOT.TLine(0, 1, len(labels), 1)
    ln.SetLineStyle(2)
    ln.Draw()
    c._keep += [p1, p2, hs, leg, hr, ln]
    return c


def hist1d_canvas(title, xlabel, values, nbins, xmin, xmax, lines=(), notes=(), fit_gauss=False):
    """Distribution of toy quantities (pulls, deviances) with optional vertical reference lines."""
    c = new_canvas(title)
    h = ROOT.TH1D(uid("t"), f"{title};{xlabel};toys", nbins, xmin, xmax)
    h.SetDirectory(0)
    for v in values:
        if v is not None and np.isfinite(v):
            h.Fill(float(v))
    h.SetLineWidth(2)
    h.SetLineColor(ROOT.kBlue + 1)
    h.Draw("HIST")
    keep = [h]
    if fit_gauss and h.GetEntries() > 5:
        f = ROOT.TF1(uid("gaus"), "gaus", xmin, xmax)
        h.Fit(f, "QN0")
        f.SetLineColor(ROOT.kRed + 1)
        f.Draw("SAME")
        keep.append(f)
    ymax = h.GetMaximum() * 1.1 if h.GetMaximum() > 0 else 1
    for x, col in lines:
        ln = ROOT.TLine(x, 0, x, ymax)
        ln.SetLineColor(col)
        ln.SetLineWidth(2)
        ln.SetLineStyle(2)
        ln.Draw()
        keep.append(ln)
    _text(c, notes, x=0.55, y=0.85)
    c._keep += keep
    return c


def pulls_grid_canvas(title, pulls):
    """One pad per parameter with its toy pull distribution and a Gaussian fit."""
    names = list(pulls)
    n = len(names)
    if not n:
        return None
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    c = new_canvas(title, 300 * cols, 260 * rows)
    c.Divide(cols, rows)
    keep = []
    for i, p in enumerate(names):
        c.cd(i + 1)
        h = ROOT.TH1D(uid("pl"), f"{p};pull;toys", 30, -4, 4)
        h.SetDirectory(0)
        for v in pulls[p]:
            if np.isfinite(v):
                h.Fill(float(v))
        h.Draw("HIST")
        if h.GetEntries() > 5:
            f = ROOT.TF1(uid("pg"), "gaus", -4, 4)
            h.Fit(f, "QN0")
            f.SetLineColor(ROOT.kRed + 1)
            f.Draw("SAME")
            keep.append(f)
        keep.append(h)
    c._keep += keep
    return c
