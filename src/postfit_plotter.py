import numpy as np
import uproot
import yaml
import ROOT
from src.style import StyleManager


class PostfitPlotter:
    """
    Produces pre/postfit comparison plots from a Combine FitDiagnostics output.

    The fit config YAML drives bin ordering via shape_transfer_fit.bin_association,
    which maps channel names (in display order) to their list of bins.  Each bin
    contributes exactly one entry to the unrolled x-axis, so the total number of
    bins is sum(len(bins) for each channel).

    Typical layout for splitCRHad_RISR (6 bins):
      Group 1 – Ch1CRHad: bins 1-3  (dxySig < 80,    RISR [0.4,0.6], [0.6,0.8], [0.8,1.0])
      Group 2 – Ch2CRHad: bins 4-6  (80 ≤ dxySig < 1000, same RISR scan)
    """

    # RISR bin labels indexed by the 2-char suffix used in bin names (00, 10, 20, ...)
    RISR_LABELS = {
        "00": "[0.4, 0.6]",
        "10": "[0.6, 0.8]",
        "20": "[0.8, 1.0]",
    }

    # Human-readable channel labels (can be overridden via fit config or constructor)
    DEFAULT_CHANNEL_LABELS = {
        # splitCRHad channels (dxySig split, 1 hadronic SV)
        "Ch1CRHad": "d_{xy}/#sigma_{d_{xy}} < 80",
        "Ch2CRHad": "80 #leq d_{xy}/#sigma_{d_{xy}} < 1000",
        "Ch1CRLep": "d_{xy}/#sigma_{d_{xy}} < 80 (lep)",
        "Ch2CRLep": "80 #leq d_{xy}/#sigma_{d_{xy}} < 1000 (lep)",
        # multiCR channels (SV multiplicity split, full CR)
        "NHad1CR":   "1 hadronic SV",
        "NLep1CR":   "1 leptonic SV",
        "NHadGe2CR": "#geq 2 hadronic SVs",
        "NLepGe2CR": "#geq 2 leptonic SVs",
        "NHadLepCR": "#geq 1 had + #geq 1 lep SV",
    }

    def __init__(self, luminosity=136, energy=13):
        self.style = StyleManager(luminosity=luminosity, energy=energy)
        self.style.set_style()
        self.luminosity = luminosity
        self.energy = energy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot(self, fit_result_path, fit_config_path, output_prefix="postfit",
             output_format="pdf", channel_labels=None):
        """
        Main entry point.  Reads the fit result and config, then produces:
          <output_prefix>_prefit.<format>
          <output_prefix>_postfit.<format>

        Parameters
        ----------
        fit_result_path : str
            Path to fitDiagnostics ROOT file.
        fit_config_path : str
            Path to fit config YAML (contains shape_transfer_fit.bin_association).
        output_prefix : str
            Prefix for output file names.
        output_format : str
            "pdf", "png", or "eps".
        channel_labels : dict or None
            Override display labels for channels, e.g. {"Ch1CRHad": "CR1"}.
        """
        bin_order = self._parse_bin_order(fit_config_path)
        labels = dict(self.DEFAULT_CHANNEL_LABELS)
        if channel_labels:
            labels.update(channel_labels)

        f = uproot.open(fit_result_path)

        pre_bkg, pre_bkg_err, pre_data, pre_data_eyl, pre_data_eyh = \
            self._extract_yields(f, "shapes_prefit", bin_order)

        post_bkg, post_bkg_err, post_data, post_data_eyl, post_data_eyh = \
            self._extract_yields(f, "shapes_fit_b", bin_order)

        decorations = self._build_decorations(bin_order, labels)

        canvas_pre = self._draw_canvas(
            pre_bkg, pre_bkg_err, pre_data, pre_data_eyl, pre_data_eyh,
            decorations, name="prefit", title="Prefit"
        )
        canvas_post = self._draw_canvas(
            post_bkg, post_bkg_err, post_data, post_data_eyl, post_data_eyh,
            decorations, name="postfit", title="Postfit"
        )

        if output_format == "root":
            root_path = f"{output_prefix}.root"
            f_out = ROOT.TFile(root_path, "RECREATE")
            canvas_pre.Write()
            canvas_post.Write()
            f_out.Close()
            print(f"Saved: {root_path}")
        else:
            canvas_pre.SaveAs(f"{output_prefix}_prefit.{output_format}")
            canvas_post.SaveAs(f"{output_prefix}_postfit.{output_format}")
            print(f"Saved: {output_prefix}_prefit.{output_format}")
            print(f"Saved: {output_prefix}_postfit.{output_format}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_bin_order(self, fit_config_path):
        """
        Returns an OrderedList of (channel_name, [bin_name, ...]) pairs,
        preserving YAML insertion order.

        Returns
        -------
        list of (str, list[str])
            e.g. [("Ch1CRHad", ["Ch1CRHad00","Ch1CRHad10","Ch1CRHad20"]),
                  ("Ch2CRHad", ["Ch2CRHad00","Ch2CRHad10","Ch2CRHad20"])]
        """
        with open(fit_config_path) as fh:
            cfg = yaml.safe_load(fh)

        assoc = cfg.get("shape_transfer_fit", {}).get("bin_association", {})
        return [(ch, bins) for ch, bins in assoc.items()]

    def _extract_yields(self, uproot_file, folder, bin_order):
        """
        Read total_background (TH1) and data (TGraphAsymmErrors) from each bin.

        Returns arrays of length N_bins (one entry per combine bin in order).
        """
        bkg_vals = []
        bkg_errs = []
        data_y = []
        data_eyl = []
        data_eyh = []

        for _ch, bins in bin_order:
            for bin_name in bins:
                key_bkg = f"{folder}/{bin_name}/total_background"
                key_data = f"{folder}/{bin_name}/data"

                # Background TH1 (single-bin)
                h = uproot_file[key_bkg]
                bkg_vals.append(float(h.values()[0]))
                bkg_errs.append(float(h.errors()[0]))

                # Data TGraphAsymmErrors
                g = uproot_file[key_data]
                data_y.append(float(g.member("fY")[0]))
                data_eyl.append(float(g.member("fEYlow")[0]))
                data_eyh.append(float(g.member("fEYhigh")[0]))

        return (np.array(bkg_vals), np.array(bkg_errs),
                np.array(data_y), np.array(data_eyl), np.array(data_eyh))

    def _build_decorations(self, bin_order, channel_labels):
        """
        Build axis label lists and group decoration dicts for the unrolled canvas.

        Returns
        -------
        dict with keys:
            n_bins          : total number of bins
            bin_labels      : list of x-axis label strings (one per bin)
            group_labels    : list of dicts {"text", "start", "end"} in bin index space
            separator_bins  : list of bin-boundary indices where vertical lines go
        """
        bin_labels = []
        group_labels = []
        separator_bins = []
        cursor = 0

        for i, (ch, bins) in enumerate(bin_order):
            start = cursor
            for bin_name in bins:
                suffix = bin_name[-2:]
                label = self.RISR_LABELS.get(suffix, suffix)
                bin_labels.append(label)
                cursor += 1
            end = cursor

            ch_label = channel_labels.get(ch, ch)
            group_labels.append({"text": ch_label, "start": start, "end": end})

            # Add separator after every group except the last
            if i < len(bin_order) - 1:
                separator_bins.append(end)

        return {
            "n_bins": cursor,
            "bin_labels": bin_labels,
            "group_labels": group_labels,
            "separator_bins": separator_bins,
        }

    def _draw_canvas(self, bkg_vals, bkg_errs, data_y, data_eyl, data_eyh,
                     decorations, name="postfit", title="Postfit"):
        """
        Draw the unrolled pre/postfit comparison canvas with a ratio pad.
        """
        n_bins = decorations["n_bins"]

        # ---- histograms ----
        h_bkg = ROOT.TH1F(f"h_bkg_{name}", "", n_bins, 0, n_bins)
        h_bkg.SetDirectory(0)
        h_bkg.Sumw2()
        for i in range(n_bins):
            h_bkg.SetBinContent(i + 1, bkg_vals[i])
            h_bkg.SetBinError(i + 1, bkg_errs[i])

        h_bkg.SetFillColor(ROOT.kAzure - 9)
        h_bkg.SetLineColor(ROOT.kBlue + 1)
        h_bkg.SetLineWidth(2)
        h_bkg.SetStats(0)

        # Data as TGraphAsymmErrors
        g_data = ROOT.TGraphAsymmErrors(n_bins)
        for i in range(n_bins):
            x = i + 0.5
            g_data.SetPoint(i, x, data_y[i])
            g_data.SetPointError(i, 0.5, 0.5, data_eyl[i], data_eyh[i])
        g_data.SetMarkerStyle(20)
        g_data.SetMarkerSize(1.2)
        g_data.SetMarkerColor(ROOT.kBlack)
        g_data.SetLineColor(ROOT.kBlack)
        g_data.SetLineWidth(2)

        # Ratio: data / postfit
        h_ratio = ROOT.TH1F(f"h_ratio_{name}", "", n_bins, 0, n_bins)
        h_ratio.SetDirectory(0)
        for i in range(n_bins):
            b = bkg_vals[i]
            d = data_y[i]
            if b > 0:
                h_ratio.SetBinContent(i + 1, d / b)
                # Propagate asymmetric data errors symmetrically (take larger side)
                h_ratio.SetBinError(i + 1, max(data_eyl[i], data_eyh[i]) / b)
            else:
                h_ratio.SetBinContent(i + 1, 0)
                h_ratio.SetBinError(i + 1, 0)

        # Ratio uncertainty band (bkg stat only, normalized to 1)
        h_ratio_band = ROOT.TH1F(f"h_ratio_band_{name}", "", n_bins, 0, n_bins)
        h_ratio_band.SetDirectory(0)
        for i in range(n_bins):
            b = bkg_vals[i]
            h_ratio_band.SetBinContent(i + 1, 1.0)
            h_ratio_band.SetBinError(i + 1, bkg_errs[i] / b if b > 0 else 0)
        h_ratio_band.SetFillColor(ROOT.kGray + 1)
        h_ratio_band.SetFillStyle(3345)
        h_ratio_band.SetMarkerSize(0)
        h_ratio_band.SetLineColor(0)

        # ---- canvas layout ----
        canvas = ROOT.TCanvas(f"c_{name}", title, 1200, 700)
        canvas.SetFillColor(0)

        left_m = 0.10
        right_m = 0.04
        split = 0.30  # fraction of canvas height for ratio pad

        pad1 = ROOT.TPad(f"pad1_{name}", "main", 0, split, 1, 1)
        pad1.SetBottomMargin(0.02)
        pad1.SetTopMargin(0.10)
        pad1.SetLeftMargin(left_m)
        pad1.SetRightMargin(right_m)
        pad1.SetLogy(True)
        pad1.SetGridx(True)
        pad1.Draw()

        pad2 = ROOT.TPad(f"pad2_{name}", "ratio", 0, 0, 1, split)
        pad2.SetTopMargin(0.02)
        pad2.SetBottomMargin(0.38)
        pad2.SetLeftMargin(left_m)
        pad2.SetRightMargin(right_m)
        pad2.SetGridx(True)
        pad2.SetGridy(True)
        pad2.Draw()

        # ---- main pad ----
        pad1.cd()

        max_val = max(np.max(bkg_vals + bkg_errs), np.max(data_y + data_eyh))
        if max_val <= 0:
            max_val = 1.0
        min_val = max(0.5, 0.3 * min(v for v in np.concatenate([bkg_vals, data_y]) if v > 0) if any(v > 0 for v in np.concatenate([bkg_vals, data_y])) else 0.5)

        h_bkg.SetMaximum(max_val * 8.0)
        h_bkg.SetMinimum(min_val)

        # Y axis
        h_bkg.GetYaxis().SetTitle("Events / bin")
        h_bkg.GetYaxis().SetTitleSize(0.075)
        h_bkg.GetYaxis().SetTitleOffset(0.65)
        h_bkg.GetYaxis().SetLabelSize(0.065)
        h_bkg.GetYaxis().CenterTitle(True)
        # X axis — hide labels (drawn manually below)
        h_bkg.GetXaxis().SetLabelSize(0)
        h_bkg.GetXaxis().SetTickLength(0)

        h_bkg.Draw("E2")  # filled band with errors
        h_bkg.Draw("HIST SAME")  # outline
        g_data.Draw("PZ SAME")  # data points, no horizontal error bars

        # Legend
        leg = ROOT.TLegend(0.75, 0.53, 0.99, 0.75)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.055)
        leg.AddEntry(g_data, "Data", "PE")
        leg.AddEntry(h_bkg, f"{title} bkg", "LF")
        leg.Draw()

        # CMS labels — match positioning used in rest of repo
        self.style.draw_cms_labels(
            cms_x=0.1, cms_y=0.92,
            prelim_str="Preliminary",
            prelim_x=0.19,
            lumi_x=0.96,
            cms_text_size_mult=1.92
        )

        # Group labels and separator lines (drawn on pad1 via NDC overlay)
        latex_objs = self._draw_group_labels(pad1, decorations, left_m, right_m)

        # ---- ratio pad ----
        pad2.cd()

        h_ratio.SetMaximum(1.99)
        h_ratio.SetMinimum(0.0)
        h_ratio.GetYaxis().SetTitle("Data / Fit")
        h_ratio.GetYaxis().SetTitleSize(0.16)
        h_ratio.GetYaxis().SetTitleOffset(0.28)
        h_ratio.GetYaxis().SetLabelSize(0.15)
        h_ratio.GetYaxis().SetNdivisions(504)
        h_ratio.GetYaxis().CenterTitle(True)
        h_ratio.GetXaxis().SetLabelSize(0)   # hide — drawn manually
        h_ratio.GetXaxis().SetTickLength(0)
        h_ratio.SetMarkerStyle(20)
        h_ratio.SetMarkerSize(1.0)
        h_ratio.SetLineColor(ROOT.kBlack)
        h_ratio.SetStats(0)

        h_ratio.Draw("PE")
        h_ratio_band.Draw("E2 SAME")
        h_ratio.Draw("PE SAME")

        # Unity line
        line_one = ROOT.TLine(0, 1, n_bins, 1)
        line_one.SetLineColor(ROOT.kRed)
        line_one.SetLineWidth(2)
        line_one.SetLineStyle(2)
        line_one.Draw()

        # x-axis bin labels on ratio pad
        bin_label_objs = self._draw_bin_labels(pad2, decorations, left_m, right_m)

        # ---- separator lines across both pads ----
        sep_objs = self._draw_separator_lines(canvas, decorations, left_m, right_m, split)

        canvas.Modified()
        canvas.Update()

        # Keep objects alive
        canvas._keep = [h_bkg, g_data, h_ratio, h_ratio_band, line_one,
                        leg, latex_objs, bin_label_objs, sep_objs]
        return canvas

    def _draw_group_labels(self, pad, decorations, left_m, right_m):
        """Draw channel group labels centered above each group in pad1 NDC."""
        pad.cd()
        n_bins = decorations["n_bins"]
        data_width = 1.0 - left_m - right_m
        latex_objs = []

        latex = ROOT.TLatex()
        latex.SetNDC(True)
        latex.SetTextFont(42)
        latex.SetTextSize(0.062)
        latex.SetTextAlign(22)

        for grp in decorations["group_labels"]:
            center = (grp["start"] + grp["end"]) / 2.0
            x_ndc = left_m + (center / n_bins) * data_width
            # y just inside the top of pad1 (pad1 top margin is 0.10, so ~0.87 in pad NDC)
            latex.DrawLatex(x_ndc, 0.83, grp["text"])

        latex_objs.append(latex)
        return latex_objs

    def _draw_bin_labels(self, pad, decorations, left_m, right_m):
        """Draw RISR bin labels on the x-axis of the ratio pad."""
        pad.cd()
        n_bins = decorations["n_bins"]
        data_width = 1.0 - left_m - right_m
        objs = []

        latex = ROOT.TLatex()
        latex.SetNDC(True)
        latex.SetTextFont(42)
        latex.SetTextSize(0.14)
        latex.SetTextAlign(22)

        for i, label in enumerate(decorations["bin_labels"]):
            x_center = i + 0.5
            x_ndc = left_m + (x_center / n_bins) * data_width
            # bottom_margin = 0.38, so label at ~0.22 NDC in pad gives good spacing
            latex.DrawLatex(x_ndc, 0.26, label)

        # "R_{ISR}" axis title below the labels
        x_title_ndc = left_m + 0.5 * data_width
        title_latex = ROOT.TLatex()
        title_latex.SetNDC(True)
        title_latex.SetTextFont(42)
        title_latex.SetTextSize(0.18)
        title_latex.SetTextAlign(22)
        title_latex.DrawLatex(x_title_ndc, 0.09, "R_{ISR}")
        objs.extend([latex, title_latex])
        return objs

    def _draw_separator_lines(self, canvas, decorations, left_m, right_m, split):
        """Draw vertical dashed lines at group boundaries, spanning both pads."""
        n_bins = decorations["n_bins"]
        data_width = 1.0 - left_m - right_m

        # Overlay transparent pad covering full canvas
        overlay = ROOT.TPad("sep_overlay", "", 0, 0, 1, 1)
        overlay.SetFillStyle(0)
        overlay.SetFrameFillStyle(0)
        overlay.SetBorderSize(0)
        overlay.SetBorderMode(0)
        overlay.SetMargin(0, 0, 0, 0)
        overlay.SetBit(ROOT.kCannotPick)
        canvas.cd()
        overlay.Draw()
        overlay.cd()

        lines = []
        for sep_bin in decorations["separator_bins"]:
            x_ndc = left_m + (sep_bin / n_bins) * data_width
            # In full canvas NDC: ratio pad bottom is 0, top of main pad is 1
            line = ROOT.TLine()
            line.SetNDC(True)
            line.SetLineColor(ROOT.kGray + 2)
            line.SetLineWidth(2)
            line.SetLineStyle(2)
            line.DrawLine(x_ndc, 0.06, x_ndc, 0.93)
            lines.append(line)

        canvas.Modified()
        canvas.Update()
        return [overlay] + lines
