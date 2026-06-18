import numpy as np
import uproot
import yaml
import ROOT
import re
from collections import namedtuple
from src.style import StyleManager
from src.plotter import PlotterDataMC

# ── Bin label tables ──────────────────────────────────────────────────────────

RISR_LABELS = {
    "00": "[0.4, 0.6)",
    "10": "[0.6, 0.75)",
    "20": "[0.75, 0.9)",
    "30": "#geq 0.9",
}

RISR_COMPACT_LABELS = {
    "00": "R_{ISR}^{lo}",
    "10": "R_{ISR}^{med}",
    "20": "R_{ISR}^{hi-}",  # [0.75, 0.9) — only valid for 4-bin channels
    "30": "R_{ISR}^{hi+}",
}

ABCD_COMPRESSED_BIN_LABELS = {
    "00": "#geq 0.75",
}

MSRS_MS_LABELS = {
    "0": "M_{S}^{SR,-}",
    "1": "M_{S}^{SR,+}",
}
MSRS_RS_LABELS = {
    "0": "R_{S}^{-}",
    "1": "R_{S}^{+}",
}

MS_DELAYED_LABELS = {
    "00": "[0.7, 1)",
    "10": "#geq 1",
}

COMBINED_MS_DELAYED_LABELS = {
    "00": "M_{S}^{CR}",
    "10": "M_{S}^{SR}",
}

COMPRESSED_FINAL_TOP_LABELS = {
    "SVonly":    "SV only",
    "DelPho":    "#gamma_{d}",
    "Eq2Pho":    "2#gamma",
    "MixDel":    "#gamma_{d}+SV",
    "MixPrompt": "#gamma_{p}+SV",
}

COMPRESSED_FINAL_RISR_LABELS = {
    "SVonly":    {"00": "lo",  "10": "med", "20": "hi", "30": "hi+"},
    "DelPho":    {"10": "med", "00": "hi",  "20": "hi", "30": "hi+"},
    "Eq2Pho":    {"00": "lo",  "10": "med", "20": "hi", "30": "hi+"},
    "MixDel":     {"10": "med", "00": "hi",  "20": "hi", "30": "hi+"},  # ABCD
    "MixDel_STF": {"00": "lo",  "10": "med", "20": "hi", "30": "hi+"},  # shape_transfer
    "MixPrompt": {"00": "lo",  "10": "med", "20": "hi", "30": "hi+"},
}

COMPRESSED_FINAL_GLOSSARY = [
    ("R_{ISR}^{lo}",  "[0.4, 0.6)"),
    ("R_{ISR}^{med}", "[0.6, 0.75)"),
    ("R_{ISR}^{hi}",  "#geq 0.75"),
    ("R_{ISR}^{hi+}", "#geq 0.9"),
]

NONCOMPRESSED_FINAL_TOP_LABELS = {
    "lep_sv":          "Lep SV",
    "had_sv":          "Had SV",
    "delayed_photon":  "#gamma_{d}",
    "one_prompt":      "1#gamma_{p}",
    "two_prompt":      "2#gamma_{p}",
}

NONCOMPRESSED_FINAL_GLOSSARY = [
    ("#gamma_{d}", ("M #geq 2.0 TeV; R #geq 0.3",)),
    ("Lep SV", ("M: [2.0,2.6), #geq 2.6", "R: [0.15,0.3), #geq 0.3")),
    ("Had SV", ("M: [2.0,2.6), [2.6,3.0), #geq 3.0", "R: [0.15,0.35), #geq 0.35")),
    ("1#gamma_{p}", ("M: [2.5,2.7), #geq 2.7", "R: <0.35, [0.35,0.45), #geq 0.45")),
    ("2#gamma_{p}", ("M: [2.5,2.7), #geq 2.7", "R: <0.3, #geq 0.3")),
]

CHANNEL_LABELS = {
    # Compressed shape_transfer
    "Ch1CRHadLow":        "1 Had. SV, d_{xy}/#sigma_{d_{xy}} #in [100, 600)",
    "Ch2CRHadHigh":       "1 Had. SV, d_{xy}/#sigma_{d_{xy}} #in [600, 1000)",
    "Ch3CRPhoIso1":       "Prompt #gamma, ISO #in [0.4, 0.5)",
    "Ch4CRPhoIso2":       "Prompt #gamma, ISO #in [0.5, 0.52)",
    # Uncompressed shape_transfer
    "Ch5CRPho1Iso":       "Prompt #gamma, ISO #in [0.4, 0.5)",
    "Ch6CRPho2Iso":       "Prompt #gamma, ISO #in [0.5, 0.52)",
    "Ch7CRHadLow":        "1 Had. SV, d_{xy}/#sigma_{d_{xy}} #in [100, 600)",
    "Ch8CRHadHigh":       "1 Had. SV, d_{xy}/#sigma_{d_{xy}} #in [600, 1000)",
    # Uncompressed ABCD
    "Ch1CRPhoBHEarly":    "BH #gamma, early timing",
    "Ch2CRPhoBHLate":     "BH #gamma, late timing",
    "Ch3CRPhonotBHEarly": "Non-BH #gamma, early timing",
    "Ch4CRPhonotBHLate":  "Non-BH #gamma, late timing",
    # Compressed FullRegions — shape_transfer
    "SVonly_AnchorCR":    "SV CR",
    "SVonly_SR":          "SV SR",
    "Eq2Pho_MedIsoCR":   "2#gamma Med. Iso CR",
    "Eq2Pho_TightIsoSR": "2#gamma Tight Iso SR",
    "MixPrompt_CR":       "1#gamma+SV CR",
    "MixPromptSV_SR":     "1#gamma+SV SR",
    # Compressed FullRegions — ABCD
    "DelPho_BHEarly":     "#gamma t_{-}^{BH}",
    "DelPho_BHLate":      "#gamma t_{+}^{BH}",
    "DelPho_NotBHEarly":  "#gamma t_{-}^{!BH}",
    "DelPho_NotBHLate":   "#gamma t_{+}^{!BH}",
    "MixDel_BHEarly":     "SV+#gamma t_{-}^{BH}",
    "MixDel_BHLate":      "SV+#gamma t_{+}^{BH}",
    "MixDel_NotBHEarly":  "SV+#gamma t_{-}^{!BH}",
    "MixDel_NotBHLate":   "SV+#gamma t_{+}^{!BH}",
}

# Compact labels used only in the combined plot where space is tight.
# ABCD channels use the comma-split two-line feature: "group, sub" → stacked.
COMBINED_CHANNEL_LABELS = {
    # ABCD
    "Ch1CRPhoBHEarly":    "#gamma^{ BH}, t_{ -}",
    "Ch2CRPhoBHLate":     "#gamma^{ BH}, t_{ +}",
    "Ch3CRPhonotBHEarly": "#gamma^{ !BH}, t_{ -}",
    "Ch4CRPhonotBHLate":  "#gamma^{ !BH}, t_{ +}",
    # Shape transfer
    "Ch5CRPho1Iso":       "#gamma^{ iso,1}",
    "Ch6CRPho2Iso":       "#gamma^{ iso,2}",
    "Ch7CRHadLow":        "SV_{qq}^{CR,-}",
    "Ch8CRHadHigh":       "SV_{qq}^{CR,+}",
    # Compressed FullRegions — shape_transfer
    "SVonly_AnchorCR":    "SV CR",
    "SVonly_SR":          "SV SR",
    "Eq2Pho_MedIsoCR":   "2#gamma CR",
    "Eq2Pho_TightIsoSR": "2#gamma SR",
    "MixPrompt_CR":       "#gamma+SV CR",
    "MixPromptSV_SR":     "#gamma+SV SR",
    # Compressed FullRegions — ABCD
    "DelPho_BHEarly":     "#gamma, t_{-}^{BH}",
    "DelPho_BHLate":      "#gamma, t_{+}^{BH}",
    "DelPho_NotBHEarly":  "#gamma, t_{-}^{!BH}",
    "DelPho_NotBHLate":   "#gamma, t_{+}^{!BH}",
    "MixDel_BHEarly":     "SV+#gamma, t_{-}^{BH}",
    "MixDel_BHLate":      "SV+#gamma, t_{+}^{BH}",
    "MixDel_NotBHEarly":  "SV+#gamma, t_{-}^{!BH}",
    "MixDel_NotBHLate":   "SV+#gamma, t_{+}^{!BH}",
}

# ── Parsed fit configuration ──────────────────────────────────────────────────

FitConfig = namedtuple("FitConfig", [
    "mode",            # 'compressed' | 'uncompressed'
    "shape_bin_order", # [(ch, [bins]), ...]
    "shape_pairs",     # {anchor_ch: [buoy_chs]}
    "anchor_bin",      # '00' or '01'
    "abcd_bin_order",  # [(ch, [bins]), ...] or []
    "abcd_pairs",      # {sr_ch: [cr_chs]} or {}
])

_canvas_counter = [0]  # mutable counter for unique ROOT names


def _uid():
    _canvas_counter[0] += 1
    return _canvas_counter[0]


def _datamc_background_key(process_name: str) -> str:
    """Map Combine/SampleTool process names onto labels known by PlotterDataMC."""
    label_map = {
        "Wjets": "WJets",
        "Zjets": "ZJets",
        "Gjets": "GJets",
        "Top": "TTJets",
    }
    return label_map.get(process_name, process_name)


def _compressed_display_key(bin_name: str) -> str:
    if bin_name.startswith("Val_"):
        return bin_name[len("Val_"):]
    return bin_name


def _noncompressed_suffix(bin_name: str) -> str:
    match = re.search(r"(?:Bin)?(\d{2})$", bin_name)
    return match.group(1) if match else bin_name[-2:]


def _noncompressed_mr_label(suffix: str, channel: str = "") -> str:
    if channel == "delayed_photon":
        return "#splitline{M}{R}"
    if len(suffix) != 2 or not suffix.isdigit():
        return suffix
    ms_labels = {"0": "M^{-}", "1": "M^{+}", "2": "M^{++}" if channel == "had_sv" else "M^{2}"}
    if channel == "one_prompt":
        rs_labels = {"0": "R^{-}", "1": "R^{0}", "2": "R^{+}"}
    else:
        rs_labels = {"0": "R^{-}", "1": "R^{+}", "2": "R^{2}"}
    ms = ms_labels.get(suffix[0], "M^{" + suffix[0] + "}")
    rs = rs_labels.get(suffix[1], "R^{" + suffix[1] + "}")
    return f"#splitline{{{ms}}}{{{rs}}}"


def _make_channel_label(ch):
    """Derive a short display label from a channel name using pattern matching."""
    prefixes = [
        ("MixPrompt", "#gamma+SV"),
        ("MixDel",    "SV+#gamma"),
        ("SVonly",    "SV"),
        ("Eq2Pho",    "2#gamma"),
        ("DelPho",    "#gamma"),
    ]
    symbol = ch
    for pfx, sym in prefixes:
        if ch.startswith(pfx):
            symbol = sym
            break

    if   "NotBHEarly" in ch: sfx = "t_{-}^{!BH}"
    elif "NotBHLate"  in ch: sfx = "t_{+}^{!BH}"
    elif "BHEarly"    in ch: sfx = "t_{-}^{BH}"
    elif "BHLate"     in ch: sfx = "t_{+}^{BH}"
    elif "AnchorCR"   in ch: sfx = "Anchor CR"
    elif "CRLow"      in ch: sfx = "CR"
    elif "TightIso"   in ch: sfx = "Tight Iso SR"
    elif "MedIso"     in ch: sfx = "Med. Iso CR"
    elif ch.endswith(("_SR", "SR")): sfx = "SR"
    elif ch.endswith(("_CR", "CR")): sfx = "CR"
    else:                    sfx = ""

    return f"{symbol} {sfx}".strip() if sfx else symbol


# ── Main class ────────────────────────────────────────────────────────────────

class FitPlotter:
    """
    Produces pre/postfit comparison plots from a Combine FitDiagnostics output.

    Handles compressed (1D RISR) and uncompressed (2D Ms×Rs + ABCD) fits.

    Output per call to plot_all():
      Compressed:
        {prefix}_shape_{pre,post}fit.{fmt}
        {prefix}_pair_{A}_{B}_together_{pre,post}fit.{fmt}
        {prefix}_pair_{A}_{B}_sidebyside_{pre,post}fit.{fmt}
      Uncompressed (adds):
        {prefix}_abcd_flat_{pre,post}fit.{fmt}
        {prefix}_abcd_grid_{pre,post}fit.{fmt}
        {prefix}_combined_{pre,post}fit.{fmt}
    """

    def __init__(self, luminosity=136, energy=13):
        self.style = StyleManager(luminosity=luminosity, energy=energy)
        self.style.set_style()
        self.datamc_plotter = PlotterDataMC(self.style)
        self.luminosity = luminosity
        self.energy = energy

    # ── Public API ────────────────────────────────────────────────────────────

    def plot_all(self, fit_result_path, fit_config_path, output_prefix="fit",
                 output_format="pdf", mode_override=None, label_scheme="auto",
                 show_sr=False, data_mc=False):
        cfg = self._load_config(fit_config_path, mode_override)
        f   = uproot.open(fit_result_path)

        print(f"[FitPlotter] mode={cfg.mode} | "
              f"{len(cfg.shape_bin_order)} shape channels"
              + (f" | {len(cfg.abcd_bin_order)} ABCD channels" if cfg.abcd_bin_order else ""))

        if label_scheme == "auto" and cfg.mode == "uncompressed" and self._has_noncompressed_final_bins(cfg):
            label_scheme = "noncompressed-final"
        print(f"[FitPlotter] label_scheme={label_scheme}")

        use_cf_labels = (label_scheme == "compressed-final")
        use_ncf_labels = (label_scheme == "noncompressed-final")
        bin_scheme     = "msrs" if cfg.mode == "uncompressed" else "risr"
        shape_all_bins = [b for _, bins in cfg.shape_bin_order for b in bins]

        pre_shape  = self._extract_yields(f, "shapes_prefit", shape_all_bins)
        post_shape = self._extract_yields(f, "shapes_fit_b",  shape_all_bins)
        if data_mc:
            pre_shape_stack = self._extract_stack_hists(f, "shapes_prefit", shape_all_bins, "shape_prefit")
            post_shape_stack = self._extract_stack_hists(f, "shapes_fit_b", shape_all_bins, "shape_postfit")

        if cfg.abcd_bin_order:
            abcd_all_bins = [b for _, bins in cfg.abcd_bin_order for b in bins]
            pre_abcd  = self._extract_yields(f, "shapes_prefit", abcd_all_bins)
            post_abcd = self._extract_yields(f, "shapes_fit_b",  abcd_all_bins)
            if data_mc:
                pre_abcd_stack = self._extract_stack_hists(f, "shapes_prefit", abcd_all_bins, "abcd_prefit")
                post_abcd_stack = self._extract_stack_hists(f, "shapes_fit_b", abcd_all_bins, "abcd_postfit")

        # Collect (canvas_pre, canvas_post, short_name) for all plots, then
        # flush to a single ROOT file or individual image files at the end.
        plots = []

        # ── comprehensive shape_transfer ──────────────────────────────────────
        if use_cf_labels:
            shape_deco = self._build_compressed_final_decorations(cfg.shape_bin_order)
        elif use_ncf_labels:
            shape_deco = self._build_noncompressed_final_decorations(cfg.shape_bin_order)
        else:
            shape_deco = self._build_decorations(cfg.shape_bin_order, bin_scheme)
        plots.append((
            self._draw_datamc_canvas(pre_shape_stack, shape_deco, f"shp_pre_{_uid()}", "Prefit") if data_mc else self._draw_standard_canvas(*pre_shape,  shape_deco, f"shp_pre_{_uid()}",  "Prefit"),
            self._draw_datamc_canvas(post_shape_stack, shape_deco, f"shp_pst_{_uid()}", "Postfit") if data_mc else self._draw_standard_canvas(*post_shape, shape_deco, f"shp_pst_{_uid()}", "Postfit"),
            "shape",
        ))

        # ── per-pair plots ────────────────────────────────────────────────────
        for anchor_ch, buoy_chs in cfg.shape_pairs.items():
            anchor_bins = dict(cfg.shape_bin_order)[anchor_ch]
            for buoy_ch in buoy_chs:
                buoy_bins = dict(cfg.shape_bin_order)[buoy_ch]
                ai = shape_all_bins.index(anchor_bins[0])
                bi = shape_all_bins.index(buoy_bins[0])
                na, nb = len(anchor_bins), len(buoy_bins)

                a_pre  = tuple(arr[ai:ai+na] for arr in pre_shape)
                a_post = tuple(arr[ai:ai+na] for arr in post_shape)
                b_pre  = tuple(arr[bi:bi+nb] for arr in pre_shape)
                b_post = tuple(arr[bi:bi+nb] for arr in post_shape)
                if data_mc:
                    a_pre_stack = self._slice_hist_bundle(pre_shape_stack, ai, na, f"a_pre_{_uid()}")
                    a_post_stack = self._slice_hist_bundle(post_shape_stack, ai, na, f"a_post_{_uid()}")
                    b_pre_stack = self._slice_hist_bundle(pre_shape_stack, bi, nb, f"b_pre_{_uid()}")
                    b_post_stack = self._slice_hist_bundle(post_shape_stack, bi, nb, f"b_post_{_uid()}")

                tag       = f"{self._ch_short(anchor_ch)}_{self._ch_short(buoy_ch)}"
                pair_ord  = [(anchor_ch, anchor_bins), (buoy_ch, buoy_bins)]
                if use_cf_labels:
                    pair_deco = self._build_compressed_final_decorations(pair_ord)
                elif use_ncf_labels:
                    pair_deco = self._build_noncompressed_final_decorations(pair_ord)
                else:
                    pair_deco = self._build_decorations(pair_ord, bin_scheme)
                tog_pre   = tuple(np.concatenate([a, b]) for a, b in zip(a_pre,  b_pre))
                tog_post  = tuple(np.concatenate([a, b]) for a, b in zip(a_post, b_post))
                if data_mc:
                    tog_pre_stack = self._concat_hist_bundles(a_pre_stack, b_pre_stack, f"tog_pre_{_uid()}")
                    tog_post_stack = self._concat_hist_bundles(a_post_stack, b_post_stack, f"tog_post_{_uid()}")

                plots.append((
                    self._draw_datamc_canvas(tog_pre_stack, pair_deco, f"tog_pre_{_uid()}", "Prefit") if data_mc else self._draw_standard_canvas(*tog_pre,  pair_deco, f"tog_pre_{_uid()}",  "Prefit"),
                    self._draw_datamc_canvas(tog_post_stack, pair_deco, f"tog_pst_{_uid()}", "Postfit") if data_mc else self._draw_standard_canvas(*tog_post, pair_deco, f"tog_pst_{_uid()}", "Postfit"),
                    f"pair_{tag}_together",
                ))

                if data_mc:
                    continue

                if use_cf_labels:
                    a_deco = self._build_compressed_final_decorations([(anchor_ch, anchor_bins)])
                    b_deco = self._build_compressed_final_decorations([(buoy_ch, buoy_bins)])
                elif use_ncf_labels:
                    a_deco = self._build_noncompressed_final_decorations([(anchor_ch, anchor_bins)])
                    b_deco = self._build_noncompressed_final_decorations([(buoy_ch, buoy_bins)])
                else:
                    a_deco = self._build_decorations([(anchor_ch, anchor_bins)], bin_scheme)
                    b_deco = self._build_decorations([(buoy_ch,   buoy_bins)],   bin_scheme)
                plots.append((
                    self._draw_sidebyside_canvas(*a_pre,  *b_pre,  a_deco, b_deco,
                                                 f"sbs_pre_{_uid()}",  "Prefit"),
                    self._draw_sidebyside_canvas(*a_post, *b_post, a_deco, b_deco,
                                                 f"sbs_pst_{_uid()}", "Postfit"),
                    f"pair_{tag}_sidebyside",
                ))

        # ── ABCD plots (uncompressed only) ────────────────────────────────────
        if cfg.abcd_bin_order:
            multi_bin_abcd = max(len(b) for _, b in cfg.abcd_bin_order) > 1
            if cfg.mode == "uncompressed":
                abcd_scheme = "ms_delayed" if multi_bin_abcd else "abcd"
            else:
                # Compressed multi-bin ABCD cells carry R_ISR bins, not M_S bins
                abcd_scheme = "risr" if multi_bin_abcd else "abcd"
            if use_cf_labels:
                abcd_flat_deco = self._build_compressed_final_decorations(cfg.abcd_bin_order)
            elif use_ncf_labels:
                abcd_flat_deco = self._build_noncompressed_final_decorations(cfg.abcd_bin_order)
            else:
                abcd_flat_deco = self._build_decorations(cfg.abcd_bin_order, abcd_scheme)
            plots.append((
                self._draw_datamc_canvas(pre_abcd_stack, abcd_flat_deco, f"abf_pre_{_uid()}", "Prefit") if data_mc else self._draw_standard_canvas(*pre_abcd,  abcd_flat_deco, f"abf_pre_{_uid()}",  "Prefit"),
                self._draw_datamc_canvas(post_abcd_stack, abcd_flat_deco, f"abf_pst_{_uid()}", "Postfit") if data_mc else self._draw_standard_canvas(*post_abcd, abcd_flat_deco, f"abf_pst_{_uid()}", "Postfit"),
                "abcd_flat",
            ))

            sr_ch = show_sr  # True → color any channel whose name contains "SR"
            if use_cf_labels:
                abcd_grid_deco = self._build_compressed_final_decorations(cfg.abcd_bin_order, sr_ch=sr_ch)
            elif use_ncf_labels:
                abcd_grid_deco = self._build_noncompressed_final_decorations(cfg.abcd_bin_order, sr_ch=sr_ch)
            else:
                abcd_grid_deco = self._build_abcd_grid_decorations(cfg.abcd_bin_order, sr_ch=sr_ch, mode=cfg.mode)
            plots.append((
                self._draw_datamc_canvas(pre_abcd_stack, abcd_grid_deco, f"abg_pre_{_uid()}", "Prefit") if data_mc else self._draw_standard_canvas(*pre_abcd,  abcd_grid_deco, f"abg_pre_{_uid()}",  "Prefit"),
                self._draw_datamc_canvas(post_abcd_stack, abcd_grid_deco, f"abg_pst_{_uid()}", "Postfit") if data_mc else self._draw_standard_canvas(*post_abcd, abcd_grid_deco, f"abg_pst_{_uid()}", "Postfit"),
                "abcd_grid",
            ))

            comb_deco  = self._build_combined_decorations(
                cfg.abcd_bin_order, cfg.shape_bin_order, sr_ch=sr_ch,
                shape_bin_scheme=bin_scheme, abcd_bin_scheme=abcd_scheme,
                label_scheme=label_scheme,
            )
            comb_pre   = tuple(np.concatenate([a, s]) for a, s in zip(pre_abcd,  pre_shape))
            comb_post  = tuple(np.concatenate([a, s]) for a, s in zip(post_abcd, post_shape))
            if data_mc:
                comb_pre_stack = self._concat_hist_bundles(pre_abcd_stack, pre_shape_stack, f"cmb_pre_{_uid()}")
                comb_post_stack = self._concat_hist_bundles(post_abcd_stack, post_shape_stack, f"cmb_post_{_uid()}")
            plots.append((
                self._draw_datamc_canvas(comb_pre_stack, comb_deco, f"cmb_pre_{_uid()}", "Prefit", right_panel=True) if data_mc else self._draw_standard_canvas(*comb_pre,  comb_deco, f"cmb_pre_{_uid()}",  "Prefit",  right_panel=True),
                self._draw_datamc_canvas(comb_post_stack, comb_deco, f"cmb_pst_{_uid()}", "Postfit", right_panel=True) if data_mc else self._draw_standard_canvas(*comb_post, comb_deco, f"cmb_pst_{_uid()}", "Postfit", right_panel=True),
                "combined",
            ))

        self._flush(plots, output_prefix, output_format)

    # ── Config loading ────────────────────────────────────────────────────────

    def _load_config(self, path, mode_override=None):
        with open(path) as fh:
            cfg = yaml.safe_load(fh)

        stf            = cfg.get("shape_transfer_fit", {})
        shape_bin_ass  = stf.get("bin_association",    {})
        shape_ch_ass   = stf.get("channel_association", {})
        anchor_bin     = stf.get("anchor_bin", "00")

        abcd           = cfg.get("ABCD_fit") or {}
        abcd_bin_ass   = abcd.get("bin_association",    {})
        abcd_ch_ass    = abcd.get("channel_association", {})

        mode = mode_override or self._detect_mode(cfg)

        return FitConfig(
            mode=mode,
            shape_bin_order=list(shape_bin_ass.items()),
            shape_pairs=shape_ch_ass,
            anchor_bin=anchor_bin,
            abcd_bin_order=list(abcd_bin_ass.items()),
            abcd_pairs=abcd_ch_ass,
        )

    def _detect_mode(self, cfg):
        assoc = cfg.get("shape_transfer_fit", {}).get("bin_association", {})
        if not assoc:
            return "compressed"
        n = len(next(iter(assoc.values())))
        return "uncompressed" if n == 4 else "compressed"

    def _has_noncompressed_final_bins(self, cfg):
        bin_names = [
            b
            for bin_order in (cfg.shape_bin_order, cfg.abcd_bin_order)
            for _, bins in bin_order
            for b in bins
        ]
        final_tokens = (
            "GeLep",
            "GeHad",
            "geq1PhoBHEarly",
            "geq1PhoBHLate",
            "geq1PhoNotBHEarly",
            "geq1PhoNotBHLate",
            "eq1Pho",
            "eq2Pho",
        )
        return any(any(token in b for token in final_tokens) for b in bin_names)

    # ── Yield extraction ──────────────────────────────────────────────────────

    def _resolve_bin_names(self, uf, folder, bin_list):
        """Map config bin names to actual names in the ROOT file.

        Combine/Higgs tools prefix bin directories with the signal point name
        (e.g. gogoGZ_2300_2200_2100_10_SVonly_AnchorCR00).  This method finds
        the matching key regardless of whether a prefix is present.
        """
        available = {k.split(";")[0] for k in uf[folder].keys()}
        resolved = []
        for bn in bin_list:
            if bn in available:
                resolved.append(bn)
            else:
                match = next((k for k in available if k.endswith(f"_{bn}")), None)
                if match is None:
                    raise uproot.exceptions.KeyInFileError(
                        f"Cannot find bin '{bn}' in folder '{folder}'"
                    )
                resolved.append(match)
        return resolved

    def _extract_yields(self, uf, folder, bin_list):
        bkg, berr, dy, eyl, eyh = [], [], [], [], []
        resolved = self._resolve_bin_names(uf, folder, bin_list)
        for bn in resolved:
            h = uf[f"{folder}/{bn}/total_background"]
            g = uf[f"{folder}/{bn}/data"]
            bkg.append(float(h.values()[0]))
            berr.append(float(h.errors()[0]))
            dy.append(float(g.member("fY")[0]))
            eyl.append(float(g.member("fEYlow")[0]))
            eyh.append(float(g.member("fEYhigh")[0]))
        return (np.array(bkg), np.array(berr),
                np.array(dy),  np.array(eyl), np.array(eyh))

    def _shape_processes(self, uf, folder, resolved_bins):
        skip = {"data", "total", "total_background", "total_signal", "total_covar", "overall_total_covar"}
        processes = []
        seen = set()
        for bn in resolved_bins:
            for key in uf[f"{folder}/{bn}"].keys():
                proc = key.split(";")[0]
                if proc in skip or proc in seen:
                    continue
                seen.add(proc)
                processes.append(proc)
        preferred = ["QCD", "Wjets", "WJets", "Zjets", "ZJets", "Top", "TTJets", "TTXJets", "Gjets", "GJets"]
        rank = {name: i for i, name in enumerate(preferred)}
        return sorted(processes, key=lambda p: (rank.get(p, len(rank)), p))

    def _extract_stack_hists(self, uf, folder, bin_list, name):
        resolved = self._resolve_bin_names(uf, folder, bin_list)
        nbins = len(bin_list)
        data_hist = ROOT.TH1F(f"h_data_{name}_{_uid()}", "data", nbins, 0, nbins)
        data_hist.SetDirectory(0)
        data_hist.Sumw2()

        for i, bn in enumerate(resolved, start=1):
            g = uf[f"{folder}/{bn}/data"]
            data_hist.SetBinContent(i, float(g.member("fY")[0]))
            data_hist.SetBinError(i, max(float(g.member("fEYlow")[0]), float(g.member("fEYhigh")[0])))

        mc_histograms = []
        for proc in self._shape_processes(uf, folder, resolved):
            hist = ROOT.TH1F(f"h_{proc}_{name}_{_uid()}", proc, nbins, 0, nbins)
            hist.SetDirectory(0)
            hist.Sumw2()
            has_yield = False
            for i, bn in enumerate(resolved, start=1):
                path = f"{folder}/{bn}/{proc}"
                if path not in uf:
                    continue
                h = uf[path]
                value = float(h.values()[0])
                error = float(h.errors()[0])
                hist.SetBinContent(i, value)
                hist.SetBinError(i, error)
                has_yield = has_yield or value != 0
            if has_yield:
                mc_histograms.append((hist, proc))

        if not mc_histograms:
            raise RuntimeError(f"No per-process MC shapes found in '{folder}'")

        return {"data": data_hist, "mc": mc_histograms}

    def _slice_hist(self, hist, start, length, name):
        out = ROOT.TH1F(name, hist.GetTitle(), length, 0, length)
        out.SetDirectory(0)
        out.Sumw2()
        for i in range(length):
            out.SetBinContent(i + 1, hist.GetBinContent(start + i + 1))
            out.SetBinError(i + 1, hist.GetBinError(start + i + 1))
        return out

    def _slice_hist_bundle(self, bundle, start, length, name):
        return {
            "data": self._slice_hist(bundle["data"], start, length, f"h_data_{name}"),
            "mc": [(self._slice_hist(h, start, length, f"h_{label}_{name}"), label) for h, label in bundle["mc"]],
        }

    def _concat_hist_bundles(self, first, second, name):
        def concat_hist(h1, h2, hname):
            n1 = h1.GetNbinsX()
            n2 = h2.GetNbinsX()
            out = ROOT.TH1F(hname, h1.GetTitle(), n1 + n2, 0, n1 + n2)
            out.SetDirectory(0)
            out.Sumw2()
            for i in range(1, n1 + 1):
                out.SetBinContent(i, h1.GetBinContent(i))
                out.SetBinError(i, h1.GetBinError(i))
            for i in range(1, n2 + 1):
                out.SetBinContent(n1 + i, h2.GetBinContent(i))
                out.SetBinError(n1 + i, h2.GetBinError(i))
            return out

        second_by_label = {label: h for h, label in second["mc"]}
        mc = []
        for h1, label in first["mc"]:
            if label in second_by_label:
                mc.append((concat_hist(h1, second_by_label[label], f"h_{label}_{name}"), label))
            else:
                zero = ROOT.TH1F(f"h_zero_second_{label}_{name}", label, second["data"].GetNbinsX(), 0, second["data"].GetNbinsX())
                zero.SetDirectory(0)
                zero.Sumw2()
                mc.append((concat_hist(h1, zero, f"h_{label}_{name}"), label))
        for h2, label in second["mc"]:
            if not any(label == existing for _, existing in mc):
                zero = ROOT.TH1F(f"h_zero_{label}_{name}", label, first["data"].GetNbinsX(), 0, first["data"].GetNbinsX())
                zero.SetDirectory(0)
                zero.Sumw2()
                mc.append((concat_hist(zero, h2, f"h_{label}_{name}"), label))
        return {
            "data": concat_hist(first["data"], second["data"], f"h_data_{name}"),
            "mc": mc,
        }

    # ── Decoration builders ───────────────────────────────────────────────────

    def _build_decorations(self, bin_order, bin_scheme, label_map=None, bin_label_map=None):
        """
        Build decoration dict for a standard canvas.

        bin_scheme: 'risr' | 'msrs' | 'ms_delayed'

        Returns dict with keys:
            n_bins, bin_labels, group_labels, sub_group_labels,
            separator_bins, sub_sep_bins,
            section_labels, section_separator,
            x_axis_title, bottom_margin
        """
        bin_labels       = []
        group_labels     = []
        sub_group_labels = []
        separator_bins   = []
        sub_sep_bins     = []
        cursor           = 0

        for i, (ch, bins) in enumerate(bin_order):
            ch_start = cursor
            prev_ms  = None  # reset per channel for msrs

            if bin_scheme == "msrs":
                for bin_name in bins:
                    suf = bin_name[-2:]
                    ms, rs = suf[0], suf[1]
                    if ms != prev_ms:
                        if prev_ms is not None:
                            sub_group_labels[-1]["end"] = cursor
                            sub_sep_bins.append(cursor)
                        sub_group_labels.append(
                            {"text": MSRS_MS_LABELS.get(ms, ms), "start": cursor, "end": None}
                        )
                        prev_ms = ms
                    bin_labels.append(MSRS_RS_LABELS.get(rs, rs))
                    cursor += 1
                if sub_group_labels and sub_group_labels[-1]["end"] is None:
                    sub_group_labels[-1]["end"] = cursor

            elif bin_scheme == "risr":
                _bll = bin_label_map if bin_label_map is not None else RISR_LABELS
                has_bin30 = any(b[-2:] == "30" for b in bins)
                for bin_name in bins:
                    suf = bin_name[-2:]
                    if suf == "20" and not has_bin30:
                        # 3-bin channel: last RISR bin is >= 0.75, not [0.75, 0.9)
                        label = "R_{ISR}^{hi}" if bin_label_map is not None else "#geq 0.75"
                    else:
                        label = _bll.get(suf, suf)
                    bin_labels.append(label)
                    cursor += 1

            elif bin_scheme == "ms_delayed":
                _bll = bin_label_map if bin_label_map is not None else MS_DELAYED_LABELS
                for bin_name in bins:
                    bin_labels.append(_bll.get(bin_name[-2:], bin_name[-2:]))
                    cursor += 1

            elif bin_scheme == "abcd":
                _bll = bin_label_map if bin_label_map is not None else ABCD_COMPRESSED_BIN_LABELS
                for bin_name in bins:
                    bin_labels.append(_bll.get(bin_name[-2:], bin_name[-2:]))
                    cursor += 1

            else:
                for _ in bins:
                    bin_labels.append("")
                    cursor += 1

            _labels = label_map if label_map is not None else CHANNEL_LABELS
            group_labels.append({
                "text":  _labels.get(ch, CHANNEL_LABELS.get(ch, _make_channel_label(ch))),
                "start": ch_start,
                "end":   cursor,
            })
            if i < len(bin_order) - 1:
                separator_bins.append(cursor)

        has_sub       = bool(sub_group_labels)
        bottom_margin = 0.52 if has_sub else 0.44

        x_titles = {
            "risr":       "R_{ISR}",
            "ms_delayed": "M_{S} [TeV]",
            "abcd":       "R_{ISR}",
        }
        sg_titles = {}

        return {
            "n_bins":              cursor,
            "bin_labels":          bin_labels,
            "group_labels":        group_labels,
            "sub_group_labels":    sub_group_labels if has_sub else None,
            "sub_group_axis_title": sg_titles.get(bin_scheme, ""),
            "separator_bins":      separator_bins,
            "sub_sep_bins":        sub_sep_bins,
            "section_labels":      None,
            "section_separator":   None,
            "x_axis_title":        x_titles.get(bin_scheme, ""),
            "bottom_margin":       bottom_margin,
            "bin_scheme":          bin_scheme,
        }

    def _build_compressed_final_decorations(self, bin_order, sr_ch=None):
        """
        Build compact hierarchical labels for the final compressed fit.

        These labels are intentionally derived from Combine bin-name patterns and
        channel-aware RISR mappings.  In the final compressed AnalysisConfig,
        suffixes are not globally meaningful: for DelPho/MixDel, suffix 00 is
        the high RISR bin while suffix 10 is the medium bin.
        """
        bin_labels       = []
        group_labels     = []
        sub_group_labels = []
        separator_bins   = []
        sub_sep_bins     = []
        minor_sep_bins   = []
        sr_bins          = []
        cursor           = 0
        current_family   = None
        current_family_start = 0
        current_leaf     = None

        def close_leaf(end):
            if sub_group_labels and sub_group_labels[-1]["end"] is None:
                sub_group_labels[-1]["end"] = end

        def close_group(end):
            if group_labels and group_labels[-1]["end"] is None:
                group_labels[-1]["end"] = end

        for ch, bins in bin_order:
            display_ch = _compressed_display_key(ch)
            family = self._compressed_final_family(display_ch)
            top_label = COMPRESSED_FINAL_TOP_LABELS.get(family, _make_channel_label(display_ch))

            if family != current_family:
                close_leaf(cursor)
                close_group(cursor)
                if current_family is not None and cursor > 0:
                    separator_bins.append(cursor)
                group_labels.append({"text": top_label, "start": cursor, "end": None})
                current_family = family
                current_family_start = cursor
                current_leaf = None

            for bin_name in bins:
                display_bin = _compressed_display_key(bin_name)
                if family == "MixPrompt":
                    rel_bin = cursor - current_family_start
                    leaf = "#gamma_{p} CR" if rel_bin < 3 else "#gamma_{p}+SV SR"
                else:
                    leaf = self._compressed_final_leaf_label(display_ch, display_bin, family)
                if leaf != current_leaf:
                    close_leaf(cursor)
                    if current_leaf is not None and cursor > 0:
                        sub_sep_bins.append(cursor)
                    sub_group_labels.append({"text": leaf, "start": cursor, "end": None})
                    current_leaf = leaf

                suf = bin_name[-2:]
                risr_key = family
                if family == "MixDel" and not any(t in bin_name for t in ("BHEarly", "BHLate", "NotBH")):
                    risr_key = "MixDel_STF"
                bin_labels.append(
                    COMPRESSED_FINAL_RISR_LABELS.get(risr_key, {}).get(suf, RISR_COMPACT_LABELS.get(suf, suf))
                )
                if sr_ch and "SR" in bin_name:
                    sr_bins.append(cursor)
                cursor += 1
                if cursor > 1:
                    minor_sep_bins.append(cursor - 1)

        close_leaf(cursor)
        close_group(cursor)
        risr_sequence_labels = [
            {
                "text": "  ".join(bin_labels[sg["start"]:sg["end"]]),
                "start": sg["start"],
                "end": sg["end"],
            }
            for sg in sub_group_labels
        ]

        return {
            "n_bins":              cursor,
            "bin_labels":          bin_labels,
            "risr_sequence_labels": risr_sequence_labels,
            "group_labels":        group_labels,
            "sub_group_labels":    sub_group_labels or None,
            "sub_group_axis_title": "",
            "separator_bins":      separator_bins,
            "sub_sep_bins":        sub_sep_bins,
            "minor_sep_bins":      minor_sep_bins,
            "section_labels":      None,
            "section_separator":   None,
            "x_axis_title":        "R_{ISR}",
            "bottom_margin":       0.5,
            "bin_scheme":          "compressed_final",
            "group_label_text_size": 0.056,
            "sub_group_label_text_size": 0.066,
            "bin_label_text_size": 0.084,
            "x_title_text_size": 0.180,
            "sub_group_label_y":    0.42,
            "risr_sequence_y":      0.33,
            "x_title_y":            0.15,
            "glossary":            COMPRESSED_FINAL_GLOSSARY,
            "sr_bins":             sr_bins,
        }

    def _compressed_final_family(self, name):
        name = _compressed_display_key(name)
        if name.startswith("SVonly"):
            return "SVonly"
        if name.startswith("DelPho"):
            return "DelPho"
        if name.startswith("Eq2Pho"):
            return "Eq2Pho"
        if name.startswith(("MixDel", "SVDelPho")):
            return "MixDel"
        if name.startswith(("MixPrompt", "SVNoDelPho")):
            return "MixPrompt"
        return name.split("_", 1)[0]

    def _compressed_final_leaf_label(self, ch, bin_name, family):
        ch = _compressed_display_key(ch)
        bin_name = _compressed_display_key(bin_name)
        stem = bin_name[:-2]

        if family == "SVonly":
            if "AnchorCR" in stem:
                return "Anchor CR"
            if stem.endswith(("CR", "SR")):
                return "SV SR"

        if family in ("DelPho", "MixDel"):
            if "BHEarly" in stem and "NotBH" not in stem:
                return "BH-"
            if "BHLate" in stem and "NotBH" not in stem:
                return "BH+"
            if "NotBHEarly" in stem:
                return "!BH-"
            if "NotBHLate" in stem:
                return "!BH+"
            if "AnchorCR" in stem:
                return "Anchor CR"
            if stem.endswith("CR"):
                # shape_transfer buoy bin: has "CR" in name but is the SR-side bin
                return "#gamma_{d}+SV SR" if family == "MixDel" else "Delayed CR"
            if stem.endswith("SR"):
                return "#gamma_{d}+SV SR" if family == "MixDel" else "Delayed SR"

        if family == "Eq2Pho":
            if "MedIso" in stem:
                return "MedIso CR"
            if "TightIso" in stem:
                return "TightIso SR"

        if family == "MixPrompt":
            if "AnchorCR" in stem:
                return "Anch CR"
            if "SVSR" in stem or stem.endswith("SR"):
                return "#gamma_{p}+SV SR"
            if stem.endswith("CR"):
                return "#gamma_{p} CR"

        return CHANNEL_LABELS.get(ch, _make_channel_label(ch))

    def parse_noncompressed_bin(self, bin_name, fit_channel=None):
        suffix = _noncompressed_suffix(bin_name)
        channel = "unknown"
        subgroup = _make_channel_label(bin_name[:-2])
        grouping_name = fit_channel or bin_name

        if "GeLep" in bin_name:
            channel = "lep_sv"
            subgroup = "SR" if "SRGeLep" in grouping_name else "CR"
        elif "GeHad" in bin_name:
            channel = "had_sv"
            subgroup = "SR" if "SRGeHad" in grouping_name else "CR"
        elif any(token in bin_name for token in ("BHEarly", "BHLate", "NotBHEarly", "NotBHLate")):
            channel = "delayed_photon"
            if "NotBHEarly" in bin_name:
                subgroup = "!BH-"
            elif "NotBHLate" in bin_name:
                subgroup = "!BH+"
            elif "BHEarly" in bin_name:
                subgroup = "BH-"
            else:
                subgroup = "BH+"
        elif "eq1Pho" in bin_name:
            channel = "one_prompt"
            subgroup = "tight iso" if "TightIsoPrompt" in grouping_name else "med iso"
        elif "eq2Pho" in bin_name:
            channel = "two_prompt"
            subgroup = "tight iso" if "TightIsoPrompt" in grouping_name else "med iso"

        return {
            "channel": channel,
            "subgroup": subgroup,
            "suffix": suffix,
            "display_label": _noncompressed_mr_label(suffix, channel),
        }

    def _build_noncompressed_final_decorations(self, bin_order, sr_ch=None):
        bin_labels       = []
        group_labels     = []
        sub_group_labels = []
        separator_bins   = []
        sub_sep_bins     = []
        minor_sep_bins   = []
        sr_bins          = []
        cursor           = 0
        current_channel  = None
        current_subgroup = None

        def close_subgroup(end):
            if sub_group_labels and sub_group_labels[-1]["end"] is None:
                sub_group_labels[-1]["end"] = end

        def close_group(end):
            if group_labels and group_labels[-1]["end"] is None:
                group_labels[-1]["end"] = end

        for ch, bins in bin_order:
            for bin_name in bins:
                parsed = self.parse_noncompressed_bin(bin_name, fit_channel=ch)
                channel = parsed["channel"]
                subgroup = parsed["subgroup"]

                if channel != current_channel:
                    close_subgroup(cursor)
                    close_group(cursor)
                    if current_channel is not None and cursor > 0:
                        separator_bins.append(cursor)
                    group_labels.append({
                        "text": NONCOMPRESSED_FINAL_TOP_LABELS.get(channel, _make_channel_label(ch)),
                        "start": cursor,
                        "end": None,
                    })
                    current_channel = channel
                    current_subgroup = None

                if subgroup != current_subgroup:
                    close_subgroup(cursor)
                    if current_subgroup is not None and cursor > 0:
                        sub_sep_bins.append(cursor)
                    sub_group_labels.append({"text": subgroup, "start": cursor, "end": None})
                    current_subgroup = subgroup

                bin_labels.append(parsed["display_label"])
                if sr_ch and "SR" in bin_name:
                    sr_bins.append(cursor)
                cursor += 1
                if cursor > 1:
                    minor_sep_bins.append(cursor - 1)

        close_subgroup(cursor)
        close_group(cursor)

        return {
            "n_bins":              cursor,
            "bin_labels":          bin_labels,
            "group_labels":        group_labels,
            "sub_group_labels":    sub_group_labels or None,
            "sub_group_axis_title": "",
            "separator_bins":      separator_bins,
            "sub_sep_bins":        sub_sep_bins,
            "minor_sep_bins":      minor_sep_bins,
            "section_labels":      None,
            "section_separator":   None,
            "x_axis_title":        "",
            "bottom_margin":       0.5,
            "bin_scheme":          "noncompressed_final",
            "group_label_text_size": 0.056,
            "sub_group_label_text_size": 0.066,
            "bin_label_text_size": 0.062,
            "x_title_text_size": 0.160,
            "sub_group_label_y":    0.42,
            "glossary":            NONCOMPRESSED_FINAL_GLOSSARY,
            "glossary_text_size":   0.034,
            "glossary_y0":          0.70,
            "glossary_line_spacing": 0.05,
            "sr_bins":             sr_bins,
        }

    def _build_abcd_grid_decorations(self, abcd_bin_order, sr_ch=None, mode="uncompressed"):
        """
        3-level decoration for the ABCD 2×2 grid layout:
          Level 1 (group_labels):     BH γ  |  Non-BH γ
          Level 2 (sub_group_labels): Early | Late  (within each major group)
          Level 3 (bin_labels):       Ms-delayed ranges (uncompressed) or R_ISR (compressed)
        """
        bh_chs    = [(ch, b) for ch, b in abcd_bin_order if "notBH" not in ch and "BH" in ch]
        notbh_chs = [(ch, b) for ch, b in abcd_bin_order if "notBH" in ch]

        bin_label_table = MS_DELAYED_LABELS if mode == "uncompressed" else RISR_COMPACT_LABELS
        x_title = "M_{S} [TeV]" if mode == "uncompressed" else "R_{ISR}"

        bin_labels       = []
        group_labels     = []
        sub_group_labels = []
        separator_bins   = []
        sub_sep_bins     = []
        sr_bins          = []
        cursor           = 0

        for gi, (grp_text, channels) in enumerate(
            [("BH #gamma", bh_chs), ("Non-BH #gamma", notbh_chs)]
        ):
            grp_start = cursor
            for ci, (ch, bins) in enumerate(channels):
                ch_start = cursor
                timing   = "Early" if "Early" in ch else "Late"
                for bin_name in bins:
                    bin_labels.append(bin_label_table.get(bin_name[-2:], bin_name[-2:]))
                    if sr_ch and "SR" in bin_name:
                        sr_bins.append(cursor)
                    cursor += 1
                sub_group_labels.append({"text": timing, "start": ch_start, "end": cursor})
                if ci < len(channels) - 1:
                    sub_sep_bins.append(cursor)

            group_labels.append({"text": grp_text, "start": grp_start, "end": cursor})
            if gi < 1:
                separator_bins.append(cursor)

        return {
            "n_bins":           cursor,
            "bin_labels":       bin_labels,
            "group_labels":     group_labels,
            "sub_group_labels": sub_group_labels,
            "separator_bins":   separator_bins,
            "sub_sep_bins":     sub_sep_bins,
            "section_labels":      None,
            "section_separator":   None,
            "x_axis_title":        x_title,
            "sub_group_axis_title": "",
            "bottom_margin":       0.52,
            "sr_bins":             sr_bins,
        }

    def _build_combined_decorations(self, abcd_bin_order, shape_bin_order, sr_ch=None,
                                    shape_bin_scheme="msrs", abcd_bin_scheme="ms_delayed",
                                    label_scheme="auto"):
        """
        Combined ABCD + shape_transfer decorations with section labels and
        a heavy separator between the two fit components.
        ABCD bins come first, shape_transfer second.
        """
        if label_scheme == "compressed-final":
            return self._build_compressed_final_decorations(abcd_bin_order + shape_bin_order, sr_ch=sr_ch)
        if label_scheme == "noncompressed-final":
            return self._build_noncompressed_final_decorations(abcd_bin_order + shape_bin_order, sr_ch=sr_ch)

        abcd_bl_map  = COMBINED_MS_DELAYED_LABELS if abcd_bin_scheme == "ms_delayed" else None
        shape_bl_map = RISR_COMPACT_LABELS if shape_bin_scheme == "risr" else None
        abcd_deco  = self._build_decorations(abcd_bin_order,  abcd_bin_scheme,  COMBINED_CHANNEL_LABELS, abcd_bl_map)
        shape_deco = self._build_decorations(shape_bin_order, shape_bin_scheme, COMBINED_CHANNEL_LABELS, shape_bl_map)
        n_abcd     = abcd_deco["n_bins"]
        # For single-bin ABCD, suppress per-bin labels and draw one centred annotation instead
        abcd_single = (abcd_bin_scheme == "abcd")
        abcd_bin_labels = [""] * n_abcd if abcd_single else abcd_deco["bin_labels"]
        abcd_annotation = {"text": "R_{ISR}^{hi}", "start": 0, "end": n_abcd} if abcd_single else None
        n_total    = n_abcd + shape_deco["n_bins"]

        def _offset(lst, off):
            return [{"text": g["text"], "start": g["start"] + off, "end": g["end"] + off}
                    for g in lst] if lst else []

        sr_bins = []
        if sr_ch:
            for gl, (ch, bins) in zip(abcd_deco["group_labels"], abcd_bin_order):
                if any("SR" in b for b in bins):
                    sr_bins.extend(range(gl["start"], gl["end"]))

        if shape_bin_scheme == "risr":
            glossary = [
                ("R_{ISR}^{lo}",  "[0.4, 0.6)"),
                ("R_{ISR}^{med}", "[0.6, 0.75)"),
                ("R_{ISR}^{hi}",  "#geq 0.75"),
                ("R_{ISR}^{hi-}", "[0.75, 0.9)"),
                ("R_{ISR}^{hi+}", "#geq 0.9"),
            ]
        else:
            glossary = [
                ("M_{S}^{CR,-}", "[0.7, 1) TeV"),
                ("M_{S}^{CR,+}", "#geq 1 TeV"),
                ("M_{S}^{SR,-}", "[1, 1.5) TeV"),
                ("M_{S}^{SR,+}", "#geq 1.5 TeV"),
                ("R_{S}^{-}",    "[0.15, 0.2)"),
                ("R_{S}^{+}",    "#geq 0.2"),
            ]

        return {
            "n_bins":      n_total,
            "bin_labels":  abcd_bin_labels + shape_deco["bin_labels"],
            "group_labels": (abcd_deco["group_labels"]
                             + _offset(shape_deco["group_labels"], n_abcd)),
            "sub_group_labels": (_offset(shape_deco["sub_group_labels"] or [], n_abcd) or None),
            "sub_group_axis_title": "",
            "separator_bins": (abcd_deco["separator_bins"]
                               + [s + n_abcd for s in shape_deco["separator_bins"]]),
            "sub_sep_bins": [s + n_abcd for s in shape_deco["sub_sep_bins"]],
            "section_labels":    None,
            "section_separator": n_abcd,
            "x_axis_title":  "",
            "bottom_margin": 0.52,
            "sr_bins":       sr_bins,
            "glossary":      glossary,
            "abcd_annotation": abcd_annotation,
        }

    # ── Canvas drawing: standard ──────────────────────────────────────────────

    def _draw_datamc_canvas(self, hist_bundle, deco, name, title, right_panel=False):
        n      = deco["n_bins"]
        bot_m  = deco.get("bottom_margin", 0.38)
        left_m = 0.10
        right_m = 0.16 if right_panel else 0.04
        split   = 0.30
        has_sub  = bool(deco.get("sub_group_labels"))
        has_sect = bool(deco.get("section_labels"))

        cw = max(1200, min(80 * n, 2400))
        self.datamc_plotter._ensure_mc_colors()
        canvas = ROOT.TCanvas(f"c_{name}", title, cw, 700)
        canvas.SetFillColor(0)

        pad1 = ROOT.TPad(f"p1_{name}", "main",  0, split, 1, 1)
        pad1.SetBottomMargin(0.02)
        pad1.SetTopMargin(0.17 if has_sect else 0.13)
        pad1.SetLeftMargin(left_m); pad1.SetRightMargin(right_m)
        pad1.SetTicks(1, 1)
        pad1.SetLogy(True); pad1.SetGridx(True); pad1.Draw()

        pad2 = ROOT.TPad(f"p2_{name}", "ratio", 0, 0, 1, split)
        pad2.SetTopMargin(0.02); pad2.SetBottomMargin(bot_m)
        pad2.SetLeftMargin(left_m); pad2.SetRightMargin(right_m)
        pad2.SetTicks(1, 1)
        pad2.SetGridx(True); pad2.SetGridy(True); pad2.Draw()

        pad1.cd()
        data_hist = hist_bundle["data"].Clone(f"h_data_{name}")
        data_hist.SetDirectory(0)
        data_hist.SetMarkerStyle(20)
        data_hist.SetMarkerSize(1.2)
        data_hist.SetMarkerColor(ROOT.kBlack)
        data_hist.SetLineColor(ROOT.kBlack)
        data_hist.SetLineWidth(2)
        data_hist.SetStats(0)

        mc_histograms = []
        for hist, label in hist_bundle["mc"]:
            h = hist.Clone(f"{hist.GetName()}_{name}")
            h.SetDirectory(0)
            datamc_label = _datamc_background_key(label)
            h.SetFillColor(self.datamc_plotter._get_background_color_index(datamc_label))
            h.SetLineColor(ROOT.kBlack)
            h.SetLineWidth(1)
            h.SetStats(0)
            mc_histograms.append((h, self.datamc_plotter._clean_mc_label(datamc_label)))
        mc_histograms.sort(key=lambda item: item[0].Integral())

        total_mc_hist = data_hist.Clone(f"total_mc_{name}")
        total_mc_hist.Reset()
        for h, _ in mc_histograms:
            total_mc_hist.Add(h)

        h_ratio = ROOT.TH1F(f"hr_{name}", "", n, 0, n)
        h_ratio.SetDirectory(0)
        h_rband = ROOT.TH1F(f"hrb_{name}", "", n, 0, n)
        h_rband.SetDirectory(0)
        for i in range(n):
            data = data_hist.GetBinContent(i + 1)
            data_err = data_hist.GetBinError(i + 1)
            bkg = total_mc_hist.GetBinContent(i + 1)
            bkg_err = total_mc_hist.GetBinError(i + 1)
            h_ratio.SetBinContent(i + 1, data / bkg if bkg > 0 else 0)
            h_ratio.SetBinError(i + 1, data_err / bkg if bkg > 0 else 0)
            h_rband.SetBinContent(i + 1, 1.0)
            h_rband.SetBinError(i + 1, bkg_err / bkg if bkg > 0 else 0)
        h_rband.SetFillColor(ROOT.kGray + 1)
        h_rband.SetFillStyle(3345)
        h_rband.SetMarkerSize(0)
        h_rband.SetLineColor(0)

        stack = ROOT.THStack(f"stack_{name}", "")
        for h, _ in mc_histograms:
            stack.Add(h)

        stack.Draw("HIST")
        pos_vals = []
        for i in range(n):
            bkg = total_mc_hist.GetBinContent(i + 1)
            bkg_err = total_mc_hist.GetBinError(i + 1)
            data = data_hist.GetBinContent(i + 1)
            data_err = data_hist.GetBinError(i + 1)
            if bkg > 0:
                pos_vals.append(bkg)
            if data > 0:
                pos_vals.append(data)
        pos_vals = np.array(pos_vals)
        min_v = max(0.5, 0.3 * pos_vals.min()) if pos_vals.size else 0.5
        max_v = max(
            max((total_mc_hist.GetBinContent(i + 1) + total_mc_hist.GetBinError(i + 1)) for i in range(n)),
            max((data_hist.GetBinContent(i + 1) + data_hist.GetBinError(i + 1)) for i in range(n)),
            1.0,
        )
        stack.SetMinimum(min_v)
        stack.SetMaximum(max_v * 10.0)
        stack.GetHistogram().GetYaxis().SetRangeUser(min_v, max_v * 10.0)
        stack.GetXaxis().SetLabelSize(0)
        stack.GetXaxis().SetTickLength(0.015)
        stack.GetXaxis().SetNdivisions(n, 0, 0, False)
        stack.GetYaxis().SetTitle("Events / bin")
        stack.GetYaxis().SetTitleSize(0.075)
        stack.GetYaxis().SetTitleOffset(0.65)
        stack.GetYaxis().SetLabelSize(0.065)
        stack.GetYaxis().SetTickLength(0.015)
        stack.GetYaxis().CenterTitle(True)

        mc_uncertainty = self.datamc_plotter._create_mc_uncertainty_band(mc_histograms)
        if mc_uncertainty:
            mc_uncertainty.Draw("E2 SAME")
        data_hist.Draw("PEX0 SAME")

        rp = 1.0 - right_m
        if right_panel:
            legend = self.datamc_plotter._create_standard_legend(
                data_hist, mc_uncertainty, mc_histograms,
                x1=rp + 0.01, x2=0.995, y1=0.62, y2=0.88, text_size=0.038
            )
        else:
            legend = self.datamc_plotter._create_standard_legend(
                data_hist, mc_uncertainty, mc_histograms,
                x1=0.77, x2=1.0, y1=0.36, y2=0.76, text_size=0.045
            )
        legend.Draw()

        rp_objs = []
        if right_panel:
            defs = deco.get("glossary", [])
            if defs:
                lt_key = ROOT.TLatex()
                lt_key.SetNDC(True)
                lt_key.SetTextFont(42)
                lt_key.SetTextSize(deco.get("glossary_text_size", 0.034 if len(defs) > 10 else 0.052))
                lt_key.SetTextAlign(12)
                y_key = deco.get("datamc_glossary_y0", 0.58 if len(defs) > 4 else 0.52)
                dy = deco.get(
                    "datamc_glossary_line_spacing",
                    0.035 if len(defs) > 4 else deco.get("glossary_line_spacing", 0.075),
                )
                for sym, rng in defs:
                    if isinstance(rng, tuple):
                        lt_key.DrawLatex(rp + 0.01, y_key, f"{sym}  :")
                        for line in rng:
                            y_key -= dy
                            lt_key.DrawLatex(rp + 0.022, y_key, line)
                        y_key -= dy
                    else:
                        lt_key.DrawLatex(rp + 0.01, y_key, f"{sym}  :  {rng}")
                        y_key -= dy
                rp_objs.append(lt_key)

        if right_panel:
            self.style.draw_cms_labels(
                cms_x=0.10, cms_y=0.89, prelim_str="Preliminary",
                prelim_x=0.175, lumi_x=0.84, cms_text_size_mult=1.92
            )
        else:
            self.style.draw_cms_labels(
                cms_x=0.10, cms_y=0.89, prelim_str="Preliminary",
                prelim_x=0.185, lumi_x=0.96, cms_text_size_mult=1.92
            )

        sect_objs = []
        if has_sect:
            dw  = 1.0 - left_m - right_m
            slt = ROOT.TLatex()
            slt.SetNDC(True)
            slt.SetTextFont(62)
            slt.SetTextSize(0.065)
            slt.SetTextAlign(22)
            for sl in deco["section_labels"]:
                cx    = (sl["start"] + sl["end"]) / 2.0
                x_ndc = left_m + (cx / n) * dw
                slt.DrawLatex(x_ndc, 0.88, sl["text"])
            sect_objs.append(slt)

        grp_y    = 0.72 if has_sect else 0.77
        grp_objs = self._draw_group_labels(pad1, deco, left_m, right_m, grp_y)

        pad2.cd()
        h_ratio.SetMaximum(1.99)
        h_ratio.SetMinimum(0.0)
        h_ratio.GetYaxis().SetTitle("")
        h_ratio.GetYaxis().SetLabelSize(0.15)
        h_ratio.GetYaxis().SetNdivisions(504)
        h_ratio.GetYaxis().SetTickLength(0.015)
        h_ratio.GetXaxis().SetLabelSize(0)
        h_ratio.GetXaxis().SetTickLength(0.015)
        h_ratio.GetXaxis().SetNdivisions(n, 0, 0, False)
        h_ratio.SetMarkerStyle(20)
        h_ratio.SetMarkerSize(1.0)
        h_ratio.SetLineColor(ROOT.kBlack)
        h_ratio.SetStats(0)
        h_ratio.Draw("PE")
        h_rband.Draw("E2 SAME")
        h_ratio.Draw("PE SAME")

        unity = ROOT.TLine(0, 1, n, 1)
        unity.SetLineColor(ROOT.kRed)
        unity.SetLineWidth(2)
        unity.SetLineStyle(2)
        unity.Draw()

        rtitle = ROOT.TLatex()
        rtitle.SetNDC(True)
        rtitle.SetTextFont(42)
        rtitle.SetTextSize(0.16)
        rtitle.SetTextAlign(22)
        rtitle.SetTextAngle(90)
        rtitle.DrawLatex(0.025, 0.65, "Data / MC")

        sg_objs = self._draw_sub_group_labels(pad2, deco, left_m, right_m) if has_sub else []
        abcd_y_off = 0.07 if deco.get("section_separator") is not None else 0.0
        bl_objs = self._draw_bin_labels(pad2, deco, left_m, right_m, has_sub, abcd_y_off)
        sep_objs = self._draw_separators(canvas, deco, left_m, right_m, name)

        canvas.Modified()
        canvas.Update()
        canvas.pad1 = pad1
        canvas.pad2 = pad2
        canvas.stack = stack
        canvas.data_hist = data_hist
        canvas.mc_histograms = mc_histograms
        canvas.mc_uncertainty = mc_uncertainty
        canvas.legend = legend
        canvas.ratio_hist = h_ratio
        canvas.total_mc_hist = total_mc_hist
        canvas.mc_ratio_uncertainty = h_rband
        canvas.line = unity
        canvas._keep = [
            pad1, pad2, stack, data_hist, mc_histograms, total_mc_hist,
            mc_uncertainty, legend, h_ratio, h_rband, unity, rtitle,
            grp_objs, sg_objs, bl_objs, sep_objs, sect_objs, rp_objs,
        ]
        return canvas

    def _draw_standard_canvas(self, bkg_vals, bkg_errs, data_y, data_eyl, data_eyh,
                               deco, name, title, right_panel=False):
        n      = deco["n_bins"]
        bot_m  = deco.get("bottom_margin", 0.38)
        left_m = 0.10
        right_m = 0.16 if right_panel else 0.04
        split   = 0.30
        has_sub  = bool(deco.get("sub_group_labels"))
        has_sect = bool(deco.get("section_labels"))

        # Scale canvas width with bin count
        cw = max(1200, min(80 * n, 2400))
        canvas = ROOT.TCanvas(f"c_{name}", title, cw, 700)
        canvas.SetFillColor(0)

        pad1 = ROOT.TPad(f"p1_{name}", "main",  0, split, 1, 1)
        pad1.SetBottomMargin(0.02)
        pad1.SetTopMargin(0.17 if has_sect else 0.13)
        pad1.SetLeftMargin(left_m); pad1.SetRightMargin(right_m)
        pad1.SetTicks(1, 1)
        pad1.SetLogy(True); pad1.SetGridx(True); pad1.Draw()

        pad2 = ROOT.TPad(f"p2_{name}", "ratio", 0, 0, 1, split)
        pad2.SetTopMargin(0.02); pad2.SetBottomMargin(bot_m)
        pad2.SetLeftMargin(left_m); pad2.SetRightMargin(right_m)
        pad2.SetTicks(1, 1)
        pad2.SetGridx(True); pad2.SetGridy(True); pad2.Draw()

        # ── histograms ────────────────────────────────────────────────────────
        h_bkg = ROOT.TH1F(f"hb_{name}", "", n, 0, n)
        h_bkg.SetDirectory(0); h_bkg.Sumw2()
        for i in range(n):
            h_bkg.SetBinContent(i + 1, bkg_vals[i])
            h_bkg.SetBinError(i + 1,   bkg_errs[i])
        h_bkg.SetFillColor(ROOT.kAzure - 9)
        h_bkg.SetLineColor(ROOT.kBlue + 1); h_bkg.SetLineWidth(2); h_bkg.SetStats(0)

        h_bkg_err = h_bkg.Clone(f"hberr_{name}")
        h_bkg_err.SetDirectory(0)
        h_bkg_err.SetFillColor(ROOT.kGray + 1)
        h_bkg_err.SetFillStyle(3345)
        h_bkg_err.SetMarkerSize(0)
        h_bkg_err.SetLineColor(0)

        g_data = ROOT.TGraphAsymmErrors(n)
        for i in range(n):
            g_data.SetPoint(i, i + 0.5, data_y[i])
            g_data.SetPointError(i, 0, 0, data_eyl[i], data_eyh[i])
        g_data.SetMarkerStyle(20); g_data.SetMarkerSize(1.2)
        g_data.SetMarkerColor(ROOT.kBlack); g_data.SetLineColor(ROOT.kBlack); g_data.SetLineWidth(2)

        h_ratio = ROOT.TH1F(f"hr_{name}", "", n, 0, n)
        h_ratio.SetDirectory(0)
        h_rband = ROOT.TH1F(f"hrb_{name}", "", n, 0, n)
        h_rband.SetDirectory(0)
        for i in range(n):
            b = bkg_vals[i]
            h_ratio.SetBinContent(i + 1, data_y[i] / b if b > 0 else 0)
            h_ratio.SetBinError(i + 1,   max(data_eyl[i], data_eyh[i]) / b if b > 0 else 0)
            h_rband.SetBinContent(i + 1, 1.0)
            h_rband.SetBinError(i + 1,   bkg_errs[i] / b if b > 0 else 0)
        h_rband.SetFillColor(ROOT.kGray + 1); h_rband.SetFillStyle(3345)
        h_rband.SetMarkerSize(0); h_rband.SetLineColor(0)

        # ── main pad ──────────────────────────────────────────────────────────
        pad1.cd()
        pos_vals = np.concatenate([bkg_vals, data_y])
        pos_vals = pos_vals[pos_vals > 0]
        min_v = max(0.5, 0.3 * pos_vals.min()) if pos_vals.size else 0.5
        max_v = max(float((bkg_vals + bkg_errs).max()),
                    float((data_y   + data_eyh).max()))
        h_bkg.SetMinimum(min_v); h_bkg.SetMaximum(max_v * 10.0)

        h_bkg.GetYaxis().SetTitle("Events / bin")
        h_bkg.GetYaxis().SetTitleSize(0.075); h_bkg.GetYaxis().SetTitleOffset(0.65)
        h_bkg.GetYaxis().SetLabelSize(0.065); h_bkg.GetYaxis().CenterTitle(True)
        h_bkg.GetYaxis().SetTickLength(0.015)
        h_bkg.GetXaxis().SetLabelSize(0);     h_bkg.GetXaxis().SetTickLength(0.015)
        h_bkg.GetXaxis().SetNdivisions(n, 0, 0, False)
        sr_bins = deco.get("sr_bins", [])
        h_sr = None
        if sr_bins:
            h_sr = ROOT.TH1F(f"hsr_{name}", "", n, 0, n)
            h_sr.SetDirectory(0)
            for i in sr_bins:
                h_sr.SetBinContent(i + 1, bkg_vals[i])
                h_sr.SetBinError(i + 1,   bkg_errs[i])
            h_sr.SetFillColor(ROOT.kOrange - 3)
            h_sr.SetLineColor(ROOT.kBlue + 1); h_sr.SetLineWidth(2); h_sr.SetStats(0)

        h_bkg.Draw("HIST")
        if h_sr:
            h_sr.Draw("HIST SAME")
        h_bkg_err.Draw("E2 SAME")
        g_data.Draw("PZ SAME")

        rp = 1.0 - right_m  # left edge of right-margin panel in pad1 NDC
        if right_panel and deco.get("bin_scheme") == "noncompressed_final":
            leg = ROOT.TLegend(rp + 0.01, 0.75, 0.995, 0.87)
        elif right_panel:
            leg = ROOT.TLegend(rp + 0.01, 0.72, 0.995, 0.87)
        else:
            leg = ROOT.TLegend(0.77, 0.42, 1., 0.7)
        leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.055)
        leg.AddEntry(g_data,   "Data",         "lp")
        leg.AddEntry(h_bkg,    f"{title} bkg", "F")
        if h_sr:
            leg.AddEntry(h_sr, "Pred. SR",     "F")
        leg.Draw()

        rp_objs = []
        if right_panel:
            defs = deco.get("glossary", [
                ("M_{S}^{CR,-}", "[0.7, 1) TeV"),
                ("M_{S}^{CR,+}", "#geq 1 TeV"),
                ("M_{S}^{SR,-}", "[1, 1.5) TeV"),
                ("M_{S}^{SR,+}", "#geq 1.5 TeV"),
                ("R_{S}^{-}",    "[0.15, 0.2)"),
                ("R_{S}^{+}",    "#geq 0.2"),
            ])
            lt_key = ROOT.TLatex(); lt_key.SetNDC(True)
            lt_key.SetTextFont(42)
            lt_key.SetTextSize(deco.get("glossary_text_size", 0.034 if len(defs) > 10 else 0.052))
            lt_key.SetTextAlign(12)
            y0 = deco.get("glossary_y0", 0.6)
            dy = deco.get("glossary_line_spacing", 0.040 if len(defs) > 10 else 0.075)
            y_key = y0
            for sym, rng in defs:
                if isinstance(rng, tuple):
                    lt_key.DrawLatex(rp + 0.01, y_key, f"{sym}  :")
                    for line in rng:
                        y_key -= dy
                        lt_key.DrawLatex(rp + 0.022, y_key, line)
                    y_key -= dy
                else:
                    lt_key.DrawLatex(rp + 0.01, y_key, f"{sym}  :  {rng}")
                    y_key -= dy
            rp_objs.append(lt_key)

        if right_panel:
            self.style.draw_cms_labels(
                cms_x=0.10, cms_y=0.89, prelim_str="Preliminary",
                prelim_x=0.175, lumi_x=0.84, cms_text_size_mult=1.92
            )
        else:
            self.style.draw_cms_labels(
                cms_x=0.10, cms_y=0.89, prelim_str="Preliminary",
                prelim_x=0.185, lumi_x=0.96, cms_text_size_mult=1.92
            )

        # Section labels (combined plot only)
        sect_objs = []
        if has_sect:
            dw  = 1.0 - left_m - right_m
            slt = ROOT.TLatex(); slt.SetNDC(True)
            slt.SetTextFont(62); slt.SetTextSize(0.065); slt.SetTextAlign(22)
            for sl in deco["section_labels"]:
                cx    = (sl["start"] + sl["end"]) / 2.0
                x_ndc = left_m + (cx / n) * dw
                slt.DrawLatex(x_ndc, 0.88, sl["text"])
            sect_objs.append(slt)

        grp_y    = 0.72 if has_sect else 0.77
        grp_objs = self._draw_group_labels(pad1, deco, left_m, right_m, grp_y)

        # ── ratio pad ─────────────────────────────────────────────────────────
        pad2.cd()
        h_ratio.SetMaximum(1.99); h_ratio.SetMinimum(0.0)
        h_ratio.GetYaxis().SetTitle("")
        h_ratio.GetYaxis().SetLabelSize(0.15); h_ratio.GetYaxis().SetNdivisions(504)
        h_ratio.GetYaxis().SetTickLength(0.015)
        h_ratio.GetXaxis().SetLabelSize(0); h_ratio.GetXaxis().SetTickLength(0.015)
        h_ratio.GetXaxis().SetNdivisions(n, 0, 0, False)
        h_ratio.SetMarkerStyle(20); h_ratio.SetMarkerSize(1.0)
        h_ratio.SetLineColor(ROOT.kBlack); h_ratio.SetStats(0)
        h_ratio.Draw("PE"); h_rband.Draw("E2 SAME"); h_ratio.Draw("PE SAME")

        unity = ROOT.TLine(0, 1, n, 1)
        unity.SetLineColor(ROOT.kRed); unity.SetLineWidth(2); unity.SetLineStyle(2)
        unity.Draw()

        rtitle = ROOT.TLatex(); rtitle.SetNDC(True)
        rtitle.SetTextFont(42); rtitle.SetTextSize(0.16)
        rtitle.SetTextAlign(22); rtitle.SetTextAngle(90)
        rtitle.DrawLatex(0.025, 0.65, "Data / Fit")

        sg_objs = self._draw_sub_group_labels(pad2, deco, left_m, right_m) if has_sub else []
        abcd_y_off = 0.07 if deco.get("section_separator") is not None else 0.0
        bl_objs = self._draw_bin_labels(pad2, deco, left_m, right_m, has_sub, abcd_y_off)

        sep_objs = self._draw_separators(canvas, deco, left_m, right_m, name)

        canvas.Modified(); canvas.Update()
        canvas._keep = [h_bkg, h_sr, h_bkg_err, g_data, h_ratio, h_rband, unity, leg,
                        rtitle, grp_objs, sg_objs, bl_objs, sep_objs, sect_objs, rp_objs]
        return canvas

    # ── Canvas drawing: side-by-side ──────────────────────────────────────────

    def _draw_sidebyside_canvas(self,
                                 a_bkg, a_berr, a_data, a_eyl, a_eyh,
                                 b_bkg, b_berr, b_data, b_eyl, b_eyh,
                                 a_deco, b_deco, name, title):
        split    = 0.30
        mid      = 0.50
        lm_l, lm_r = 0.14, 0.08
        rm       = 0.04

        canvas = ROOT.TCanvas(f"c_{name}", title, 1400, 700)
        canvas.SetFillColor(0)

        pML = ROOT.TPad(f"pML_{name}", "", 0,   split, mid, 1)
        pMR = ROOT.TPad(f"pMR_{name}", "", mid, split, 1,   1)
        pRL = ROOT.TPad(f"pRL_{name}", "", 0,   0,     mid, split)
        pRR = ROOT.TPad(f"pRR_{name}", "", mid, 0,     1,   split)

        bot_l = a_deco.get("bottom_margin", 0.38)
        bot_r = b_deco.get("bottom_margin", 0.38)

        for pad, lm, bm, log in [
            (pML, lm_l, 0.02, True),  (pMR, lm_r, 0.02, True),
            (pRL, lm_l, bot_l, False), (pRR, lm_r, bot_r, False),
        ]:
            pad.SetLeftMargin(lm); pad.SetRightMargin(rm)
            pad.SetTopMargin(0.13 if log else 0.02); pad.SetBottomMargin(bm)
            pad.SetTicks(1, 1)
            if log:
                pad.SetLogy(True); pad.SetGridx(True)
            else:
                pad.SetGridx(True); pad.SetGridy(True)
            pad.Draw()

        keep = []

        for side, (bkg, berr, data, eyl, eyh, deco, pm, pr, lm, is_r) in enumerate([
            (a_bkg, a_berr, a_data, a_eyl, a_eyh, a_deco, pML, pRL, lm_l, False),
            (b_bkg, b_berr, b_data, b_eyl, b_eyh, b_deco, pMR, pRR, lm_r, True),
        ]):
            nb = deco["n_bins"]
            has_sub = bool(deco.get("sub_group_labels"))

            h_bkg = ROOT.TH1F(f"hb_{name}_{side}", "", nb, 0, nb)
            h_bkg.SetDirectory(0); h_bkg.Sumw2()
            for i in range(nb):
                h_bkg.SetBinContent(i + 1, bkg[i]); h_bkg.SetBinError(i + 1, berr[i])
            h_bkg.SetFillColor(ROOT.kAzure - 9)
            h_bkg.SetLineColor(ROOT.kBlue + 1); h_bkg.SetLineWidth(2); h_bkg.SetStats(0)

            h_bkg_err = h_bkg.Clone(f"hberr_{name}_{side}")
            h_bkg_err.SetDirectory(0)
            h_bkg_err.SetFillColor(ROOT.kGray + 1)
            h_bkg_err.SetFillStyle(1001)
            h_bkg_err.SetLineColor(ROOT.kGray + 2)
            h_bkg_err.SetLineWidth(1)

            g_data = ROOT.TGraphAsymmErrors(nb)
            for i in range(nb):
                g_data.SetPoint(i, i + 0.5, data[i])
                g_data.SetPointError(i, 0, 0, eyl[i], eyh[i])
            g_data.SetMarkerStyle(20); g_data.SetMarkerSize(1.2)
            g_data.SetMarkerColor(ROOT.kBlack); g_data.SetLineColor(ROOT.kBlack)
            g_data.SetLineWidth(2)

            h_ratio = ROOT.TH1F(f"hr_{name}_{side}", "", nb, 0, nb)
            h_rband = ROOT.TH1F(f"hrb_{name}_{side}", "", nb, 0, nb)
            h_ratio.SetDirectory(0); h_rband.SetDirectory(0)
            for i in range(nb):
                b = bkg[i]
                h_ratio.SetBinContent(i + 1, data[i] / b if b > 0 else 0)
                h_ratio.SetBinError(i + 1,   max(eyl[i], eyh[i]) / b if b > 0 else 0)
                h_rband.SetBinContent(i + 1, 1.0)
                h_rband.SetBinError(i + 1,   berr[i] / b if b > 0 else 0)
            h_rband.SetFillColor(ROOT.kGray + 1); h_rband.SetFillStyle(3345)
            h_rband.SetMarkerSize(0); h_rband.SetLineColor(0)

            # Main pad
            pm.cd()
            pv = np.concatenate([bkg, data]); pv = pv[pv > 0]
            min_v = max(0.5, 0.3 * pv.min()) if pv.size else 0.5
            max_v = max(float((bkg + berr).max()), float((data + eyh).max()))
            h_bkg.SetMinimum(min_v); h_bkg.SetMaximum(max_v * 10.0)
            h_bkg.GetYaxis().SetLabelSize(0.08)
            if not is_r:
                h_bkg.GetYaxis().SetTitle("Events / bin")
                h_bkg.GetYaxis().SetTitleSize(0.09)
                h_bkg.GetYaxis().SetTitleOffset(0.55)
                h_bkg.GetYaxis().CenterTitle(True)
            h_bkg.GetYaxis().SetTickLength(0.015)
            h_bkg.GetXaxis().SetLabelSize(0); h_bkg.GetXaxis().SetTickLength(0.015)
            h_bkg.GetXaxis().SetNdivisions(nb, 0, 0, False)
            h_bkg.Draw("HIST")
            h_bkg_err.Draw("E2 SAME")
            g_data.Draw("PZ SAME")

            if not is_r:
                leg = ROOT.TLegend(0.57, 0.47, 0.98, 0.74)
                leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.07)
                leg.AddEntry(g_data,    "Data",         "lp")
                leg.AddEntry(h_bkg,     f"{title} bkg", "F")
                leg.Draw(); keep.append(leg)

            grp = self._draw_group_labels(pm, deco, lm, rm, 0.83)

            # Ratio pad
            pr.cd()
            h_ratio.SetMaximum(1.99); h_ratio.SetMinimum(0.0)
            h_ratio.GetYaxis().SetTitle("")
            h_ratio.GetYaxis().SetLabelSize(0.17); h_ratio.GetYaxis().SetNdivisions(504)
            h_ratio.GetYaxis().SetTickLength(0.015)
            h_ratio.GetXaxis().SetLabelSize(0); h_ratio.GetXaxis().SetTickLength(0.015)
            h_ratio.GetXaxis().SetNdivisions(nb, 0, 0, False)
            h_ratio.SetMarkerStyle(20); h_ratio.SetMarkerSize(1.0)
            h_ratio.SetLineColor(ROOT.kBlack); h_ratio.SetStats(0)
            h_ratio.Draw("PE"); h_rband.Draw("E2 SAME"); h_ratio.Draw("PE SAME")

            unity = ROOT.TLine(0, 1, nb, 1)
            unity.SetLineColor(ROOT.kRed); unity.SetLineWidth(2); unity.SetLineStyle(2)
            unity.Draw()

            if not is_r:
                rtitle = ROOT.TLatex(); rtitle.SetNDC(True)
                rtitle.SetTextFont(42); rtitle.SetTextSize(0.19)
                rtitle.SetTextAlign(22); rtitle.SetTextAngle(90)
                rtitle.DrawLatex(0.048, 0.50, "Data / Fit")
                keep.append(rtitle)

            sg = self._draw_sub_group_labels(pr, deco, lm, rm) if has_sub else []
            bl = self._draw_bin_labels(pr, deco, lm, rm, has_sub)

            # Local separators within each side-by-side panel.
            sub_lines = []
            dw = 1.0 - lm - rm
            for key, color, width, style in [
                ("minor_sep_bins", ROOT.kGray + 1, 1, 3),
                ("sub_sep_bins",   ROOT.kGray + 2, 2, 2),
                ("separator_bins", ROOT.kBlack if deco.get("bin_scheme") in ("compressed_final", "noncompressed_final") else ROOT.kGray + 2,
                 3 if deco.get("bin_scheme") in ("compressed_final", "noncompressed_final") else 2,
                 1 if deco.get("bin_scheme") in ("compressed_final", "noncompressed_final") else 2),
            ]:
                for sb in deco.get(key, []):
                    x = lm + (sb / nb) * dw
                    for pad in [pm, pr]:
                        pad.cd()
                        sl = ROOT.TLine(); sl.SetNDC(True)
                        sl.SetLineColor(color)
                        sl.SetLineWidth(width); sl.SetLineStyle(style)
                        y_low = 0.52 if deco.get("bin_scheme") in ("compressed_final", "noncompressed_final") and pad == pr else 0.04
                        sl.DrawLine(x, y_low, x, 0.95)
                        sub_lines.append(sl)

            keep.extend([h_bkg, h_bkg_err, g_data, h_ratio, h_rband, unity, grp, sg, bl, sub_lines])

        # CMS labels on full-canvas overlay
        canvas.cd()
        ov = ROOT.TPad(f"ov_{name}", "", 0, split, 1, 1)
        ov.SetFillStyle(0); ov.SetFrameFillStyle(0)
        ov.SetBorderSize(0); ov.SetBorderMode(0); ov.SetMargin(0, 0, 0, 0)
        ov.SetBit(ROOT.kCannotPick); ov.Draw(); ov.cd()
        self.style.draw_cms_labels(
            cms_x=0.05, cms_y=0.93, prelim_str="Preliminary",
            prelim_x=0.12, lumi_x=0.98, cms_text_size_mult=1.85
        )
        keep.append(ov)

        canvas.Modified(); canvas.Update()
        canvas._keep = keep
        return canvas

    # ── Label helpers ─────────────────────────────────────────────────────────

    def _draw_group_labels(self, pad, deco, left_m, right_m, y_pos=0.6):
        pad.cd()
        n, dw = deco["n_bins"], 1.0 - left_m - right_m
        lt = ROOT.TLatex(); lt.SetNDC(True)
        lt.SetTextFont(42)
        lt.SetTextSize(deco.get("group_label_text_size", 0.062))
        lt.SetTextAlign(22)
        gap = 0.035  # half-gap between two stacked lines
        for g in deco["group_labels"]:
            cx   = (g["start"] + g["end"]) / 2.0
            x    = left_m + (cx / n) * dw
            parts = g["text"].split(", ", 1)
            if len(parts) == 2:
                lt.DrawLatex(x, y_pos + gap, parts[0])
                lt.DrawLatex(x, y_pos - gap, parts[1])
            else:
                lt.DrawLatex(x, y_pos, g["text"])
        return [lt]

    def _draw_sub_group_labels(self, pad, deco, left_m, right_m, y_pos=0.43):
        pad.cd()
        if not deco.get("sub_group_labels"):
            return []
        n, dw = deco["n_bins"], 1.0 - left_m - right_m
        objs  = []

        lt = ROOT.TLatex(); lt.SetNDC(True)
        lt.SetTextFont(42)
        lt.SetTextSize(deco.get("sub_group_label_text_size", 0.12))
        lt.SetTextAlign(22)
        y_pos = deco.get("sub_group_label_y", y_pos)
        for sg in deco["sub_group_labels"]:
            cx = (sg["start"] + sg["end"]) / 2.0
            lt.DrawLatex(left_m + (cx / n) * dw, y_pos, sg["text"])
        objs.append(lt)

        sg_title = deco.get("sub_group_axis_title", "")
        if sg_title:
            tlt = ROOT.TLatex(); tlt.SetNDC(True)
            tlt.SetTextFont(42); tlt.SetTextSize(0.12); tlt.SetTextAlign(32)
            tlt.DrawLatex(left_m + 0.005, y_pos, sg_title)
            objs.append(tlt)

        return objs

    def _draw_bin_labels(self, pad, deco, left_m, right_m, has_sub_groups=False, abcd_y_offset=0.0):
        pad.cd()
        n, dw = deco["n_bins"], 1.0 - left_m - right_m
        y_bin   = 0.31 if has_sub_groups else 0.32
        y_title = 0.09 if has_sub_groups else 0.12
        sect_sep = deco.get("section_separator")

        is_risr = deco.get("bin_scheme") == "risr"
        is_compressed_final = deco.get("bin_scheme") == "compressed_final"
        is_noncompressed_final = deco.get("bin_scheme") == "noncompressed_final"
        if is_noncompressed_final:
            y_bin = 0.3 if has_sub_groups else 0.285
        lt = ROOT.TLatex(); lt.SetNDC(True)
        lt.SetTextFont(42)
        lt.SetTextSize(deco.get("bin_label_text_size", 0.13))
        if is_compressed_final:
            lt.SetTextAlign(22)
            for seq in deco.get("risr_sequence_labels", []):
                cx = (seq["start"] + seq["end"]) / 2.0
                x = left_m + (cx / n) * dw
                lt.DrawLatex(x, deco.get("risr_sequence_y", 0.205), seq["text"])
            objs = [lt]

            xt = deco.get("x_axis_title", "")
            if xt:
                tlt = ROOT.TLatex(); tlt.SetNDC(True)
                tlt.SetTextFont(42)
                tlt.SetTextSize(deco.get("x_title_text_size", 0.17))
                tlt.SetTextAlign(22)
                tlt.DrawLatex(left_m + 0.5 * dw, deco.get("x_title_y", 0.052), xt)
                objs.append(tlt)
            return objs

        if is_risr:
            lt.SetTextAngle(-20)
            lt.SetTextAlign(12)  # left end at bin left-edge, text slopes down-right into margin
        else:
            lt.SetTextAlign(22)
        for i, label in enumerate(deco["bin_labels"]):
            if is_risr:
                x = left_m + (i / n) * dw          # left edge of bin
            else:
                x = left_m + ((i + 0.5) / n) * dw  # centre of bin
            y = y_bin + (0.04 if is_risr else 0.0) + (abcd_y_offset if sect_sep is not None and i < sect_sep else 0.0)
            if is_compressed_final:
                y = 0.25
            lt.DrawLatex(x, y, label)

        objs = [lt]

        ann = deco.get("abcd_annotation")
        if ann:
            cx    = (ann["start"] + ann["end"]) / 2.0
            x_ann = left_m + (cx / n) * dw
            y_ann = y_bin + abcd_y_offset
            ann_lt = ROOT.TLatex(); ann_lt.SetNDC(True)
            ann_lt.SetTextFont(42); ann_lt.SetTextSize(0.13); ann_lt.SetTextAlign(22)
            ann_lt.DrawLatex(x_ann, y_ann, ann["text"])
            objs.append(ann_lt)

        xt = deco.get("x_axis_title", "")
        if xt:
            tlt = ROOT.TLatex(); tlt.SetNDC(True)
            tlt.SetTextFont(42)
            tlt.SetTextSize(deco.get("x_title_text_size", 0.17))
            tlt.SetTextAlign(22)
            tlt.DrawLatex(left_m + 0.5 * dw, y_title, xt)
            objs.append(tlt)
        return objs

    def _draw_separators(self, canvas, deco, left_m, right_m, name):
        n, dw = deco["n_bins"], 1.0 - left_m - right_m

        ov = ROOT.TPad(f"sep_{name}", "", 0, 0, 1, 1)
        ov.SetFillStyle(0); ov.SetFrameFillStyle(0)
        ov.SetBorderSize(0); ov.SetBorderMode(0); ov.SetMargin(0, 0, 0, 0)
        ov.SetBit(ROOT.kCannotPick)
        canvas.cd(); ov.Draw(); ov.cd()

        objs = [ov]

        #y1 = 0.08 if deco.get("bin_scheme") == "compressed_final" else 0.08
        y1 = 0.08
        y2 = 0.91

        for sb in deco.get("minor_sep_bins", []):
            x = left_m + (sb / n) * dw
            ln = ROOT.TLine(); ln.SetNDC(True)
            ln.SetLineColor(ROOT.kGray + 1); ln.SetLineWidth(1); ln.SetLineStyle(3)
            ln.DrawLine(x, y1, x, y2); objs.append(ln)

        for sb in deco.get("sub_sep_bins", []):
            x = left_m + (sb / n) * dw
            ln = ROOT.TLine(); ln.SetNDC(True)
            ln.SetLineColor(ROOT.kGray + 2); ln.SetLineWidth(2); ln.SetLineStyle(2)
            ln.DrawLine(x, y1, x, y2); objs.append(ln)

        for sb in deco.get("separator_bins", []):
            x = left_m + (sb / n) * dw
            ln = ROOT.TLine(); ln.SetNDC(True)
            if deco.get("bin_scheme") in ("compressed_final", "noncompressed_final"):
                ln.SetLineColor(ROOT.kBlack); ln.SetLineWidth(3); ln.SetLineStyle(1)
            else:
                ln.SetLineColor(ROOT.kGray + 2); ln.SetLineWidth(2); ln.SetLineStyle(2)
            ln.DrawLine(x, y1, x, y2); objs.append(ln)

        ss = deco.get("section_separator")
        if ss is not None:
            x = left_m + (ss / n) * dw
            ln = ROOT.TLine(); ln.SetNDC(True)
            ln.SetLineColor(ROOT.kBlack); ln.SetLineWidth(3); ln.SetLineStyle(1)
            ln.DrawLine(x, 0.08, x, 0.91); objs.append(ln)

        canvas.Modified(); canvas.Update()
        return objs

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _ch_short(ch):
        return ch.lower().replace("cr", "").replace("_", "")

    def _flush(self, plots, output_prefix, fmt):
        """
        Save all collected (canvas_pre, canvas_post, name) tuples.

        ROOT format: one file at {output_prefix}.root containing every canvas,
                     named '{name}_prefit' / '{name}_postfit'.
        Other formats: one file per canvas at {output_prefix}_{name}_{tag}.{fmt}.
        """
        if fmt == "root":
            path = f"{output_prefix}.root"
            fo   = ROOT.TFile(path, "RECREATE")
            for c_pre, c_post, name in plots:
                c_pre.SetName(f"{name}_prefit")
                c_post.SetName(f"{name}_postfit")
                c_pre.Write()
                c_post.Write()
            fo.Close()
            print(f"Saved: {path}  ({len(plots)*2} canvases)")
        else:
            for c_pre, c_post, name in plots:
                for c, tag in [(c_pre, "prefit"), (c_post, "postfit")]:
                    path = f"{output_prefix}_{name}_{tag}.{fmt}"
                    c.SaveAs(path)
                    print(f"Saved: {path}")
