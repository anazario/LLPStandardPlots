# LLP Standard Plots

A flexible plotting framework for Long-Lived Particle (LLP) analyses, designed to generate 1D, 2D, Data/MC comparison, unrolled, and pre/postfit plots from ROOT files using `uproot` for fast I/O and `PyROOT` + `cmsstyle` for publication-quality visualization.

## Requirements

- `ROOT` (with `PyROOT` support)
- `uproot`
- `numpy`
- `cmsstyle` (CMS plotting style)

## Scripts

| Script | Purpose |
|--------|---------|
| `main.py` | Main plotting entry point — 1D, 2D, Data/MC, unrolled, CR-vs-SR plots |
| `fit_plots.py` | Pre/postfit yield plots from Combine `FitDiagnostics` output |
| `replot_yields.py` | Redraw the Data/MC plots from a yields file with MC scale factors (seconds) |
| `fit_sfs.py` | Likelihood fit of per-process (× era) MC scale factors to data from a yields file |

---

## `main.py` — Standard Plots

### Arguments

| Argument | Description | Required | Default |
|----------|-------------|:--------:|:-------:|
| `--signal` | Signal ROOT files or directories | Yes* | — |
| `--background` | Background ROOT files or directories | No | — |
| `--data` | Data ROOT files or directories (enables Data/MC plots) | No | — |
| `--input-config` | YAML config file specifying inputs and run parameters | No | — |
| `--tree` | ROOT tree name | No | `kuSkimTree` |
| `--flags` | Selection flags (`passN...`) or custom cut strings | No | `passN{Had,Lep}Ge1SelectionHighDxySigSR` |
| `--labels` | Custom display labels for custom cut regions (1:1 with non-flag entries in `--flags`) | No | — |
| `--plots` | Plot types: `1d`, `2d`, `ratio`, `unrolled`, `cr_sig`, `all` | No | `all` |
| `--vars` | Variables to plot in 1D/CR-vs-SR mode (must be in `src/config.py`) | No | Mode-dependent |
| `--analysis-type` | Analysis type: `uncompressed` or `compressed` | No | `uncompressed` |
| `--isr-pt-cut` | Minimum p_T(ISR) cut in GeV (compressed mode only) | No | `700` |
| `--met-cut` | Override the baseline `selCMet` cut in GeV | No | `150` |
| `--output` | Output file (ROOT) or directory (PDF/PNG) | No | `standard_plots.root` |
| `--format` | Output format: `root`, `pdf`, `png`, `eps` | No | `root` |
| `--save-hists` | (ROOT format only) Also write histogram objects alongside canvases | No | False |
| `--lumi` | One global integrated luminosity in fb⁻¹ for all MC | No | per-run (see below), else `400.0` |
| `--years` | Data years setting the per-run MC luminosities: `run2`, `run3`, `all`, or e.g. `16,17,18,22` | No | years of the data files |
| `--energy` | Centre-of-mass energy in TeV | No | `13.6` |
| `--normalize` | Normalize 1D and Data/MC plots to unit area | No | False |
| `--workers` | Number of parallel worker processes for file loading | No | `1` |
| `--unblind` | **WARNING**: Show data in all regions including signal regions | No | False |
| `--data-flag` | Selection flag used to load data for CR-vs-SR overlay plots | No | — |
| `--no-merge-qcd-gjets` | Disable automatic QCD+GJets merging in SV regions | No | False |
| `--yields [FILE]` | Also write the per-process × era yields file (YAML: `yields: true` or a path) | No | `<output>_yields.root` |

*`--signal` is required unless `--input-config` is provided.

### Plot types

- **`1d`** — per-sample signal and background distributions, plus signal vs. net background overlay
- **`2d`** — 2D distributions in the plane defined by the mode's `plot_2d_configs` (e.g. M_S vs R_S)
- **`ratio`** — Data/MC comparison with ratio panel; variables chosen automatically based on final state
- **`unrolled`** — Unrolled 1D representation of the 2D (M_S, R_S) plane, both `merged_rs` and `merged_ms` schemes
- **`cr_sig`** — CR data overlaid with SR signal, requires `--data-flag`

### Analysis modes

| Mode | Default 1D variables | Notes |
|------|---------------------|-------|
| `uncompressed` | `rjr_Ms`, `rjr_Rs` | Full uncompressed RJR variables |
| `compressed` | `rjrIsr_Ms`, `rjrIsr_MsPerp`, `rjrIsr_PtIsr`, `rjrIsr_RIsr`, `rjrIsr_Rs` | `ratio` and `unrolled` not yet supported |

Data/MC plots always include `rjr_Ms`, `rjr_Rs`, `selCMet`, `rjrIsr_RIsr`, and `rjrIsr_PtIsr`, plus SV and photon variables determined by the selection flag.

### Baseline selection

Every event must pass `selCMet > 150`, `evtFillWgt < 10`, `Flag_MetFilters == 1`, `Flag_hemVeto == 0`, and the OR of `Trigger_PFMET120_PFMHT120_IDTight`, `Trigger_PFMETNoMu120_PFMHTNoMu120_IDTight`, `Trigger_PFMET120_PFMHT120_IDTight_PFHT60`, and `Trigger_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60`, plus `rjrPTS[0] < 150` (uncompressed) or `rjrIsr_nSVisObjects > 0` (compressed). A warning is printed for any file missing one of these branches, since that cut is then not applied.

### Blinding

Data is blinded per region from the selection flags it requires (`== 1`, a bare event flag, or `+`-joined):

- flags ending in `SR` are signal regions; flags ending in `CR`, and all validation flags (`...ValSR`, `...ValCR`), are control regions
- flags with neither suffix (`passNSVGe1Selection`, `passNSVEq0Selection`, `passNPhoGe1NonPrompt`, `passNPhoGe1SelectionLateSignal`) are inclusive and overlap the SRs
- a region is blinded if it requires an SR flag, or an inclusive flag without a CR flag (so `passNSVGe1SelectionLowDxySigCR+passNPhoGe1SelectionLateSignal` is shown, `passNPhoGe1SelectionLateSignal` alone is blinded); vetoed flags (`== 0`) are ignored

The same rule applies to custom cut strings. Custom cuts on raw variables only are blinded with `blind: true`. Only `--unblind` overrides the rule.

### YAML input config

Instead of passing files on the command line, you can use `--input-config` with a YAML file. This also supports per-group options like `combine` (merge files into one sample) and `scale` (per-group event weight multiplier).

```yaml
base_dir: /path/to/skims   # optional prefix for all file paths

lumi: 59.7
energy: 13.6
format: pdf
output: my_plots
analysis_type: uncompressed
normalize: true
global_cuts:
  - "selCMet > 250"

signal:
  - /eos/.../SMS_*_ct0p1_rjrskim.root        # plain path/glob — scale defaults to 1.0
  - name: SMS ct5
    scale: 10.0                               # optional per-group weight multiplier
    files:
      - /eos/.../SMS_*_ct5_rjrskim.root

background:
  - name: QCD multijets
    files:
      - QCD_Pt-*.root
  - name: EWK combined
    combine: true          # merge all files into a single sample
    scale: 1.3
    files:
      - WJets.root
      - ZJets.root

data:
  - name: Run 2+3
    combine: true
    files:
      - MET_R16.root
      - MET_R17.root
      - MET_R18.root

flags:
  - passNHadGe1SelectionHighDxySigSR
  - passNLepGe1SelectionLowDxySigCR

blind_cuts: [false, false]   # per-flag data blinding for custom cut strings
```

#### Run 2 + Run 3 luminosity

Leave `lumi` out and each MC file is scaled by the luminosity of its own run, so a process made of Run 2 and Run 3 samples enters the stack as one entry with each part normalized to its run. The luminosities are summed over the selected data years (table in `src/lumi.py`, same numbers as `llpcombine.py`), and the header shows the rounded total, e.g. `424 fb⁻¹ (13 TeV + 13.6 TeV)`.

```yaml
years: all        # run2, run3, all, or a list like [16, 17, 18, 22]; default: the years of the data files

background:
  - name: QCD multijets
    files:
      - QCD_HT*_RunIISummer20UL18_*.root           # era from the file name -> run2
      - QCD-4Jets_HT-*_Run3Summer23BPix_*.root     # -> run3
      - {path: QCD_other.root, era: run3}          # explicit: run2 | run3 | all
```

The era is read from the file name (`_R18`/`_R23`, `RunIISummer20UL18`, `Run3Summer23`, `Rn3Sm23`, `Run2024C`); a background file that names no run is an error until it is given an `era:` (per file, or once for the whole group). Signal files that name no run get the total. The file → era → luminosity table is printed before loading. Setting `lumi` keeps one global luminosity for every file. See `config/datamc_v34_run2run3_splitlumi.yaml`.

### Example commands

```bash
# Basic run with explicit files
python main.py \
    --signal data/signal_mGl-1500_*.root \
    --background data/QCD_*.root data/WJets_*.root \
    --data data/JetHT_*.root \
    --flags passNHadGe1SelectionHighDxySigSR passNLepGe1SelectionLowDxySigCR \
    --plots all \
    --output my_analysis_plots \
    --format pdf \
    --lumi 59.7

# Compressed analysis with ISR pT cut and parallel loading
python main.py \
    --input-config config/my_compressed.yaml \
    --analysis-type compressed \
    --isr-pt-cut 800 \
    --workers 4 \
    --plots 1d 2d

# Data/MC only, with CR-vs-SR overlay
python main.py \
    --input-config config/my_config.yaml \
    --plots ratio cr_sig \
    --data-flag passNPhoEq1SelectionPromptMedIsoCR \
    --format pdf
```

---

## MC scale factors: `--yields`, `replot_yields.py`, `fit_sfs.py`

Run the full dataset once with `--yields` (or `yields: true` in the YAML). Next to the
plots, `main.py` writes `<output>_yields.root`: for every region and Data/MC variable,
one `TH1D` per **background process × era** (`bkg__<process>__<run2|run3>`) and the
data per era (`data__<era>`). The histograms are filled before the files of a stack
entry are combined and before the QCD + γ+jets merge of the SV regions, and they keep
the MC stat errors. A `meta` object (JSON) holds the processes, regions, variables and
per-run luminosities.

```bash
# 1. Full run (slow, once)
python main.py --input-config config/datamc_v34_run2run3_splitlumi.yaml --yields

# 2. Fit the SFs (one SF per process; separate Run 2 / Run 3 SFs for QCD); or put the options
#    in the config's sf_fit: block and run  python fit_sfs.py --config <config>
python fit_sfs.py datamc_v34_run2run3_splitlumi_yields.root --per-era "QCD multijets" -o sfs.yaml

# 3. Redraw every Data/MC plot with them (seconds); prints data, MC before/after and chi2 per plot
python replot_yields.py datamc_v34_run2run3_splitlumi_yields.root --sf sfs.yaml
```

SF files can also be written by hand:

```yaml
scale_factors:
  QCD multijets: 1.10                 # all eras
  W+jets: {run2: 1.05, run3: 0.95}    # per era
regions:                              # optional per-region overrides
  CustomRegion2:
    QCD multijets: 0.8
```

`replot_yields.py` options: `--regions`, `--vars`, `--era run3` (Run 3 MC vs Run 3 data),
`--normalize`, `--format pdf png root`, `--table-only`.

`fit_sfs.py` maximizes a binned Poisson likelihood with the MC stat uncertainty profiled
per bin (Barlow–Beeston-lite); `-2 ln L` at the minimum is a χ²-like goodness of fit.
The channels default to the first variable of every unblinded region and are split by
era when the data allow it. Fit one variable per region — every variable of a region
contains the same events — and prefer non-overlapping regions. Processes with the same
shape in all channels are degenerate (large anti-correlation in the printed matrix):
`--fix "t#bar{t}+jets=1"` them or add a region that separates them. Other options:
`--channels REGION:VAR ...`, `--float`, `--per-era all`, `--era`, `--prior 0.5`
(log-normal constraint), `--rebin N` or `--rebin EDGE EDGE ...` (fit coarser bins: SFs only
fix normalizations, so coarse bins avoid fitting fine shape mismodelling), `--no-mc-stat`.
SFs are limited to [1e-4, 100]; a parameter the data push to a limit is reported as
`at lower/upper bound` with no error, and the other errors are computed with it held there.

#### Fit from the YAML config (`sf_fit:`)

Add an `sf_fit:` block to the `main.py` config. A full `main.py` run then also writes
the yields (implied), fits, and — with `replot: true` — redraws the plots with the SFs.
To iterate, rerun only the fit and replot (seconds) with `fit_sfs.py --config`, which
reads the same block and finds the yields file from the config's `output:`. CLI flags
override the block. Without the block, `main.py` behaves as before.

```yaml
output: datamc_v34_run2run3_splitlumi
sf_fit:
  channels: [CustomRegion1:isr_risr, CustomRegion2:isr_risr]   # default: first var of each unblinded region
  per_era: ["QCD multijets"]       # or "all"
  fix: {"t#bar{t}+jets": 1.0}
  float: [...]                     # default: every process not in fix
  era: [run3]                      # fit only these eras
  era_split: true                  # fit Run 2 / Run 3 data separately (default)
  prior: 0.5                       # log-normal constraint width (default: none)
  mc_stat: true                    # Barlow-Beeston-lite (default)
  min_mc: 0
  rebin: 5                         # merge every 5 bins; or edges [0.4, 0.6, 0.8, 1.0] (must be existing
                                   # edges); or per variable {isr_risr: 5, isr_ptisr: [300, 600, 1000, 2800]}
  replot: true                     # default false: fit + SF file only
  output: my_sfs.yaml              # default <output>_sfs.yaml
  replot_output: my_dir            # default <output>_sf
  replot_format: [pdf, png]        # default: the config's format
```

```bash
python main.py --input-config cfg.yaml                    # full run + fit (+ replot)
python fit_sfs.py --config cfg.yaml                       # fit (+ replot) only, from the saved yields
python fit_sfs.py --config cfg.yaml --per-era all --replot   # one-off change, YAML untouched
```

Output, next to the nominal plots:

```
datamc_v34_run2run3_splitlumi/                 nominal plots (unchanged)
datamc_v34_run2run3_splitlumi_yields.root      yields file
datamc_v34_run2run3_splitlumi_sfs.yaml         fitted SFs + fit summary (replot_yields.py --sf)
datamc_v34_run2run3_splitlumi_sf/<region>/datamc_plots/datamc_<var>_<region>.pdf   replot with SFs
```

A failing fit (e.g. a misspelled process) does not lose the full run: the plots and
yields are written first.

---

## `fit_plots.py` — Pre/Postfit Plots

Produces pre- and postfit yield comparison plots from a Combine `FitDiagnostics` ROOT file. Supports both compressed and uncompressed fit configurations, auto-detected from the fit config YAML.

### Arguments

| Argument | Description | Required | Default |
|----------|-------------|:--------:|:-------:|
| `--fit-result` | Path to `fitDiagnostics` ROOT file | Yes | — |
| `--fit-config` | Path to fit config YAML | Yes | — |
| `--output` | Output path prefix (e.g. `plots/myfit`) | No | `fit` |
| `--format` | Output format: `pdf`, `png`, `eps`, `root` | No | `pdf` |
| `--lumi` | Integrated luminosity in fb⁻¹ | No | `136.0` |
| `--energy` | Centre-of-mass energy in TeV | No | `13.0` |
| `--mode` | Override auto-detected fit mode: `compressed` or `uncompressed` | No | auto |
| `--label-scheme` | Axis label scheme: `auto`, `legacy`, `compressed-final`, `noncompressed-final` | No | `auto` |
| `--sr-color` | Highlight predicted SR bin in orange on ABCD plots | No | False |

### Example commands

```bash
# Compressed fit (auto-detected)
python fit_plots.py \
    --fit-result root/fits/compressed/fitDiagnostics.root \
    --fit-config root/fits/compressed/FitConfig.yaml \
    --output plots/compressed --lumi 12 --format pdf

# Uncompressed fit with final-state label scheme
python fit_plots.py \
    --fit-result root/fits/uncompressed/fitDiagnostics.root \
    --fit-config root/fits/uncompressed/FitConfig.yaml \
    --output plots/uncompressed \
    --label-scheme noncompressed-final \
    --sr-color --format pdf
```

---

## Configuration

Variables for 1D and Data/MC plots are defined in `src/config.py`. To add a new variable:

```python
class AnalysisConfig:
    VARIABLES = {
        'my_new_var': {
            'name': 'branch_name_in_root_tree',
            'label': 'Axis Label [Units]',
            'bins': 50,
            'range': (0, 100),
            'mc_only': False,   # set True to exclude from Data/MC plots
        },
    }
```

## Project Structure

```
├── main.py                    # Main CLI entry point
├── fit_plots.py               # Pre/postfit plot CLI
├── replot_yields.py           # Data/MC replot from a yields file with MC SFs
├── fit_sfs.py                 # Likelihood fit of MC SFs from a yields file
├── src/
│   ├── config.py              # Central configuration (variables, bins, mode configs)
│   ├── loader.py              # Data loading logic (Uproot → NumPy), parallel workers
│   ├── plotter.py             # Plotting classes (1D, 2D, Data/MC, Unrolled)
│   ├── fit_plotter.py         # Pre/postfit plotting (FitDiagnostics)
│   ├── style.py               # Style management (cmsstyle wrapper)
│   ├── selections.py          # Selection definitions and helpers
│   ├── input_config.py        # YAML input config parsing and group assembly
│   ├── yields.py              # Per-process × era yields file, scale factors
│   ├── unrolled.py            # Unrolled binning mathematics
│   └── utils.py               # Helper functions (name parsing)
└── tools/
    └── eps_to_pdf.py          # Helper to convert EPS plots to PDF
```
