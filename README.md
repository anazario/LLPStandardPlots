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
| `--flags` | Selection flags (`passN...`) or custom cut strings | No | Tight SR selections |
| `--labels` | Custom display labels for custom cut regions (1:1 with non-flag entries in `--flags`) | No | — |
| `--plots` | Plot types: `1d`, `2d`, `ratio`, `unrolled`, `cr_sig`, `all` | No | `all` |
| `--vars` | Variables to plot in 1D/CR-vs-SR mode (must be in `src/config.py`) | No | Mode-dependent |
| `--analysis-type` | Analysis type: `uncompressed` or `compressed` | No | `uncompressed` |
| `--isr-pt-cut` | Minimum p_T(ISR) cut in GeV (compressed mode only) | No | `700` |
| `--output` | Output file (ROOT) or directory (PDF/PNG) | No | `standard_plots.root` |
| `--format` | Output format: `root`, `pdf`, `png`, `eps` | No | `root` |
| `--save-hists` | (ROOT format only) Also write histogram objects alongside canvases | No | False |
| `--lumi` | Integrated luminosity in fb⁻¹ | No | `400.0` |
| `--energy` | Centre-of-mass energy in TeV | No | `13.6` |
| `--normalize` | Normalize 1D and Data/MC plots to unit area | No | False |
| `--workers` | Number of parallel worker processes for file loading | No | `1` |
| `--unblind` | **WARNING**: Show data in all regions including signal regions | No | False |
| `--data-flag` | Selection flag used to load data for CR-vs-SR overlay plots | No | — |
| `--no-merge-qcd-gjets` | Disable automatic QCD+GJets merging in SV regions | No | False |

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

### YAML input config

Instead of passing files on the command line, you can use `--input-config` with a YAML file. This also supports per-group options like `combine` (merge files into one sample) and `scale` (per-group event weight multiplier).

```yaml
base_dir: /path/to/skims   # optional prefix for all file paths

lumi: 59.7
energy: 13.6
format: pdf
output: my_plots
analysis_type: uncompressed

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
  - passNHad1SelectionSRTight
  - passNLep1SelectionCRLoose

blind_cuts: [false, false]   # per-flag data blinding for custom cut strings
```

### Example commands

```bash
# Basic run with explicit files
python main.py \
    --signal data/signal_mGl-1500_*.root \
    --background data/QCD_*.root data/WJets_*.root \
    --data data/JetHT_*.root \
    --flags passNHad1SelectionSRTight passNLep1SelectionCRLoose \
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
    --data-flag passNPhoGe1SelectionPromptLooseNotTightIsoCR \
    --format pdf
```

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
├── src/
│   ├── config.py              # Central configuration (variables, bins, mode configs)
│   ├── loader.py              # Data loading logic (Uproot → NumPy), parallel workers
│   ├── plotter.py             # Plotting classes (1D, 2D, Data/MC, Unrolled)
│   ├── fit_plotter.py         # Pre/postfit plotting (FitDiagnostics)
│   ├── style.py               # Style management (cmsstyle wrapper)
│   ├── selections.py          # Selection definitions and helpers
│   ├── input_config.py        # YAML input config parsing and group assembly
│   ├── unrolled.py            # Unrolled binning mathematics
│   └── utils.py               # Helper functions (name parsing)
└── tools/
    └── eps_to_pdf.py          # Helper to convert EPS plots to PDF
```
