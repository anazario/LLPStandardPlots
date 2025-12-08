# LLP Standard Plots

A flexible plotting framework for Long-Lived Particle (LLP) analyses, designed to generate 1D, 2D, and Data/MC comparison plots from ROOT files using `uproot` for fast I/O and `PyROOT` + `cmsstyle` for publication-quality visualization.

## Requirements

The project relies on the following Python packages:

- `ROOT` (with `PyROOT` support)
- `uproot`
- `numpy`
- `cmsstyle` (CMS plotting style)

## CLI Usage

The main entry point is `main.py`. You can generate plots directly from the command line by specifying your input files and desired configuration.

### Arguments

| Argument | Description | Required | Default |
|----------|-------------|:--------:|:-------:|
| `--signal` | List of signal ROOT files | Yes | - |
| `--background` | List of background ROOT files | Yes | - |
| `--data` | List of data ROOT files (for Data/MC plots) | No | - |
| `--flags` | List of selection flags (e.g., `passNHad1SelectionSRTight`) or custom cut strings | No | Tight selections |
| `--plots` | Types of plots: `1d`, `2d`, `ratio` (Data/MC), or `all` | No | `all` |
| `--vars` | Variables to plot in 1D mode (must be in config) | No | `rjr_Ms`, `rjr_Rs` |
| `--output` | Output filename (for ROOT) or directory (for PDF/PNG) | No | `standard_plots.root` |
| `--format` | Output format: `root`, `pdf`, `png`, `eps` | No | `root` |
| `--lumi` | Integrated luminosity in fb⁻¹ | No | 400.0 |
| `--normalize` | Normalize 1D plots to unit area | No | False |
| `--unblind` | **WARNING**: Bypass blinding to show data in signal regions | No | False |

### Example Command

```bash
python main.py \
    --signal data/signal_mGl-1500_*.root \
    --background data/QCD_*.root data/WJets_*.root \
    --data data/JetHT_*.root \
    --flags passNHad1SelectionSRTight passNLep1SelectionCRLoose \
    --plots all \
    --output my_analysis_plots \
    --format pdf \
    --lumi 59.7
```

## Programmatic Usage

You can also use the classes in `src/` directly in your own Python scripts for more advanced customization.

### Example Script

```python
import ROOT
from src.loader import DataLoader
from src.style import StyleManager
from src.plotter import Plotter1D

# 1. Setup Style
style = StyleManager(luminosity=59.7)
style.set_style()

# 2. Initialize Loader and Plotter
loader = DataLoader("kuSkimTree", luminosity=59.7)
plotter = Plotter1D(style)

# 3. Load Data
# Unified loader handles file I/O and selection application
signal_files = ["path/to/signal.root"]
bg_files = ["path/to/background.root"]
flags = ["passNHad1SelectionSRTight"]

sig_data, _ = loader.load_data_unified(signal_files, flags, [])
bg_data, _ = loader.load_data_unified(bg_files, flags, [])

# 4. Generate Plot
# Data structure is: data[flag][filename] = {variable: numpy_array}
current_sig = sig_data["passNHad1SelectionSRTight"]

# Plot 'rjr_Ms' (must be defined in src/config.py)
canvas = plotter.plot_collection(
    data_map=current_sig,
    var_name="rjr_Ms",
    x_label="M_{s} [GeV]",
    nbins=50,
    xmin=0,
    xmax=2000,
    collection_type="Signal"
)

# 5. Save
canvas.SaveAs("my_custom_plot.pdf")
```

## Configuration

The plotting configuration for variables is central located in `src/config.py`. To plot a new variable, add it to the `VARIABLES` dictionary:

```python
# src/config.py

class AnalysisConfig:
    VARIABLES = {
        # ... existing variables ...
        'my_new_var': {
            'name': 'branch_name_in_root_tree',
            'label': 'Axis Label [Units]',
            'bins': 50,
            'range': (0, 100)
        },
    }
```

## Project Structure

```
├── main.py              # CLI Entry point
├── src/
│   ├── config.py        # Central configuration (variables, bins)
│   ├── loader.py        # Data loading logic (Uproot -> NumPy)
│   ├── plotter.py       # Plotting classes (1D, 2D, Data/MC)
│   ├── style.py         # Style management (cmsstyle wrapper)
│   ├── selections.py    # Selection definitions and helpers
│   └── utils.py         # Helper functions (name parsing)
```
