#!/usr/bin/env python3
import argparse
import sys
import ROOT
from pathlib import Path
try:
    import cmsstyle as CMS  # noqa: F401
except ImportError:
    print("Warning: cmsstyle not available, using built-in fallback style.")
import os
import glob

from src.style import StyleManager
from src.loader import DataLoader
from src.plotter import Plotter1D, Plotter2D, PlotterDataMC
from src.selections import FinalStateResolver
from src.utils import parse_signal_name, parse_background_name

from src.config import AnalysisConfig, AnalysisMode, ModeConfig
from src.input_config import (load_input_config, apply_config_to_args,
                               unique_files_from_groups, assemble_grouped_map)

def expand_input_paths(input_paths):
    """
    Expand input paths to handle both individual files and directories with ROOT files.
    
    Args:
        input_paths: List of file paths and/or directory paths
        
    Returns:
        List of ROOT file paths
        
    Raises:
        FileNotFoundError: If a directory contains no ROOT files
    """
    expanded_paths = []
    
    for path_str in input_paths:
        path = Path(path_str)
        
        if path.is_file():
            # Individual file - add as is
            expanded_paths.append(str(path))
        elif path.is_dir():
            # Directory - search for ROOT files
            try:
                root_files = list(path.glob("*.root"))
                if not root_files:
                    raise FileNotFoundError(f"No ROOT files found in directory: {path}")
                
                # Sort for consistent ordering
                root_files.sort()
                expanded_paths.extend([str(f) for f in root_files])
                print(f"Found {len(root_files)} ROOT files in {path}")
                
            except Exception as e:
                if isinstance(e, FileNotFoundError):
                    raise
                else:
                    raise Exception(f"Error accessing directory {path}: {e}")
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")
    
    return expanded_paths

def prompt_unblind_warning():
    """
    Interactive warning prompt for unblinding data with cursor navigation.
    Returns True if user chooses to continue, False if abort.
    """
    import sys
    import termios
    import tty
    
    print("\n" + "="*80)
    print("⚠️  DATA UNBLINDING WARNING ⚠️")
    print("="*80)
    print("You have requested to bypass data blinding with --unblind.")
    print("This will show data points in ALL regions, including signal regions.")
    print("")
    print("IMPORTANT REMINDERS:")
    print("• Signal regions (SR) should remain blinded until analysis is finalized")
    print("• Custom cuts may contain signal-like selections")
    print("• Unblinding should only be done for:")
    print("  - Final result validation")
    print("  - Control region checks") 
    print("  - Systematic studies with approval")
    print("")
    print("Are you sure you want to proceed with unblinding?")
    print("="*80)
    print("\nUse ↑↓ arrow keys to navigate, Enter to select, Ctrl+C to abort:")
    
    options = ["Continue", "Abort"]
    selected = 1  # Default to "Abort" for safety
    
    def get_char():
        """Get a single character from stdin without pressing enter."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            if ch == '\x1b':  # Arrow key sequence
                ch += sys.stdin.read(2)
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    def display_menu():
        """Display the menu with current selection highlighted."""
        # Clear current line and redraw all options
        for i, option in enumerate(options):
            print("\033[2K", end="")  # Clear current line
            if i == selected:
                print(f"  → {option} ←")
            else:
                print(f"    {option}")
        # Move cursor back up to start of options
        print(f"\033[{len(options)}A", end="", flush=True)
    
    try:
        display_menu()
        
        while True:
            char = get_char()
            
            if char == '\r' or char == '\n':  # Enter key
                # Clear the menu and move cursor down
                for i in range(len(options)):
                    print("\033[2K")  # Clear line
                    if i < len(options) - 1:
                        print("\033[1B", end="")  # Move down
                print()
                
                if selected == 0:
                    print("⚠️  Proceeding with data unblinding...")
                    return True
                else:
                    print("✓ Aborting. Data will remain blinded.")
                    return False
                    
            elif char == '\x1b[A':  # Up arrow
                selected = max(0, selected - 1)
                display_menu()
                
            elif char == '\x1b[B':  # Down arrow
                selected = min(len(options) - 1, selected + 1)
                display_menu()
                
            elif char == '\x03':  # Ctrl+C
                raise KeyboardInterrupt
                
    except (KeyboardInterrupt, EOFError):
        print("\n\n✓ Interrupted. Aborting unblinding.")
        return False

def _write_hists_from_pad(pad):
    """Recursively write all TH1-derived histograms found in a pad to the current ROOT directory."""
    for prim in pad.GetListOfPrimitives():
        if prim.InheritsFrom("TPad"):
            _write_hists_from_pad(prim)
        elif prim.InheritsFrom("TH1"):
            prim.Write()


def save_canvas(canvas, output_format, f_out=None, output_dir=None, subdir_path="", canvas_name=None, save_hists=False):
    """
    Save canvas in the specified format.
    For ROOT: writes to ROOT file in current directory
    For PDF/PNG: saves to output_dir with subdir_path structure
    If save_hists=True (ROOT format only), also writes each histogram as a
    standalone object alongside the canvas.
    """
    if output_format == 'root':
        if f_out is not None:
            canvas.Write()
            if save_hists:
                _write_hists_from_pad(canvas)
    else:
        # Create subdirectory structure for PDF/PNG
        save_dir = Path(output_dir) / subdir_path if subdir_path else Path(output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Use canvas name or generate one
        if canvas_name is None:
            canvas_name = canvas.GetName()

        # Save in the specified format
        save_path = save_dir / f"{canvas_name}.{output_format}"
        canvas.SaveAs(str(save_path))

_save_canvas_impl = save_canvas  # module-level alias used by the wrapper inside main()


def is_event_flag(flag_string):
    """Check if the string is a predefined event flag (starts with 'pass') or a custom cut."""
    return flag_string.startswith('pass')

def is_signal_region(flag_string):
    """Check if the flag represents a signal region (data should be blinded)."""
    # SR = Signal Region (blind data), CR = Control Region (show data)
    return 'SR' in flag_string

def parse_arguments():
    parser = argparse.ArgumentParser(description='Standard Plots CLI')
    
    # Input Files
    parser.add_argument('--signal', nargs='+', default=[], help='Signal ROOT files or directories containing ROOT files')
    parser.add_argument('--background', nargs='+', default=[], help='Background ROOT files or directories containing ROOT files')
    parser.add_argument('--data', nargs='*', help='Data ROOT files or directories containing ROOT files (for data/MC comparison plots)')
    parser.add_argument('--output', default='standard_plots.root', help='Output ROOT file')
    parser.add_argument('--tree', default='kuSkimTree', help='Tree name')
    
    # Selections
    parser.add_argument('--flags', nargs='+', default=[
        'passNHad1SelectionSRTight',
        'passNLep1SelectionSRTight'
    ], help='List of Final State Flags or custom cut strings to process')
    
    # Plot Types
    parser.add_argument('--plots', nargs='+', choices=['1d', '2d', 'ratio', 'unrolled', 'cr_sig', 'all'], default=['all'],
                       help='Types of plots to generate')
    
    # 1D Options
    parser.add_argument('--vars', nargs='+', default=None,
                       help='Variables to plot in 1D mode (must be defined in src/config.py). Defaults based on analysis type.')
    parser.add_argument('--normalize', action='store_true', help='Normalize 1D plots to area 1')

    # Analysis Options
    parser.add_argument('--analysis-type', choices=['uncompressed', 'compressed'],
                       default='uncompressed',
                       help='Analysis type: uncompressed (default) or compressed')
    parser.add_argument('--isr-pt-cut', type=float, default=None,
                       help='Minimum pT(ISR) cut in GeV (compressed mode only). Default: 700 when compressed mode is used.')
    parser.add_argument('--lumi', type=float, default=400.0, help='Integrated luminosity in fb^-1 (default: 400)')
    parser.add_argument('--energy', type=float, default=13.6, help='Centre-of-mass energy in TeV (default: 13.6)')
    parser.add_argument('--unblind', action='store_true', help='Bypass data blinding (shows data in all regions including signal regions)')
    parser.add_argument('--data-flag', default=None,
                       help='Flag used to load data files for CR-vs-SR overlay plots '
                            '(e.g. passNPhoGe1SelectionPromptLooseNotTightIsoCR). '
                            'Data is loaded with this flag independently of --flags.')
    parser.add_argument('--labels', nargs='+', default=None,
                       help='Custom labels for custom cut regions (1:1 with non-event-flag entries in --flags)')
    
    
    # Input config
    parser.add_argument('--input-config', default=None, metavar='FILE',
                       help='YAML config file specifying input files and optional run parameters')

    # Output Format Options
    parser.add_argument('--format', choices=['root', 'pdf', 'png', 'eps'], default='root',
                       help='Output format: root (default), pdf, png, or eps')
    parser.add_argument('--save-hists', action='store_true', default=False,
                       help='(ROOT format only) Also write individual histogram objects alongside canvases')
    
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load YAML input config if provided
    bg_groups = None
    data_groups = None
    if args.input_config:
        config = load_input_config(args.input_config)
        apply_config_to_args(args, config['overrides'])
        if not args.signal:
            args.signal = config['signal_files']
        if not args.background:
            bg_groups = config['bg_groups']
        if not args.data:
            data_groups = config['data_groups']

    if not args.signal:
        print("Error: --signal is required when not using --input-config.")
        sys.exit(1)

    # Get mode configuration
    analysis_mode = args.analysis_type
    mode_config = ModeConfig.get(analysis_mode)

    # Set default variables based on mode if not specified
    if args.vars is None:
        args.vars = mode_config['default_vars']

    # Handle ISR pT cut - only relevant in compressed mode
    isr_pt_cut = None
    if analysis_mode == AnalysisMode.COMPRESSED:
        if args.isr_pt_cut is not None:
            isr_pt_cut = args.isr_pt_cut
        else:
            isr_pt_cut = mode_config.get('isr_pt_cut_default', 700.0)
    elif args.isr_pt_cut is not None:
        print(f"Warning: --isr-pt-cut is ignored in uncompressed mode")

    # Check for unsupported plot types in compressed mode
    if analysis_mode == AnalysisMode.COMPRESSED:
        if 'ratio' in args.plots or 'unrolled' in args.plots or 'all' in args.plots:
            print(f"Note: Data/MC ratio and unrolled plots are not yet implemented for compressed mode.")
            print(f"      Only 1D, 2D, and cr_sig plots will be generated.")
            # Filter to only supported plot types
            if 'all' in args.plots:
                args.plots = ['1d', '2d', 'cr_sig']
            else:
                args.plots = [p for p in args.plots if p in ['1d', '2d', 'cr_sig']]

    # Expand input paths to handle directories
    try:
        signal_files = expand_input_paths(args.signal)
        if bg_groups is None:
            bg_groups = [{'name': None, 'files': expand_input_paths(args.background) if args.background else [], 'combine': False}]
        if data_groups is None:
            data_groups = [{'name': None, 'files': expand_input_paths(args.data) if args.data else [], 'combine': False}]
    except (FileNotFoundError, Exception) as e:
        print(f"Error: {e}")
        sys.exit(1)

    all_bg_files = unique_files_from_groups(bg_groups)
    all_data_files = unique_files_from_groups(data_groups)

    # Handle unblinding warning if requested
    if args.unblind and all_data_files:
        if not prompt_unblind_warning():
            print("Exiting...")
            return

    # Handle output format and smart file naming
    output_format = args.format
    output_path = args.output

    if output_format == 'root':
        # For ROOT format: handle smart .root extension
        if not output_path.endswith('.root'):
            output_path = f"{output_path}.root"
        # Will create a ROOT file
        use_root_file = True
        output_dir = None
    else:
        # For PDF/PNG formats: use output_path as directory name
        # Remove .root extension if accidentally provided
        if output_path.endswith('.root'):
            output_dir = output_path[:-5]  # Remove the last 5 characters (.root)
        else:
            output_dir = output_path
        use_root_file = False
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Setup
    style = StyleManager(luminosity=args.lumi, energy=args.energy)
    style.set_style()

    loader = DataLoader(args.tree, luminosity=args.lumi,
                       analysis_mode=analysis_mode, isr_pt_cut=isr_pt_cut)
    plotter1d = Plotter1D(style)
    plotter2d = Plotter2D(style)
    plotter_datamc = PlotterDataMC(style)
    resolver = FinalStateResolver()
    
    # Load Data
    # Separate event flags from custom cut strings
    event_flags = [flag for flag in args.flags if is_event_flag(flag)]
    custom_cuts = [flag for flag in args.flags if not is_event_flag(flag)]
    
    print(f"Loading data for {len(event_flags)} event flags and {len(custom_cuts)} custom cuts...")
    
    # Load each category once (background files deduplicated across groups)
    sig_data_map, custom_sig_data_map = loader.load_data_unified(signal_files, event_flags, custom_cuts)
    bg_flat_map, custom_bg_flat_map = loader.load_data_unified(all_bg_files, event_flags, custom_cuts)

    # Load data files if provided.
    # --data-flag can be either an event flag ("pass...") or a custom cut string.
    # We fold it into the appropriate list so everything loads in one pass.
    data_flat_map, custom_data_flat_map = {}, {}
    data_flag_is_event = args.data_flag and is_event_flag(args.data_flag)
    data_flag_key = None   # key used to retrieve cr_data_collection later
    if all_data_files:
        data_event_flags = event_flags.copy()
        data_custom_cuts = custom_cuts.copy()
        if args.data_flag:
            if data_flag_is_event:
                if args.data_flag not in data_event_flags:
                    data_event_flags.append(args.data_flag)
                data_flag_key = args.data_flag          # key in data_flat_map
            else:
                data_custom_cuts.append(args.data_flag)
                data_flag_key = f'CustomRegion{len(data_custom_cuts)}'  # key in custom_data_flat_map
        data_flat_map, custom_data_flat_map = loader.load_data_unified(
            all_data_files, data_event_flags, data_custom_cuts, is_data=True
        )

    # Assemble grouped maps (combines files within each group where combine=True)
    bg_data_map = assemble_grouped_map(bg_flat_map, bg_groups, loader.combine_data)
    custom_bg_data_map = assemble_grouped_map(custom_bg_flat_map, bg_groups, loader.combine_data)
    data_data_map = assemble_grouped_map(data_flat_map, data_groups, loader.combine_data)
    custom_data_data_map = assemble_grouped_map(custom_data_flat_map, data_groups, loader.combine_data)
    
    # Print comprehensive data loading summary
    loader.print_comprehensive_summary()
    
    # Output File (only for ROOT format)
    if use_root_file:
        f_out = ROOT.TFile(output_path, "RECREATE")
    else:
        f_out = None
    
    # Shadow the module-level save_canvas to bake in save_hists, keeping all call sites unchanged
    def save_canvas(canvas, fmt, fout, outdir, subdir="", cname=None):  # noqa: F811
        _save_canvas_impl(canvas, fmt, fout, outdir, subdir, cname, save_hists=args.save_hists)

    # Combine both event flags and custom cuts into one processing loop
    all_flags_and_cuts = []
    
    # Add event flags
    for flag in event_flags:
        all_flags_and_cuts.append({
            'name': flag,
            'data_source': 'event_flag',
            'sig_data': sig_data_map.get(flag, {}),
            'bg_data': bg_data_map.get(flag, {}),
            'show_region_label': True
        })
    
    # Add custom cuts
    for i, custom_region in enumerate(custom_sig_data_map.keys()):
        original_cut = custom_cuts[i] if i < len(custom_cuts) else "Unknown"
        custom_label = args.labels[i] if args.labels and i < len(args.labels) else None
        all_flags_and_cuts.append({
            'name': custom_region,
            'data_source': 'custom_cut',
            'sig_data': custom_sig_data_map.get(custom_region, {}),
            'bg_data': custom_bg_data_map.get(custom_region, {}),
            'show_region_label': False,
            'original_cut': original_cut,
            'custom_label': custom_label
        })
    
    # Process all flags and cuts uniformly
    for item in all_flags_and_cuts:
        flag = item['name']
        current_sig_data = item['sig_data'] 
        current_bg_data = item['bg_data']
        show_region_label = item['show_region_label']
        if item['data_source'] == 'custom_cut':
            print(f"\nProcessing {item['data_source']}: {flag} ('{item['original_cut']}')")
        else:
            print(f"\nProcessing {item['data_source']}: {flag}")
        
        # Check if we have data for this item
        has_sig = bool(current_sig_data)
        has_bg = bool(current_bg_data)
        
        if not has_sig and not has_bg:
            print(f"  Skipping {flag} (No events found)")
            continue
            
        # Create Directory structure
        if show_region_label:
            fs_label_latex = resolver.format_sv_label(flag)
        else:
            fs_label_latex = item.get('custom_label', None)
        # Clean latex for folder name (remove special chars)
        folder_name = flag 
        
        if use_root_file:
            fs_dir = f_out.mkdir(folder_name)
            fs_dir.cd()
        else:
            # For PDF/PNG, we'll pass folder_name as subdir_path to save_canvas
            fs_dir = None
        
        # Data is already assigned above
        current_bg_combined = loader.combine_data(current_bg_data)

        # Get corresponding data for this flag/cut
        if item['data_source'] == 'event_flag':
            current_data_data = data_data_map.get(flag, {})
        else:
            # For custom cuts, find the matching region (only if data files were provided)
            if custom_data_data_map:
                custom_region_idx = list(custom_sig_data_map.keys()).index(flag) if flag in custom_sig_data_map else -1
                if custom_region_idx >= 0 and custom_region_idx < len(custom_data_data_map):
                    custom_region_name = list(custom_data_data_map.keys())[custom_region_idx]
                    current_data_data = custom_data_data_map.get(custom_region_name, {})
                else:
                    current_data_data = {}
            else:
                # No data files provided
                current_data_data = {}
        
        # Retrieve CR data collection for --data-flag (shared by 2D and cr_sig blocks)
        cr_data_collection = {}
        if all_data_files and args.data_flag:
            if data_flag_is_event:
                cr_data_collection = data_data_map.get(data_flag_key, {})
            else:
                cr_data_collection = custom_data_data_map.get(data_flag_key, {})

        # --- Data/MC Comparison Plots (FIRST to avoid palette interference) ---
        if args.data and ('ratio' in args.plots or 'all' in args.plots):
            print("  Generating Data/MC Comparison Plots...")
            datamc_subdir = f"{folder_name}/datamc_plots"
            
            if use_root_file:
                datamc_dir = fs_dir.mkdir("datamc_plots")
                datamc_dir.cd()
            
            # Check if we should blind data in signal regions or custom cuts
            if args.unblind:
                blind_data = False  # Override blinding if --unblind flag is set
            else:
                blind_data = is_signal_region(flag) or item['data_source'] == 'custom_cut'
            
            # Determine variable set based on final state (like datamc_batch_process.py)
            datamc_vars = []
            
            # Always include event-level variables
            event_level_vars = ['rjr_Ms', 'rjr_Rs', 'selCMet']
            datamc_vars.extend(event_level_vars)
            
            if "NHad" in flag and "NLep" not in flag:
                # Pure hadronic final state - use HadronicSV variables
                hadSV_vars = ['HadronicSV_mass', 'HadronicSV_dxy', 'HadronicSV_dxySig',
                              'HadronicSV_pOverE', 'HadronicSV_decayAngle', 'HadronicSV_cosTheta',
                              'HadronicSV_nTracks']
                datamc_vars.extend(hadSV_vars)
            elif "NLep" in flag and "NHad" not in flag:
                # Pure leptonic final state - use LeptonicSV variables
                lepSV_vars = ['LeptonicSV_mass', 'LeptonicSV_dxy', 'LeptonicSV_dxySig',
                              'LeptonicSV_pOverE', 'LeptonicSV_decayAngle', 'LeptonicSV_cosTheta']
                datamc_vars.extend(lepSV_vars)
            elif "NHad" in flag and "NLep" in flag:
                # Combined final state - use BOTH HadronicSV and LeptonicSV variables
                hadSV_vars = ['HadronicSV_mass', 'HadronicSV_dxy', 'HadronicSV_dxySig',
                              'HadronicSV_pOverE', 'HadronicSV_decayAngle', 'HadronicSV_cosTheta',
                              'HadronicSV_nTracks']
                lepSV_vars = ['LeptonicSV_mass', 'LeptonicSV_dxy', 'LeptonicSV_dxySig',
                              'LeptonicSV_pOverE', 'LeptonicSV_decayAngle', 'LeptonicSV_cosTheta']
                datamc_vars.extend(hadSV_vars)
                datamc_vars.extend(lepSV_vars)
            # else: for other flags (custom cuts, etc.) use only event-level variables

            # Photon variables: added whenever NPho appears in the flag.
            # mc_only variables (Gen-level) are excluded here since data/MC
            # comparison requires the variable to exist in data files too.
            if "NPho" in flag:
                photon_vars = [
                    v for v, c in AnalysisConfig.VARIABLES.items()
                    if v.startswith('baseLinePhoton_') and not c.get('mc_only', False)
                ]
                datamc_vars.extend(photon_vars)
            
            for var_key in datamc_vars:
                conf = AnalysisConfig.VARIABLES.get(var_key)
                if not conf:
                    continue
                
                short_name = conf['name']
                label = conf['label']
                nbins = conf['bins']
                xmin, xmax = conf['range']
                
                # Create data/MC comparison plot
                if current_bg_data:  # At least need MC backgrounds
                    canvas = plotter_datamc.create_data_mc_comparison(
                        current_data_data, current_bg_data, short_name, label, 
                        nbins, xmin, xmax, blind_data=blind_data, 
                        final_state_label=fs_label_latex, suffix=flag
                    )
                    save_canvas(canvas, output_format, f_out, output_dir, datamc_subdir)
            
            # Return to parent directory
            if use_root_file:
                fs_dir.cd()
                
            # Generate normalized versions if --normalize is specified
            if args.normalize:
                print("  Generating Normalized Data/MC Comparison Plots...")
                datamc_norm_subdir = f"{folder_name}/datamc_plots_norm"
                
                if use_root_file:
                    datamc_norm_dir = fs_dir.mkdir("datamc_plots_norm")
                    datamc_norm_dir.cd()
                
                for var_key in datamc_vars:
                    conf = AnalysisConfig.VARIABLES.get(var_key)
                    if not conf:
                        continue
                    
                    short_name = conf['name']
                    label = conf['label']
                    nbins = conf['bins']
                    xmin, xmax = conf['range']
                    
                    # Create normalized data/MC comparison plot
                    if current_bg_data:  # At least need MC backgrounds
                        canvas_norm = plotter_datamc.create_data_mc_comparison(
                            current_data_data, current_bg_data, short_name, label, 
                            nbins, xmin, xmax, blind_data=blind_data, 
                            final_state_label=fs_label_latex, suffix=flag, normalized=True
                        )
                        save_canvas(canvas_norm, output_format, f_out, output_dir, datamc_norm_subdir)
                
                # Return to parent directory
                if use_root_file:
                    fs_dir.cd()
        
        # --- Unrolled Plots ---
        if ('unrolled' in args.plots or 'all' in args.plots) and current_bg_data:
            print(f"  Generating Unrolled Plots (both merged_rs and merged_ms)...")
            unrolled_subdir = f"{folder_name}/unrolled_plots"
            unrolled_norm_subdir = f"{folder_name}/unrolled_plots_norm"

            # Check blinding
            if args.unblind:
                blind_data = False
            else:
                blind_data = is_signal_region(flag) or item['data_source'] == 'custom_cut'

            # Create directories once before the loop
            if use_root_file:
                u_dir = fs_dir.mkdir("unrolled_plots")
                if args.normalize:
                    u_norm_dir = fs_dir.mkdir("unrolled_plots_norm")

            # Generate both schemes
            for scheme in ['merged_rs', 'merged_ms']:
                print(f"    Creating {scheme} plots...")

                # 1. Standard Unrolled
                if use_root_file:
                    u_dir.cd()

                canvas = plotter_datamc.create_unrolled_comparison(
                    current_data_data, current_bg_data,
                    scheme=scheme,
                    blind_data=blind_data,
                    final_state_label=fs_label_latex,
                    suffix=f"{flag}_{scheme}",
                    normalized=False
                )
                save_canvas(canvas, output_format, f_out, output_dir, unrolled_subdir)

                if use_root_file:
                    fs_dir.cd()

                # 2. Normalized Unrolled (if requested via --normalize)
                if args.normalize:
                    if use_root_file:
                        u_norm_dir.cd()

                    canvas_norm = plotter_datamc.create_unrolled_comparison(
                        current_data_data, current_bg_data,
                        scheme=scheme,
                        blind_data=blind_data,
                        final_state_label=fs_label_latex,
                        suffix=f"{flag}_{scheme}",
                        normalized=True
                    )
                    save_canvas(canvas_norm, output_format, f_out, output_dir, unrolled_norm_subdir)

                    if use_root_file:
                        fs_dir.cd()

        # --- 2D Plots ---
        if '2d' in args.plots or 'all' in args.plots:
            print("  Generating 2D Plots...")
            plots_2d_subdir = f"{folder_name}/2D_plots"

            if use_root_file:
                plots_2d_dir = fs_dir.mkdir("2D_plots")
                plots_2d_dir.cd()

            # Get 2D plot configurations from mode config
            plot_2d_configs = mode_config['plot_2d_configs']

            for plot_config in plot_2d_configs:
                x_var = plot_config['x_var']
                y_var = plot_config['y_var']
                suffix = plot_config['suffix']

                # Signal 2D
                for fname, data in current_sig_data.items():
                    sig_name = parse_signal_name(fname)
                    sample_label_x_pos = 0.32  # Original logic for signal
                    canvas, _ = plotter2d.plot_2d(
                        data, f"sig_2d_{Path(fname).stem}_{flag}_{suffix}",
                        sig_name, fs_label_latex,
                        sample_label_x_pos=sample_label_x_pos,
                        x_var=x_var, y_var=y_var
                    )
                    if canvas:
                        save_canvas(canvas, output_format, f_out, output_dir, plots_2d_subdir)

                # Background 2D (Individual)
                for fname, data in current_bg_data.items():
                    bg_name = parse_background_name(fname)
                    # Original logic for individual background
                    sample_label_x_pos = 0.62 if "QCD" in bg_name else 0.69
                    canvas, _ = plotter2d.plot_2d(
                        data, f"bg_2d_{Path(fname).stem}_{flag}_{suffix}",
                        bg_name, fs_label_latex,
                        sample_label_x_pos=sample_label_x_pos,
                        x_var=x_var, y_var=y_var
                    )
                    if canvas:
                        save_canvas(canvas, output_format, f_out, output_dir, plots_2d_subdir)

                # Background 2D (Combined)
                if current_bg_combined:
                    sample_label_x_pos = 0.59  # Original logic for combined background
                    canvas, _ = plotter2d.plot_2d(
                        current_bg_combined, f"bg_2d_total_{flag}_{suffix}",
                        "Total Background", fs_label_latex,
                        sample_label_x_pos=sample_label_x_pos,
                        x_var=x_var, y_var=y_var
                    )
                    if canvas:
                        save_canvas(canvas, output_format, f_out, output_dir, plots_2d_subdir)

                # Data 2D — two independent sources, each producing its own canvas:
                # 1. Current flag (CR only): data loaded under the same flag as signal/bg
                # 2. --data-flag CR collection (if provided and different from current flag)
                data_2d_cases = []
                if current_data_data and not is_signal_region(flag):
                    data_2d_cases.append((
                        current_data_data, fs_label_latex,
                        f"data_2d_{flag}_{suffix}"
                    ))
                if cr_data_collection and args.data_flag != flag:
                    cr_region_label = (resolver.format_sv_label(args.data_flag)
                                       if data_flag_is_event else f"Region: {args.data_flag}")
                    data_2d_cases.append((
                        cr_data_collection, cr_region_label,
                        f"data_2d_{args.data_flag}_{suffix}"
                    ))
                for data_coll, region_label, canvas_name in data_2d_cases:
                    combined = loader.combine_data(data_coll)
                    if combined:
                        canvas, _ = plotter2d.plot_2d(
                            combined, canvas_name,
                            "Data", region_label,
                            sample_label_x_pos=0.59,
                            x_var=x_var, y_var=y_var, is_data=True
                        )
                        if canvas:
                            save_canvas(canvas, output_format, f_out, output_dir, plots_2d_subdir)

            # Return to parent directory
            if use_root_file:
                fs_dir.cd()
        
        # --- 1D Plots ---
        if '1d' in args.plots or 'all' in args.plots:
            print("  Generating 1D Plots...")
            plots_1d_subdir = f"{folder_name}/1D_plots"
            
            if use_root_file:
                plots_1d_dir = fs_dir.mkdir("1D_plots")
                plots_1d_dir.cd()
            
            for var_key in args.vars:
                conf = AnalysisConfig.VARIABLES.get(var_key)
                if not conf:
                    print(f"Warning: Variable {var_key} not defined in config. Skipping.")
                    continue
                
                short_name = conf['name']
                label = conf['label']
                nbins = conf['bins']
                xmin, xmax = conf['range']
                
                # 1. All Signals
                if current_sig_data:
                    c_sig = plotter1d.plot_collection(current_sig_data, short_name, label, nbins, xmin, xmax, collection_type="Signal", normalized=args.normalize, suffix=flag, final_state_label=fs_label_latex)
                    save_canvas(c_sig, output_format, f_out, output_dir, plots_1d_subdir)
                
                # 2. All Backgrounds
                if current_bg_data:
                    c_bg = plotter1d.plot_collection(current_bg_data, short_name, label, nbins, xmin, xmax, collection_type="Background", normalized=args.normalize, suffix=flag, final_state_label=fs_label_latex)
                    save_canvas(c_bg, output_format, f_out, output_dir, plots_1d_subdir)
                    
                # 3. Signal vs Net Background
                if current_sig_data and current_bg_combined:
                    c_comp, _, _ = plotter1d.plot_signals_vs_net_background(current_sig_data, current_bg_combined, short_name, label, nbins, xmin, xmax, args.normalize, suffix=flag, final_state_label=fs_label_latex)
                    save_canvas(c_comp, output_format, f_out, output_dir, plots_1d_subdir)
                    
            # Return to parent directory
            if use_root_file:
                fs_dir.cd()

        # --- CR Data vs SR Signal Plots ---
        do_cr_sig = ('cr_sig' in args.plots or 'all' in args.plots) and args.data_flag and all_data_files
        if do_cr_sig and current_sig_data:
            # cr_data_collection already computed above
            if not cr_data_collection:
                print(f"  Skipping CR-vs-SR plots: no data events passed '{args.data_flag}'")
            else:
                print(f"  Generating CR Data vs SR Signal Plots (data flag: {args.data_flag})...")
                cr_sig_subdir = f"{folder_name}/cr_vs_sr_plots"
                # Use formatted label for named flags, raw cut string for custom cuts
                cr_label = resolver.format_sv_label(args.data_flag) if data_flag_is_event else args.data_flag

                if use_root_file:
                    cr_sig_dir = fs_dir.mkdir("cr_vs_sr_plots")
                    cr_sig_dir.cd()

                for var_key in args.vars:
                    conf = AnalysisConfig.VARIABLES.get(var_key)
                    if not conf:
                        continue
                    # Skip MC-only variables — data CR cannot provide them
                    if conf.get('mc_only', False):
                        continue
                    short_name = conf['name']
                    label = conf['label']
                    nbins = conf['bins']
                    xmin, xmax = conf['range']

                    canvas = plotter1d.plot_cr_data_vs_sr_signal(
                        cr_data_collection, current_sig_data,
                        short_name, label, nbins, xmin, xmax,
                        cr_label=cr_label, suffix=flag,
                        final_state_label=fs_label_latex
                    )
                    save_canvas(canvas, output_format, f_out, output_dir, cr_sig_subdir)

                if use_root_file:
                    fs_dir.cd()


    if use_root_file:
        f_out.Close()
        print(f"\nDone! Plots saved to {output_path}")
    else:
        print(f"\nDone! Plots saved to {output_dir}/ directory in {output_format} format")

if __name__ == "__main__":
    main()
