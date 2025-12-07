#!/usr/bin/env python3
import argparse
import sys
import ROOT
from pathlib import Path
import cmsstyle as CMS

from src.style import StyleManager
from src.loader import DataLoader
from src.plotter import Plotter1D, Plotter2D, PlotterDataMC
from src.selections import FinalStateResolver
from src.utils import parse_signal_name, parse_background_name

from src.config import AnalysisConfig

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

def save_canvas(canvas, output_format, f_out=None, output_dir=None, subdir_path="", canvas_name=None):
    """
    Save canvas in the specified format.
    For ROOT: writes to ROOT file in current directory
    For PDF/PNG: saves to output_dir with subdir_path structure
    """
    if output_format == 'root':
        if f_out is not None:
            canvas.Write()
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
    parser.add_argument('--signal', nargs='+', required=True, help='Signal ROOT files')
    parser.add_argument('--background', nargs='+', required=True, help='Background ROOT files')
    parser.add_argument('--data', nargs='*', help='Data ROOT files (for data/MC comparison plots)')
    parser.add_argument('--output', default='standard_plots.root', help='Output ROOT file')
    parser.add_argument('--tree', default='kuSkimTree', help='Tree name')
    
    # Selections
    parser.add_argument('--flags', nargs='+', default=[
        'passNHad1SelectionSRTight',
        'passNLep1SelectionSRTight'
    ], help='List of Final State Flags or custom cut strings to process')
    
    # Plot Types
    parser.add_argument('--plots', nargs='+', choices=['1d', '2d', 'ratio', 'all'], default=['all'],
                       help='Types of plots to generate')
    
    # 1D Options
    parser.add_argument('--vars', nargs='+', default=['rjr_Ms', 'rjr_Rs'], 
                       help='Variables to plot in 1D mode (must be defined in src/config.py)')
    parser.add_argument('--normalize', action='store_true', help='Normalize 1D plots to area 1')
    
    # Analysis Options
    parser.add_argument('--lumi', type=float, default=400.0, help='Integrated luminosity in fb^-1 (default: 400)')
    parser.add_argument('--unblind', action='store_true', help='Bypass data blinding (shows data in all regions including signal regions)')
    
    # Output Format Options
    parser.add_argument('--format', choices=['root', 'pdf', 'png'], default='root', 
                       help='Output format: root (default), pdf, or png')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Handle unblinding warning if requested
    if args.unblind and args.data:
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
    style = StyleManager(luminosity=args.lumi)
    style.set_style()
    
    loader = DataLoader(args.tree, luminosity=args.lumi)
    plotter1d = Plotter1D(style)
    plotter2d = Plotter2D(style)
    plotter_datamc = PlotterDataMC(style)
    resolver = FinalStateResolver()
    
    # Load Data
    # Separate event flags from custom cut strings
    event_flags = [flag for flag in args.flags if is_event_flag(flag)]
    custom_cuts = [flag for flag in args.flags if not is_event_flag(flag)]
    
    print(f"Loading data for {len(event_flags)} event flags and {len(custom_cuts)} custom cuts...")
    
    # Use unified loader to load everything in one pass
    sig_data_map, custom_sig_data_map = loader.load_data_unified(args.signal, event_flags, custom_cuts)
    bg_data_map, custom_bg_data_map = loader.load_data_unified(args.background, event_flags, custom_cuts)
    
    # Load data files if provided
    data_data_map, custom_data_data_map = {}, {}
    if args.data:
        data_data_map, custom_data_data_map = loader.load_data_unified(args.data, event_flags, custom_cuts, is_data=True)
    
    # Output File (only for ROOT format)
    if use_root_file:
        f_out = ROOT.TFile(output_path, "RECREATE")
    else:
        f_out = None
    
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
        all_flags_and_cuts.append({
            'name': custom_region,
            'data_source': 'custom_cut',
            'sig_data': custom_sig_data_map.get(custom_region, {}),
            'bg_data': custom_bg_data_map.get(custom_region, {}),
            'show_region_label': False,
            'original_cut': original_cut
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
        fs_label_latex = resolver.format_sv_label(flag) if show_region_label else None
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
        
        # --- Data/MC Comparison Plots (FIRST to avoid palette interference) ---
        if args.data and ('ratio' in args.plots or 'all' in args.plots):
            print("  Generating Data/MC Comparison Plots...")
            datamc_subdir = f"{folder_name}/datamc_plots"
            
            if use_root_file:
                datamc_dir = fs_dir.mkdir("datamc_plots")
                datamc_dir.cd()
            
            # Get corresponding data for this flag/cut
            if item['data_source'] == 'event_flag':
                current_data_data = data_data_map.get(flag, {})
            else:
                # For custom cuts, find the matching region
                custom_region_idx = list(custom_sig_data_map.keys()).index(flag) if flag in custom_sig_data_map else -1
                if custom_region_idx >= 0:
                    custom_region_name = list(custom_data_data_map.keys())[custom_region_idx]
                    current_data_data = custom_data_data_map.get(custom_region_name, {})
                else:
                    current_data_data = {}
            
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
                              'HadronicSV_pOverE', 'HadronicSV_decayAngle', 'HadronicSV_cosTheta']
                datamc_vars.extend(hadSV_vars)
            elif "NLep" in flag and "NHad" not in flag:
                # Pure leptonic final state - use LeptonicSV variables  
                lepSV_vars = ['LeptonicSV_mass', 'LeptonicSV_dxy', 'LeptonicSV_dxySig',
                              'LeptonicSV_pOverE', 'LeptonicSV_decayAngle', 'LeptonicSV_cosTheta']
                datamc_vars.extend(lepSV_vars)
            elif "NHad" in flag and "NLep" in flag:
                # Combined final state - use BOTH HadronicSV and LeptonicSV variables
                hadSV_vars = ['HadronicSV_mass', 'HadronicSV_dxy', 'HadronicSV_dxySig', 
                              'HadronicSV_pOverE', 'HadronicSV_decayAngle', 'HadronicSV_cosTheta']
                lepSV_vars = ['LeptonicSV_mass', 'LeptonicSV_dxy', 'LeptonicSV_dxySig',
                              'LeptonicSV_pOverE', 'LeptonicSV_decayAngle', 'LeptonicSV_cosTheta']
                datamc_vars.extend(hadSV_vars)
                datamc_vars.extend(lepSV_vars)
            # else: for other flags (custom cuts, etc.) use only event-level variables
            
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
        
        # --- 2D Plots ---
        if '2d' in args.plots or 'all' in args.plots:
            print("  Generating 2D Plots...")
            plots_2d_subdir = f"{folder_name}/2D_plots"
            
            if use_root_file:
                plots_2d_dir = fs_dir.mkdir("2D_plots")
                plots_2d_dir.cd()
            
            # Signal 2D
            for fname, data in current_sig_data.items():
                sig_name = parse_signal_name(fname)
                sample_label_x_pos = 0.32 # Original logic for signal
                canvas, _ = plotter2d.plot_2d(data, f"sig_2d_{Path(fname).stem}_{flag}", sig_name, fs_label_latex, sample_label_x_pos=sample_label_x_pos)
                save_canvas(canvas, output_format, f_out, output_dir, plots_2d_subdir)
                
            # Background 2D (Individual)
            for fname, data in current_bg_data.items():
                bg_name = parse_background_name(fname)
                # Original logic for individual background
                sample_label_x_pos = 0.62 if "QCD" in bg_name else 0.69 
                canvas, _ = plotter2d.plot_2d(data, f"bg_2d_{Path(fname).stem}_{flag}", bg_name, fs_label_latex, sample_label_x_pos=sample_label_x_pos)
                save_canvas(canvas, output_format, f_out, output_dir, plots_2d_subdir)
                
            # Background 2D (Combined)
            if current_bg_combined:
                sample_label_x_pos = 0.59 # Original logic for combined background
                canvas, _ = plotter2d.plot_2d(current_bg_combined, f"bg_2d_total_{flag}", "Total Background", fs_label_latex, sample_label_x_pos=sample_label_x_pos)
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
        

    if use_root_file:
        f_out.Close()
        print(f"\nDone! Plots saved to {output_path}")
    else:
        print(f"\nDone! Plots saved to {output_dir}/ directory in {output_format} format")

if __name__ == "__main__":
    main()
