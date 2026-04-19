import uproot
import numpy as np
import concurrent.futures
import os
from src.selections import SelectionManager
from src.config import AnalysisConfig, AnalysisMode, ModeConfig


def _load_file_task(args):
    """Module-level worker for ProcessPoolExecutor — loads one file."""
    loader, file_path, branches, event_flags, custom_cuts, is_data = args
    return loader._load_one_file(file_path, branches, event_flags, custom_cuts, is_data)

class DataLoader:
    def __init__(self, tree_name='kuSkimTree', luminosity=400,
                 analysis_mode='uncompressed', isr_pt_cut=None):
        self.tree_name = tree_name
        self.luminosity = luminosity
        self.analysis_mode = analysis_mode
        self.isr_pt_cut = isr_pt_cut
        self.selection_manager = SelectionManager()
        self.loading_summary = {
            'data_types_loaded': set(),
            'event_flags': set(),
            'custom_cuts': set(),
            'files_processed': 0,
            'analysis_mode': analysis_mode,
            'isr_pt_cut': isr_pt_cut
        }

    def _track_loading(self, event_flags=None, custom_cuts=None, is_data=False, file_count=0):
        """Track what's being loaded for comprehensive summary."""
        self.loading_summary['data_types_loaded'].add('Data' if is_data else 'MC')
        if event_flags:
            self.loading_summary['event_flags'].update(event_flags)
        if custom_cuts:
            self.loading_summary['custom_cuts'].update(custom_cuts)
        self.loading_summary['files_processed'] += file_count

    def _get_branches_for_mode(self):
        """Get the list of branches to load based on analysis mode."""
        mode_config = ModeConfig.get(self.analysis_mode)

        # Common branches for all modes
        base_branches = [
            'evtFillWgt', 'SV_nHadronic', 'SV_nLeptonic', 'nSelPhotons', 'selCMet',
            # SV variables for data/MC comparisons
            'HadronicSV_mass', 'HadronicSV_dxy', 'HadronicSV_dxySig',
            'HadronicSV_pOverE', 'HadronicSV_decayAngle', 'HadronicSV_cosTheta',
            'HadronicSV_nTracks',
            'LeptonicSV_mass', 'LeptonicSV_dxy', 'LeptonicSV_dxySig',
            'LeptonicSV_pOverE', 'LeptonicSV_decayAngle', 'LeptonicSV_cosTheta',
            # Photon timing variables (Gen branches absent in data files; handled gracefully)
            'baseLinePhoton_WTimeSig',
            'baseLinePhoton_GenTimeSig',
            'baseLinePhoton_GenLabTimeSig'
        ]

        # Add mode-specific branches
        branches = base_branches + mode_config['branches']

        return branches
    
    def print_comprehensive_summary(self):
        """Print comprehensive summary after all loading is complete."""
        print(f"\n🎨 DATA LOADER SUMMARY:")
        print("=" * 60)
        print(f"    • Analysis mode: {self.analysis_mode}")
        if self.analysis_mode == AnalysisMode.COMPRESSED and self.isr_pt_cut is not None:
            print(f"    • ISR pT cut: {self.isr_pt_cut:.0f} GeV")
        print(f"    • Luminosity: {self.luminosity:.1f} fb⁻¹")
        print(f"    • Tree name: {self.tree_name}")
        print(f"    • Data types loaded: {', '.join(sorted(self.loading_summary['data_types_loaded']))}")
        print(f"    • Files processed: {self.loading_summary['files_processed']}")

        # Show baseline cuts (mode-aware)
        baseline_cuts = ["selCMet > 150", "evtFillWgt < 10"]
        baseline_cuts.extend([f"({flag} == 1)" for flag in self.selection_manager.flags])

        # Add mode-specific cuts
        if self.analysis_mode == AnalysisMode.UNCOMPRESSED:
            baseline_cuts.append("rjrPTS < 150")
        elif self.analysis_mode == AnalysisMode.COMPRESSED and self.isr_pt_cut is not None:
            baseline_cuts.append(f"rjrIsr_PtIsr >= {self.isr_pt_cut:.0f}")

        print(f"    • Baseline cuts: {', '.join(baseline_cuts)}")

        if self.loading_summary['event_flags']:
            print(f"    • Event flags: {', '.join(sorted(self.loading_summary['event_flags']))}")
        if self.loading_summary['custom_cuts']:
            print(f"    • Custom cuts: {', '.join(sorted(self.loading_summary['custom_cuts']))}")
        print("=" * 60)

    def load_data(self, file_paths, final_state_flags):
        """
        Loads data for multiple files and multiple final states.
        """
        self._track_loading(event_flags=final_state_flags, file_count=len(file_paths))
        # branches to load
        branches = [
            'rjr_Ms', 'rjr_Rs', 'evtFillWgt', 'SV_nHadronic', 'SV_nLeptonic',
            'nSelPhotons', 'selCMet', 'rjrPTS',
            # SV variables for data/MC comparisons
            'HadronicSV_mass', 'HadronicSV_dxy', 'HadronicSV_dxySig',
            'HadronicSV_pOverE', 'HadronicSV_decayAngle', 'HadronicSV_cosTheta',
            'HadronicSV_nTracks',
            'LeptonicSV_mass', 'LeptonicSV_dxy', 'LeptonicSV_dxySig',
            'LeptonicSV_pOverE', 'LeptonicSV_decayAngle', 'LeptonicSV_cosTheta'
        ]
        # Add flag branches
        branches.extend(final_state_flags)
        branches.extend(self.selection_manager.flags)
        
        all_data = {flag: {} for flag in final_state_flags}
        
        for file_path in file_paths:
            print(f"Loading {file_path}...")
            try:
                with uproot.open(file_path) as f:
                    if self.tree_name not in f:
                        print(f"  Warning: Tree {self.tree_name} not found in {file_path}")
                        continue
                        
                    tree = f[self.tree_name]
                    data = tree.arrays(branches, library='np')
                    
                    n_events = len(data['evtFillWgt'])
                    base_mask = np.ones(n_events, dtype=bool)
                    
                    # Scalar cuts using Config
                    base_mask &= (data['selCMet'] > AnalysisConfig.MET_CUT)
                    base_mask &= (data['evtFillWgt'] < AnalysisConfig.EVT_WGT_CUT)
                    
                    # Flag cuts (filters)
                    for flag in self.selection_manager.flags:
                        if flag in data:
                            base_mask &= (data[flag] == 1)
                        elif flag == 'hlt_flags':
                            # Try fallback expression for HLT flags
                            try:
                                hlt_mask = self._apply_hlt_fallback(tree)
                                base_mask &= hlt_mask
                            except Exception:
                                print(f"  Warning: High-Level Trigger (HLT) not found")
                        # No warning for other missing flags to keep output clean
                    
                    for fs_flag in final_state_flags:
                        if fs_flag not in data:
                            # print(f"  Warning: Flag {fs_flag} not found in {file_path}")
                            continue
                            
                        combined_mask = base_mask & (data[fs_flag] == 1)
                        
                        if np.sum(combined_mask) == 0:
                            continue
                            
                        ms_values = []
                        rs_values = []
                        weights = []
                        
                        indices = np.where(combined_mask)[0]
                        
                        for i in indices:
                            if (len(data['rjr_Ms'][i]) > 0 and 
                                len(data['rjr_Rs'][i]) > 0 and 
                                len(data['rjrPTS'][i]) > 0 and 
                                data['rjrPTS'][i][0] < AnalysisConfig.RJR_PTS_CUT): 
                                
                                # Use Scaling from Config
                                ms_val = data['rjr_Ms'][i][0] * AnalysisConfig.VARIABLES['rjr_Ms']['scale']
                                rs_val = data['rjr_Rs'][i][0] * AnalysisConfig.VARIABLES['rjr_Rs']['scale']
                                
                                ms_values.append(ms_val)
                                rs_values.append(rs_val)
                                weights.append(data['evtFillWgt'][i] * self.luminosity)
                        
                        if ms_values:
                            all_data[fs_flag][file_path] = {
                                'rjr_Ms': np.array(ms_values),
                                'rjr_Rs': np.array(rs_values),
                                'weights': np.array(weights)
                            }
                            # print(f"    [{fs_flag}] Loaded {len(ms_values)} events")

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
                
        return all_data

    def _apply_hlt_fallback(self, tree):
        """Apply HLT fallback using individual trigger branches."""
        try:
            # Load individual trigger branches
            trigger_branches = [
                'Trigger_PFMET120_PFMHT120_IDTight',
                'Trigger_PFMETNoMu120_PFMHTNoMu120_IDTight', 
                'Trigger_PFMET120_PFMHT120_IDTight_PFHT60',
                'Trigger_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60'
            ]
            
            # Load the trigger data
            trigger_data = tree.arrays(trigger_branches, library='np')
            
            # Apply OR logic: any trigger passes
            hlt_mask = np.zeros(len(trigger_data[trigger_branches[0]]), dtype=bool)
            for branch in trigger_branches:
                if branch in trigger_data:
                    hlt_mask |= (trigger_data[branch] == 1)
            
            return hlt_mask
            
        except Exception as e:
            raise Exception(f"HLT fallback failed: {e}")

    def load_data_unified(self, file_paths, event_flags, custom_cuts, is_data=False):
        """
        Unified loader that handles both event flags and custom cuts in one pass.
        Args:
            is_data: If True, treat as data files (no MC scaling)
        Returns: (event_flag_data, custom_cut_data)
        """
        self._track_loading(event_flags=event_flags, custom_cuts=custom_cuts, is_data=is_data, file_count=len(file_paths))
        # branches to load - use mode-aware branch selection
        branches = self._get_branches_for_mode()
        # Add flag branches, expanding '+'(AND) and '|'(OR) operators into components
        for flag in event_flags:
            for or_part in flag.split('|'):
                branches.extend(f.strip() for f in or_part.split('+'))
        branches.extend(self.selection_manager.flags)
        
        # Initialize data structures
        event_data = {flag: {} for flag in event_flags}
        custom_data = {f"CustomRegion{i+1}": {} for i in range(len(custom_cuts))}
        
        n_workers = min(os.cpu_count() or 4, len(file_paths), 8)

        if len(file_paths) > 1 and n_workers > 1:
            tasks = [(self, fp, branches, event_flags, custom_cuts, is_data) for fp in file_paths]
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                    results = list(executor.map(_load_file_task, tasks))
            except Exception as e:
                print(f"Warning: parallel loading failed ({e}), falling back to sequential")
                results = [self._load_one_file(fp, branches, event_flags, custom_cuts, is_data)
                           for fp in file_paths]
        else:
            results = [self._load_one_file(fp, branches, event_flags, custom_cuts, is_data)
                       for fp in file_paths]

        for file_path, file_event, file_custom in results:
            for flag, fdata in file_event.items():
                event_data[flag][file_path] = fdata
            for region, fdata in file_custom.items():
                custom_data[region][file_path] = fdata

        return event_data, custom_data

    def _load_one_file(self, file_path, branches, event_flags, custom_cuts, is_data):
        """Load and process a single file; returns (file_path, event_results, custom_results)."""
        event_result = {}
        custom_result = {}

        print(f"Loading {file_path}...")
        try:
            with uproot.open(file_path) as f:
                if self.tree_name not in f:
                    print(f"  Warning: Tree {self.tree_name} not found in {file_path}")
                    return file_path, event_result, custom_result

                tree = f[self.tree_name]
                available_branches = [b for b in branches if b in tree]
                cut_expr = (f"(selCMet > {AnalysisConfig.MET_CUT}) &"
                            f" (evtFillWgt < {AnalysisConfig.EVT_WGT_CUT})")
                data = tree.arrays(available_branches, cut=cut_expr, library='np')

                n_events = len(data['evtFillWgt'])
                base_mask = np.ones(n_events, dtype=bool)

                for flag in self.selection_manager.flags:
                    if flag in data:
                        base_mask &= (data[flag] == 1)

                # Process event flags ('|' = OR, '+' = AND, AND binds tighter)
                for fs_flag in event_flags:
                    or_parts = [p.strip() for p in fs_flag.split('|')]
                    flag_mask = np.zeros(n_events, dtype=bool)
                    all_missing = True

                    for or_part in or_parts:
                        sub_flags = [f.strip() for f in or_part.split('+')]
                        missing = [sf for sf in sub_flags if sf not in data]
                        if missing:
                            print(f"  Warning: Flag(s) {missing} not found in {file_path}"
                                  f" — skipping OR part '{or_part}'")
                            continue
                        all_missing = False
                        and_mask = np.ones(n_events, dtype=bool)
                        for sf in sub_flags:
                            and_mask &= (data[sf] == 1)
                        flag_mask |= and_mask

                    if all_missing:
                        continue

                    combined_mask = base_mask & flag_mask
                    n_flag = int(np.sum(flag_mask))
                    n_pass = int(np.sum(combined_mask))

                    if n_pass == 0:
                        print(f"  Warning: 0 events pass baseline cuts for '{fs_flag}' in {file_path} "
                              f"({n_flag} passed the flag(s) before baseline cuts)")
                        continue

                    extracted_vars = self._extract_values(data, combined_mask, is_data)
                    file_data = self._process_extracted_data(extracted_vars)

                    if file_data:
                        event_result[fs_flag] = file_data
                    else:
                        print(f"  Warning: Events passed '{fs_flag}' and baseline cuts but failed "
                              f"mode validation (e.g. rjrPTS cut) in {file_path}")

                # Process custom cuts
                for i, custom_cut in enumerate(custom_cuts):
                    custom_region_name = f"CustomRegion{i+1}"
                    try:
                        nSelPhotons = data.get("nSelPhotons", np.zeros(n_events))
                        SV_nHadronic = data.get("SV_nHadronic", np.zeros(n_events))
                        SV_nLeptonic = data.get("SV_nLeptonic", np.zeros(n_events))
                        selCMet = data.get("selCMet", np.zeros(n_events))

                        cut_variables = {
                            'nSelPhotons': nSelPhotons,
                            'SV_nHadronic': SV_nHadronic,
                            'SV_nLeptonic': SV_nLeptonic,
                            'selCMet': selCMet
                        }

                        if self.analysis_mode == AnalysisMode.COMPRESSED:
                            for isr_var in ['rjrIsr_Ms', 'rjrIsr_MsPerp', 'rjrIsr_PtIsr',
                                           'rjrIsr_RIsr', 'rjrIsr_Rs', 'rjrIsrPTS',
                                           'rjrIsr_nSVisObjects', 'rjrIsr_nIsrVisObjects']:
                                if isr_var in data:
                                    cut_variables[isr_var] = data[isr_var]

                        custom_mask = self._parse_simple_cut(custom_cut, cut_variables)
                        combined_mask = base_mask & custom_mask
                    except Exception as e:
                        print(f"  Warning: Failed to evaluate custom cut '{custom_cut}': {e}")
                        continue

                    if np.sum(combined_mask) == 0:
                        continue

                    extracted_vars = self._extract_values(data, combined_mask, is_data)
                    file_data = self._process_extracted_data(extracted_vars)

                    if file_data:
                        custom_result[custom_region_name] = file_data

        except Exception as e:
            print(f"  Error loading {file_path}: {e}")

        return file_path, event_result, custom_result

    def _extract_values(self, data, mask, is_data=False):
        """Helper method to extract values for both event flags and custom cuts."""
        # Initialize storage for all variables
        extracted_data = {}

        indices = np.where(mask)[0]

        for idx in indices:
            # Mode-specific validation
            passes_validation = False

            if self.analysis_mode == AnalysisMode.UNCOMPRESSED:
                # Uncompressed mode: require rjr_Ms, rjr_Rs, and rjrPTS < 150
                if ('rjr_Ms' in data and 'rjr_Rs' in data and 'rjrPTS' in data and
                    len(data['rjr_Ms'][idx]) > 0 and
                    len(data['rjr_Rs'][idx]) > 0 and
                    len(data['rjrPTS'][idx]) > 0 and
                    data['rjrPTS'][idx][0] < AnalysisConfig.RJR_PTS_CUT):
                    passes_validation = True
            else:
                # Compressed mode: require ISR variables and optionally apply ISR pT cut
                if 'rjrIsrPTS' in data:
                    if self.isr_pt_cut is not None:
                        # Apply ISR pT cut if specified (skip events below the cut)
                        if 'rjrIsr_PtIsr' in data and data['rjrIsr_PtIsr'][idx] >= self.isr_pt_cut:
                            passes_validation = True
                    else:
                        passes_validation = True

            if not passes_validation:
                continue

            # Get base event weight
            base_weight = 1.0 if is_data else data['evtFillWgt'][idx] * self.luminosity

            # Extract all configured variables
            for var_key, var_config in AnalysisConfig.VARIABLES.items():
                if var_key not in data:
                    continue

                # Skip MC-only variables when processing data
                if var_config.get('mc_only', False) and is_data:
                    continue

                if var_key not in extracted_data:
                    extracted_data[var_key] = []
                    if var_key != 'weights':  # Don't create weights array for weights key
                        extracted_data[f'{var_key}_weights'] = []

                if var_key in ['rjr_Ms', 'rjr_Rs']:
                    # Special case: rjr variables take element [0]
                    if len(data[var_key][idx]) > 0:
                        raw_val = data[var_key][idx][0]
                        scaled_val = raw_val * var_config['scale']

                        # Apply cross-cut on the paired RJR variable if defined
                        cross_cut = var_config.get('cross_cut')
                        if cross_cut:
                            other_branch, op, threshold = cross_cut
                            other_scale = AnalysisConfig.VARIABLES[other_branch]['scale']
                            if (other_branch in data and len(data[other_branch][idx]) > 0):
                                other_val = data[other_branch][idx][0] * other_scale
                                if not (other_val > threshold if op == '>' else other_val < threshold):
                                    continue

                        extracted_data[var_key].append(scaled_val)
                        extracted_data[f'{var_key}_weights'].append(base_weight)

                elif var_key.startswith('HadronicSV_') or var_key.startswith('LeptonicSV_'):
                    # SV variables: flatten jagged arrays - one entry per SV object
                    sv_array = data[var_key][idx]
                    for sv_val in sv_array:
                        scaled_val = sv_val * var_config['scale']
                        extracted_data[var_key].append(scaled_val)
                        extracted_data[f'{var_key}_weights'].append(base_weight)

                elif var_key.startswith('baseLinePhoton_'):
                    # Photon variables: flatten jagged arrays - one entry per photon object
                    photon_array = data[var_key][idx]
                    for ph_val in photon_array:
                        scaled_val = ph_val * var_config['scale']
                        extracted_data[var_key].append(scaled_val)
                        extracted_data[f'{var_key}_weights'].append(base_weight)

                elif not var_config['is_vector']:
                    # Scalar event-level variables (like selCMet, ISR variables)
                    raw_val = data[var_key][idx]
                    scaled_val = raw_val * var_config['scale']
                    extracted_data[var_key].append(scaled_val)
                    extracted_data[f'{var_key}_weights'].append(base_weight)

        # Convert all lists to numpy arrays
        for var_key in extracted_data:
            extracted_data[var_key] = np.array(extracted_data[var_key])

        return extracted_data

    def _process_extracted_data(self, extracted_vars):
        """Convert extracted variables to file data structure."""
        if not extracted_vars:
            return None

        # Check for mode-appropriate primary variable
        has_uncompressed_vars = 'rjr_Ms' in extracted_vars
        has_compressed_vars = 'rjrIsr_Ms' in extracted_vars or 'rjrIsr_PtIsr' in extracted_vars

        if not has_uncompressed_vars and not has_compressed_vars:
            return None

        # Create data structure with all variables and their specific weights
        file_data = {}
        for var_key in extracted_vars:
            if not var_key.endswith('_weights'):
                # Store the variable data
                file_data[var_key] = extracted_vars[var_key]
                # Store the variable-specific weights
                var_weights_key = f'{var_key}_weights'
                if var_weights_key in extracted_vars:
                    file_data[var_weights_key] = extracted_vars[var_weights_key]

        # Set default 'weights' for backward compatibility
        if 'rjr_Ms_weights' in extracted_vars:
            file_data['weights'] = extracted_vars['rjr_Ms_weights']
        elif 'rjrIsr_Ms_weights' in extracted_vars:
            file_data['weights'] = extracted_vars['rjrIsr_Ms_weights']
        elif 'rjrIsr_PtIsr_weights' in extracted_vars:
            file_data['weights'] = extracted_vars['rjrIsr_PtIsr_weights']

        return file_data

    def _parse_simple_cut(self, cut_string, variables):
        """
        Parse simple cut expressions without eval().
        Supports patterns like: 'var==value', 'var1==val1 && var2==val2'
        Handles &&, &, and || as logical operators.
        """
        import re

        cut_string = cut_string.strip()

        # Strip matching outer parentheses and recurse
        if cut_string.startswith('(') and cut_string.endswith(')'):
            # Verify they actually match (not e.g. "(a>1) | (b>1)")
            depth = 0
            matched = True
            for i, ch in enumerate(cut_string):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                if depth == 0 and i < len(cut_string) - 1:
                    matched = False
                    break
            if matched:
                return self._parse_simple_cut(cut_string[1:-1], variables)

        # Normalize logical operators: replace '&&' with ' & ' and '||' with ' | '
        normalized = cut_string.replace('&&', ' & ').replace('||', ' | ')

        # Split on OR first (lower precedence), respecting parentheses
        or_parts = self._split_respecting_parens(normalized, ' | ')
        if len(or_parts) > 1:
            result_mask = None
            for part in or_parts:
                part_mask = self._parse_simple_cut(part, variables)
                if result_mask is None:
                    result_mask = part_mask
                else:
                    result_mask = result_mask | part_mask
            return result_mask

        # Then split on AND, respecting parentheses
        and_parts = self._split_respecting_parens(normalized, ' & ')
        if len(and_parts) > 1:
            result_mask = None
            for part in and_parts:
                part_mask = self._parse_simple_cut(part, variables)
                if result_mask is None:
                    result_mask = part_mask
                else:
                    result_mask = result_mask & part_mask
            return result_mask

        # Single condition
        return self._evaluate_single_condition(cut_string.strip(), variables)

    def _split_respecting_parens(self, text, delimiter):
        """Split text on delimiter only when not inside parentheses."""
        parts = []
        depth = 0
        current = []
        i = 0
        while i < len(text):
            if text[i] == '(':
                depth += 1
                current.append(text[i])
                i += 1
            elif text[i] == ')':
                depth -= 1
                current.append(text[i])
                i += 1
            elif depth == 0 and text[i:i+len(delimiter)] == delimiter:
                parts.append(''.join(current).strip())
                current = []
                i += len(delimiter)
            else:
                current.append(text[i])
                i += 1
        parts.append(''.join(current).strip())
        return parts

    def _evaluate_single_condition(self, condition, variables):
        """
        Evaluate a single condition like 'nSelPhotons==1'
        """
        import re
        
        # Parse condition with regex ($ anchor ensures full match, no trailing text ignored)
        match = re.match(r'(\w+)\s*(==|!=|<=|>=|<|>)\s*(\d+(?:\.\d+)?)\s*$', condition)
        if not match:
            raise ValueError(f"Cannot parse condition: {condition}")
            
        var_name, operator, value_str = match.groups()
        
        if var_name not in variables:
            raise ValueError(f"Unknown variable: {var_name}")
            
        array = variables[var_name]
        value = float(value_str) if '.' in value_str else int(value_str)
        
        # Apply the operation
        if operator == '==':
            return array == value
        elif operator == '!=':
            return array != value
        elif operator == '<':
            return array < value
        elif operator == '>':
            return array > value
        elif operator == '<=':
            return array <= value
        elif operator == '>=':
            return array >= value
        else:
            raise ValueError(f"Unknown operator: {operator}")

    def combine_data(self, data_dict):
        """Combines data from multiple files (e.g. for total background)."""
        if not data_dict:
            return None

        # Get all keys from the first file's data
        first_data = next(iter(data_dict.values()))
        all_keys = list(first_data.keys())

        # Combine all variables that exist across all files
        combined = {}
        for key in all_keys:
            try:
                combined[key] = np.concatenate([d[key] for d in data_dict.values() if key in d])
            except (KeyError, ValueError):
                # Skip keys that don't exist in all files or can't be concatenated
                continue

        return combined if combined else None
