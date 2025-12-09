import uproot
import numpy as np
from src.selections import SelectionManager
from src.config import AnalysisConfig

class DataLoader:
    def __init__(self, tree_name='kuSkimTree', luminosity=400):
        self.tree_name = tree_name
        self.luminosity = luminosity
        self.selection_manager = SelectionManager()
        self.loading_summary = {
            'data_types_loaded': set(),
            'event_flags': set(),
            'custom_cuts': set(),
            'files_processed': 0
        }

    def _track_loading(self, event_flags=None, custom_cuts=None, is_data=False, file_count=0):
        """Track what's being loaded for comprehensive summary."""
        self.loading_summary['data_types_loaded'].add('Data' if is_data else 'MC')
        if event_flags:
            self.loading_summary['event_flags'].update(event_flags)
        if custom_cuts:
            self.loading_summary['custom_cuts'].update(custom_cuts)
        self.loading_summary['files_processed'] += file_count
    
    def print_comprehensive_summary(self):
        """Print comprehensive summary after all loading is complete."""
        print(f"\n🎨 DATA LOADER SUMMARY:")
        print("=" * 60)
        print(f"    • Luminosity: {self.luminosity:.1f} fb⁻¹")
        print(f"    • Tree name: {self.tree_name}")
        print(f"    • Data types loaded: {', '.join(sorted(self.loading_summary['data_types_loaded']))}")
        print(f"    • Files processed: {self.loading_summary['files_processed']}")
        
        # Show baseline cuts
        baseline_cuts = []
        baseline_cuts.extend(self.selection_manager.common_cuts)
        baseline_cuts.extend([f"({flag} == 1)" for flag in self.selection_manager.flags])
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
            'rjr_Ms', 'rjr_Rs', 'evtFillWgt', 'SV_nHadronic', 
            'selCMet', 'rjrPTS',
            # SV variables for data/MC comparisons
            'HadronicSV_mass', 'HadronicSV_dxy', 'HadronicSV_dxySig',
            'HadronicSV_pOverE', 'HadronicSV_decayAngle', 'HadronicSV_cosTheta',
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
        # branches to load
        branches = [
            'rjr_Ms', 'rjr_Rs', 'evtFillWgt', 'SV_nHadronic', 
            'selCMet', 'rjrPTS', 'nSelPhotons',
            # SV variables for data/MC comparisons
            'HadronicSV_mass', 'HadronicSV_dxy', 'HadronicSV_dxySig',
            'HadronicSV_pOverE', 'HadronicSV_decayAngle', 'HadronicSV_cosTheta',
            'LeptonicSV_mass', 'LeptonicSV_dxy', 'LeptonicSV_dxySig',
            'LeptonicSV_pOverE', 'LeptonicSV_decayAngle', 'LeptonicSV_cosTheta'
        ]
        # Add flag branches and selection manager flags
        branches.extend(event_flags)
        branches.extend(self.selection_manager.flags)
        
        # Initialize data structures
        event_data = {flag: {} for flag in event_flags}
        custom_data = {f"CustomRegion{i+1}": {} for i in range(len(custom_cuts))}
        
        for file_path in file_paths:
            print(f"Loading {file_path}...")
            try:
                with uproot.open(file_path) as f:
                    if self.tree_name not in f:
                        print(f"  Warning: Tree {self.tree_name} not found in {file_path}")
                        continue
                        
                    tree = f[self.tree_name]
                    # Get all available branches first
                    available_branches = [b for b in branches if b in tree]
                    data = tree.arrays(available_branches, library='np')
                    
                    n_events = len(data['evtFillWgt'])
                    base_mask = np.ones(n_events, dtype=bool)
                    
                    # Scalar cuts using Config
                    base_mask &= (data['selCMet'] > AnalysisConfig.MET_CUT)
                    base_mask &= (data['evtFillWgt'] < AnalysisConfig.EVT_WGT_CUT)
                    
                    # Flag cuts (filters)
                    for flag in self.selection_manager.flags:
                        if flag in data:
                            base_mask &= (data[flag] == 1)
                    
                    # Process event flags
                    for fs_flag in event_flags:
                        if fs_flag not in data:
                            continue
                            
                        combined_mask = base_mask & (data[fs_flag] == 1)
                        
                        if np.sum(combined_mask) == 0:
                            continue
                            
                        extracted_vars = self._extract_values(data, combined_mask, is_data)
                        file_data = self._process_extracted_data(extracted_vars)
                        
                        if file_data:
                            event_data[fs_flag][file_path] = file_data
                    
                    # Process custom cuts
                    for i, custom_cut in enumerate(custom_cuts):
                        custom_region_name = f"CustomRegion{i+1}"
                        try:
                            # Get the arrays we need
                            nSelPhotons = data.get("nSelPhotons", np.zeros(n_events))
                            SV_nHadronic = data.get("SV_nHadronic", np.zeros(n_events))
                            selCMet = data.get("selCMet", np.zeros(n_events))
                            
                            # Parse and evaluate the custom cut manually to avoid eval() issues
                            custom_mask = self._parse_simple_cut(custom_cut, {
                                'nSelPhotons': nSelPhotons,
                                'SV_nHadronic': SV_nHadronic, 
                                'selCMet': selCMet
                            })
                            
                            combined_mask = base_mask & custom_mask
                        except Exception as e:
                            print(f"  Warning: Failed to evaluate custom cut '{custom_cut}': {e}")
                            continue
                        
                        if np.sum(combined_mask) == 0:
                            continue
                            
                        extracted_vars = self._extract_values(data, combined_mask, is_data)
                        file_data = self._process_extracted_data(extracted_vars)
                        
                        if file_data:
                            custom_data[custom_region_name][file_path] = file_data
                            
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
                
        return event_data, custom_data

    def _extract_values(self, data, mask, is_data=False):
        """Helper method to extract values for both event flags and custom cuts."""
        # Initialize storage for all variables
        extracted_data = {}
        
        indices = np.where(mask)[0]
        
        for idx in indices:
            if (len(data['rjr_Ms'][idx]) > 0 and 
                len(data['rjr_Rs'][idx]) > 0 and 
                len(data['rjrPTS'][idx]) > 0 and 
                data['rjrPTS'][idx][0] < AnalysisConfig.RJR_PTS_CUT): 
                
                # Get base event weight
                base_weight = 1.0 if is_data else data['evtFillWgt'][idx] * self.luminosity
                
                # Extract all configured variables
                for var_key, var_config in AnalysisConfig.VARIABLES.items():
                    if var_key not in data:
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
                            extracted_data[var_key].append(scaled_val)
                            extracted_data[f'{var_key}_weights'].append(base_weight)
                    
                    elif var_key.startswith('HadronicSV_') or var_key.startswith('LeptonicSV_'):
                        # SV variables: flatten jagged arrays - one entry per SV object
                        sv_array = data[var_key][idx]
                        for sv_val in sv_array:
                            scaled_val = sv_val * var_config['scale']
                            extracted_data[var_key].append(scaled_val)
                            extracted_data[f'{var_key}_weights'].append(base_weight)
                    
                    elif not var_config['is_vector']:
                        # Scalar event-level variables (like selCMet)
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
        if not extracted_vars or 'rjr_Ms' not in extracted_vars:
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
        
        # For backward compatibility with existing code, use rjr_Ms weights as default 'weights'
        if 'rjr_Ms_weights' in extracted_vars:
            file_data['weights'] = extracted_vars['rjr_Ms_weights']
        
        return file_data

    def _parse_simple_cut(self, cut_string, variables):
        """
        Parse simple cut expressions without eval().
        Supports patterns like: 'var==value', 'var1==val1 & var2==val2'
        """
        import re
        
        # Handle the specific patterns we expect
        if ' & ' in cut_string:
            # Split on & and evaluate each part
            parts = cut_string.split(' & ')
            result_mask = None
            
            for part in parts:
                part_mask = self._evaluate_single_condition(part.strip(), variables)
                if result_mask is None:
                    result_mask = part_mask
                else:
                    result_mask = result_mask & part_mask
            return result_mask
        else:
            # Single condition
            return self._evaluate_single_condition(cut_string, variables)
    
    def _evaluate_single_condition(self, condition, variables):
        """
        Evaluate a single condition like 'nSelPhotons==1'
        """
        import re
        
        # Parse condition with regex
        match = re.match(r'(\w+)\s*(==|!=|<=|>=|<|>)\s*(\d+(?:\.\d+)?)', condition)
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
            
        ms_combined = np.concatenate([d['rjr_Ms'] for d in data_dict.values()])
        rs_combined = np.concatenate([d['rjr_Rs'] for d in data_dict.values()])
        weights_combined = np.concatenate([d['weights'] for d in data_dict.values()])
        
        return {
            'rjr_Ms': ms_combined,
            'rjr_Rs': rs_combined,
            'weights': weights_combined
        }
