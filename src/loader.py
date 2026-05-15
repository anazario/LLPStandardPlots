import uproot
import numpy as np
from dataclasses import dataclass
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else _TqdmDummy()

    class _TqdmDummy:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def update(self, n=1): pass
from src.selections import SelectionManager
from src.config import AnalysisConfig, AnalysisMode, ModeConfig



def _merge_chunks(chunks):
    """Concatenate a list of extracted_vars dicts into one."""
    if not chunks:
        return {}
    merged = {}
    for key in chunks[0]:
        arrays = [c[key] for c in chunks if key in c and len(c[key]) > 0]
        merged[key] = np.concatenate(arrays) if arrays else np.array([])
    return merged


@dataclass
class _CutEvalResult:
    """Intermediate result for custom-cut evaluation."""
    kind: str
    values: object
    collection: str = None
    object_masks: object = None


class DataLoader:
    _PROMPT_PHO_CR_COMPRESSED_GE1 = 'PROMPT_PHO_CR_COMPRESSED_GE1'
    _PROMPT_PHO_CR_UNCOMPRESSED_GE1 = 'PROMPT_PHO_CR_UNCOMPRESSED_GE1'

    def __init__(self, tree_name='kuSkimTree', luminosity=400,
                 analysis_mode='uncompressed', isr_pt_cut=None, n_workers=1, verbose=False):
        self.tree_name = tree_name
        self.luminosity = luminosity
        self.analysis_mode = analysis_mode
        self.isr_pt_cut = isr_pt_cut
        self.n_workers = max(1, int(n_workers))
        self.verbose = verbose
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
            # Photon variables (Gen branches absent in data files; handled gracefully)
            'nBaseLinePhotons',
            'baseLinePhoton_WTimeSig',
            'baseLinePhoton_Pt',
            'baseLinePhoton_Eta',
            'baseLinePhoton_beamHaloCNNScore',
            'baseLinePhoton_isoANNScore',
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
        baseline_cuts.extend([f"({flag} == 0)" for flag in self.selection_manager.inverted_flags])

        # Add mode-specific cuts
        if self.analysis_mode == AnalysisMode.UNCOMPRESSED:
            baseline_cuts.append("rjrPTS < 150")
        elif self.analysis_mode == AnalysisMode.COMPRESSED:
            baseline_cuts.append("rjrIsr_nSVisObjects > 0")
            if self.isr_pt_cut is not None:
                baseline_cuts.append(f"per-variable PtISR N-1 cut >= {self.isr_pt_cut:.0f}")
            baseline_cuts.append(f"per-variable RISR N-1 cut >= {AnalysisConfig.ISR_RISR_CUT:.1f}")

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
        branches.extend(self.selection_manager.inverted_flags)
        
        all_data = {flag: {} for flag in final_state_flags}
        
        for file_path in file_paths:
            if self.verbose:
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
                    for flag in self.selection_manager.inverted_flags:
                        if flag in data:
                            base_mask &= (data[flag] == 0)
                    
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
        Uses self.n_workers > 1 for file-level parallelism via ProcessPoolExecutor.
        """
        self._track_loading(event_flags=event_flags, custom_cuts=custom_cuts, is_data=is_data, file_count=len(file_paths))
        branches = self._get_branches_for_mode()
        branches.extend(self._branches_for_custom_cuts(custom_cuts))
        for flag in event_flags:
            for or_part in flag.split('|'):
                branches.extend(f.strip() for f in or_part.split('+'))
        branches.extend(self.selection_manager.flags)
        branches.extend(self.selection_manager.inverted_flags)
        branches = list(dict.fromkeys(branches))

        event_data = {flag: {} for flag in event_flags}
        custom_data = {f"CustomRegion{i+1}": {} for i in range(len(custom_cuts))}

        def _merge_result(file_path, file_event, file_custom):
            for flag, fdata in file_event.items():
                event_data[flag][file_path] = fdata
            for region, fdata in file_custom.items():
                custom_data[region][file_path] = fdata

        if self.n_workers > 1 and len(file_paths) > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            if self.verbose:
                print(f"  Parallel loading: {len(file_paths)} files across {self.n_workers} workers")
            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                futures = {
                    pool.submit(self._load_one_file, fp, branches, event_flags, custom_cuts, is_data): fp
                    for fp in file_paths
                }
                with tqdm(total=len(file_paths), unit="file", desc="Loading") as pbar:
                    for fut in as_completed(futures):
                        fp_done, file_event, file_custom = fut.result()
                        _merge_result(fp_done, file_event, file_custom)
                        pbar.update(1)
        else:
            import gc, ctypes

            def _trim_heap():
                gc.collect()
                try:
                    ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
                except Exception:
                    pass

            for fp in tqdm(file_paths, unit="file", desc="Loading"):
                file_path, file_event, file_custom = self._load_one_file(
                    fp, branches, event_flags, custom_cuts, is_data)
                _merge_result(file_path, file_event, file_custom)
                _trim_heap()

        return event_data, custom_data

    # Branches we know are scalar (one value per event) and safe to push into
    # uproot's C-level cut expression.  Jagged branches cannot be used there.
    _KNOWN_SCALAR_BRANCHES = {
        'SV_nHadronic', 'SV_nLeptonic', 'nSelPhotons', 'selCMet', 'evtFillWgt',
        'nBaseLinePhotons',
        'rjrIsr_Ms', 'rjrIsr_MsPerp', 'rjrIsr_PtIsr', 'rjrIsr_RIsr', 'rjrIsr_Rs',
        'rjrIsrPTS', 'rjrIsrSdphiBV', 'rjrIsr_nSVisObjects', 'rjrIsr_nIsrVisObjects',
        'rjr_Ms', 'rjr_Rs', 'rjrPTS',
    }

    _CUSTOM_CUT_FUNCTIONS = {'any', 'all', 'count', 'lead', 'abs'}

    def _branches_for_custom_cuts(self, custom_cuts):
        """Return additional tree branches referenced by custom cut strings."""
        if not custom_cuts:
            return []

        known_branches = (
            set(AnalysisConfig.VARIABLES) |
            self._KNOWN_SCALAR_BRANCHES |
            set(self.selection_manager.flags) |
            set(self.selection_manager.inverted_flags)
        )
        branches = []
        for cut in custom_cuts:
            if self._is_named_custom_cut(cut):
                continue
            for token in self._extract_cut_tokens(cut):
                if token in known_branches:
                    branches.append(token)
        return branches

    @classmethod
    def _extract_cut_tokens(cls, cut_string):
        import re
        return {
            tok for tok in re.findall(r'\b[A-Za-z_]\w*\b', cut_string)
            if tok not in cls._CUSTOM_CUT_FUNCTIONS
        }

    def _is_named_custom_cut(self, cut_name):
        return cut_name in {
            self._PROMPT_PHO_CR_COMPRESSED_GE1,
            self._PROMPT_PHO_CR_UNCOMPRESSED_GE1,
        }

    def _evaluate_named_custom_cut(self, cut_name, chunk, n_events):
        """Evaluate named analysis cuts that need vector-aware object logic."""
        if cut_name == self._PROMPT_PHO_CR_COMPRESSED_GE1:
            return self._prompt_photon_cr_compressed_ge1(chunk, n_events)
        if cut_name == self._PROMPT_PHO_CR_UNCOMPRESSED_GE1:
            return self._prompt_photon_cr_uncompressed_ge1(chunk, n_events)
        return None

    def _missing_named_cut_branches(self, cut_name, available_branches):
        if cut_name == self._PROMPT_PHO_CR_COMPRESSED_GE1:
            required = {
                'nBaseLinePhotons',
                'baseLinePhoton_WTimeSig',
                'baseLinePhoton_isoANNScore',
            }
        elif cut_name == self._PROMPT_PHO_CR_UNCOMPRESSED_GE1:
            required = {
                'nBaseLinePhotons',
                'baseLinePhoton_WTimeSig',
                'baseLinePhoton_isoANNScore',
                'baseLinePhoton_Pt',
                'baseLinePhoton_Eta',
            }
        else:
            return set()
        return required - set(available_branches)

    @staticmethod
    def _event_photon_arrays(chunk, idx, require_pt_eta=True):
        required = [
            'nBaseLinePhotons',
            'baseLinePhoton_WTimeSig',
            'baseLinePhoton_isoANNScore',
        ]
        if require_pt_eta:
            required.extend(['baseLinePhoton_Pt', 'baseLinePhoton_Eta'])
        if any(name not in chunk for name in required):
            return None

        n_pho = int(chunk['nBaseLinePhotons'][idx])
        if n_pho < 1:
            return None

        times = np.asarray(chunk['baseLinePhoton_WTimeSig'][idx], dtype=float)
        iso = np.asarray(chunk['baseLinePhoton_isoANNScore'][idx], dtype=float)
        if not require_pt_eta:
            if min(len(times), len(iso)) < n_pho:
                return None
            return n_pho, times[:n_pho], iso[:n_pho], None, None

        pts = np.asarray(chunk['baseLinePhoton_Pt'][idx], dtype=float)
        etas = np.asarray(chunk['baseLinePhoton_Eta'][idx], dtype=float)
        if min(len(times), len(iso), len(pts), len(etas)) < n_pho:
            return None

        return n_pho, times[:n_pho], iso[:n_pho], pts[:n_pho], etas[:n_pho]

    def _prompt_photon_cr_compressed_ge1(self, chunk, n_events):
        """Compressed prompt-photon CR: prompt timing and no tight ANN photons."""
        mask = np.zeros(n_events, dtype=bool)
        for idx in range(n_events):
            values = self._event_photon_arrays(chunk, idx, require_pt_eta=False)
            if values is None:
                continue
            _, times, iso, _, _ = values
            mask[idx] = np.all(np.isfinite(times)) and np.all(np.isfinite(iso)) and (
                np.all(np.abs(times) < 2.5) and np.all(iso < 0.96)
            )
        return mask

    def _prompt_photon_cr_uncompressed_ge1(self, chunk, n_events):
        """Uncompressed prompt-photon CR using BigGuy pt-dependent ANN windows."""
        mask = np.zeros(n_events, dtype=bool)
        for idx in range(n_events):
            values = self._event_photon_arrays(chunk, idx)
            if values is None:
                continue
            n_pho, times, iso, pts, etas = values
            if not (
                np.all(np.isfinite(times)) and np.all(np.isfinite(iso)) and
                np.all(np.isfinite(pts)) and np.all(np.isfinite(etas))
            ):
                continue
            if not np.all(np.abs(times) < 2.5):
                continue
            if not np.all(np.abs(etas) < 1.479):
                continue

            if n_pho == 1:
                high_pt = pts > 100
                lower = np.where(high_pt, -0.000198 * pts + 0.7698, 0.75)
                upper = np.where(high_pt, -0.000198 * pts + 1.0188, 0.999)
            elif n_pho == 2:
                high_pt = pts > 100
                lower = np.where(high_pt, -0.0001 * pts + 0.76, 0.75)
                upper = np.where(high_pt, -0.0001 * pts + 0.96, 0.95)
            else:
                continue

            mask[idx] = np.all((iso >= lower) & (iso < upper))
        return mask

    def _build_scalar_prefilter(self, custom_cuts, available_branches, tree):
        """
        Extract scalar necessary-conditions from custom cuts and return a combined
        uproot-compatible filter string.

        For each custom cut, split on top-level OR, extract scalar sub-conditions
        from each AND-clause, then reassemble as OR of AND-clauses.  The result is
        a necessary (but not sufficient) condition: any event failing it is
        guaranteed to fail all custom cuts, so it can be dropped before entering
        Python.
        """
        import re

        # Only use scalar branches that are actually present in this tree.
        scalar_branches = self._KNOWN_SCALAR_BRANCHES & set(available_branches)
        if not scalar_branches:
            return None

        cond_re = re.compile(
            r'\b(' + '|'.join(re.escape(b) for b in sorted(scalar_branches, key=len, reverse=True)) +
            r')\s*(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?)')

        per_cut_filters = []
        for cut in custom_cuts:
            if self._is_named_custom_cut(cut):
                continue
            normalized = cut.replace('&&', ' & ').replace('||', ' | ')
            or_clauses = self._split_respecting_parens(normalized, ' | ')

            clause_filters = []
            for clause in or_clauses:
                scalar_conds = cond_re.findall(clause)
                if scalar_conds:
                    clause_filters.append(
                        ' & '.join(f'({v} {op} {val})' for v, op, val in scalar_conds))

            if clause_filters:
                per_cut_filters.append('(' + ' | '.join(f'({c})' for c in clause_filters) + ')')

        return ' | '.join(per_cut_filters) if per_cut_filters else None

    @staticmethod
    def _rss_mb():
        """Return current process RSS in MB (Linux /proc; falls back to 0)."""
        try:
            with open('/proc/self/status') as fh:
                for line in fh:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) / 1024
        except Exception:
            pass
        return 0

    def _load_one_file(self, file_path, branches, event_flags, custom_cuts, is_data):
        """Load and process a single file in chunks to bound memory usage."""
        CHUNK_SIZE = 100_000

        event_result = {}
        custom_result = {}

        if self.verbose:
            print(f"Loading {file_path}...  [RSS {self._rss_mb():.0f} MB]")
        try:
            with uproot.open(file_path) as f:
                if self.tree_name not in f:
                    print(f"  Warning: Tree {self.tree_name} not found in {file_path}")
                    return file_path, event_result, custom_result

                tree = f[self.tree_name]
                n_entries = tree.num_entries
                tree_keys = set(tree.keys())
                available_branches = [b for b in branches if b in tree_keys]
                # Warn once if requested photon branches are missing from this tree
                _pho_requested = {'baseLinePhoton_beamHaloCNNScore', 'baseLinePhoton_WTimeSig',
                                  'baseLinePhoton_Pt', 'baseLinePhoton_Eta',
                                  'baseLinePhoton_isoANNScore', 'nBaseLinePhotons'}
                _pho_missing = _pho_requested - tree_keys
                if _pho_missing and custom_cuts and self.verbose:
                    print(f"  Note: photon branch(es) absent from tree: {sorted(_pho_missing)}")
                    print(f"  Available photon-like keys: {sorted(k for k in tree_keys if 'oto' in k or 'Pho' in k)[:10]}")
                if self.verbose:
                    for custom_cut in custom_cuts:
                        missing_named = self._missing_named_cut_branches(custom_cut, available_branches)
                        if missing_named:
                            print(f"  Warning: named cut '{custom_cut}' cannot pass; missing branch(es): {sorted(missing_named)}")
                cut_expr = (f"(selCMet > {AnalysisConfig.MET_CUT}) &"
                            f" (evtFillWgt < {AnalysisConfig.EVT_WGT_CUT})")
                baseline_flag_cuts = []
                missing_baseline_flags = []
                for flag in self.selection_manager.flags:
                    if flag in available_branches:
                        baseline_flag_cuts.append(f"({flag} == 1)")
                    else:
                        missing_baseline_flags.append(flag)
                for flag in self.selection_manager.inverted_flags:
                    if flag in available_branches:
                        baseline_flag_cuts.append(f"({flag} == 0)")
                    else:
                        missing_baseline_flags.append(flag)
                if baseline_flag_cuts:
                    cut_expr += " & " + " & ".join(baseline_flag_cuts)
                if self.verbose and missing_baseline_flags:
                    print(f"  Note: baseline flag branch(es) absent from tree: {missing_baseline_flags}")

                scalar_prefilter = self._build_scalar_prefilter(
                    custom_cuts, available_branches, tree)
                if scalar_prefilter:
                    cut_expr = f"({cut_expr}) & ({scalar_prefilter})"

                if self.analysis_mode == AnalysisMode.COMPRESSED:
                    if 'rjrIsr_nSVisObjects' in available_branches:
                        cut_expr += " & (rjrIsr_nSVisObjects > 0)"

                if self.verbose:
                    print(f"  tree entries: {n_entries:,}  |  uproot cut: {cut_expr}")

                event_chunks = {flag: [] for flag in event_flags}
                custom_chunks = {f"CustomRegion{i+1}": [] for i in range(len(custom_cuts))}
                flag_counts  = {flag: 0 for flag in event_flags}
                pass_counts  = {flag: 0 for flag in event_flags}

                import gc, ctypes
                def _trim():
                    gc.collect()
                    try:
                        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
                    except Exception:
                        pass

                total_loaded = 0
                total_base   = 0
                custom_pass  = [0] * len(custom_cuts)
                custom_stored_events = [0] * len(custom_cuts)
                chunk_count  = 0

                for chunk in tree.iterate(available_branches, cut=cut_expr,
                                          library='np', step_size=CHUNK_SIZE):
                    n_events = len(chunk['evtFillWgt'])
                    total_loaded += n_events
                    base_mask = np.ones(n_events, dtype=bool)

                    for flag in self.selection_manager.flags:
                        if flag in chunk:
                            base_mask &= (chunk[flag] == 1)
                    for flag in self.selection_manager.inverted_flags:
                        if flag in chunk:
                            base_mask &= (chunk[flag] == 0)
                    total_base += int(np.sum(base_mask))

                    # Process event flags ('|' = OR, '+' = AND)
                    for fs_flag in event_flags:
                        or_parts = [p.strip() for p in fs_flag.split('|')]
                        flag_mask = np.zeros(n_events, dtype=bool)
                        all_missing = True

                        for or_part in or_parts:
                            sub_flags = [f.strip() for f in or_part.split('+')]
                            if any(sf not in chunk for sf in sub_flags):
                                continue
                            all_missing = False
                            and_mask = np.ones(n_events, dtype=bool)
                            for sf in sub_flags:
                                and_mask &= (chunk[sf] == 1)
                            flag_mask |= and_mask

                        if all_missing:
                            continue

                        combined_mask = base_mask & flag_mask
                        flag_counts[fs_flag] += int(np.sum(flag_mask))
                        pass_counts[fs_flag] += int(np.sum(combined_mask))

                        if np.sum(combined_mask) == 0:
                            continue

                        extracted_vars = self._extract_values(chunk, combined_mask, is_data)
                        if extracted_vars:
                            event_chunks[fs_flag].append(extracted_vars)

                    # Process custom cuts
                    for i, custom_cut in enumerate(custom_cuts):
                        custom_region_name = f"CustomRegion{i+1}"
                        try:
                            named_mask = self._evaluate_named_custom_cut(custom_cut, chunk, n_events)
                            if named_mask is not None:
                                custom_mask = named_mask
                                selected_object_masks = None
                            else:
                                cut_variables = self._build_custom_cut_variables(chunk, n_events)
                                custom_mask, selected_object_masks = self._evaluate_custom_cut_with_objects(
                                    custom_cut, cut_variables)
                            combined_mask = base_mask & custom_mask
                        except Exception as e:
                            print(f"  Warning: Failed to evaluate custom cut '{custom_cut}': {e}")
                            continue

                        n_pass = int(np.sum(combined_mask))
                        custom_pass[i] += n_pass
                        if n_pass == 0:
                            continue
                        extracted_vars = self._extract_values(
                            chunk, combined_mask, is_data,
                            selected_object_masks=selected_object_masks)
                        if extracted_vars:
                            n_stored = max((len(v) for v in extracted_vars.values()), default=0)
                            custom_stored_events[i] += n_stored
                            custom_chunks[custom_region_name].append(extracted_vars)

                    chunk_count += 1
                    _trim()

                # Per-file summary
                if self.verbose:
                    print(f"  loaded {total_loaded:,} evts after uproot cut  |  "
                          f"{total_base:,} pass base mask  |  RSS {self._rss_mb():.0f} MB")
                    for i, cut in enumerate(custom_cuts):
                        acc_mb = 0
                        region = f"CustomRegion{i+1}"
                        for chunk_vars in custom_chunks[region]:
                            acc_mb += sum(a.nbytes for a in chunk_vars.values()) / 1e6
                        print(f"  custom cut {i+1}: {custom_pass[i]:,} events pass cut  |  "
                              f"{custom_stored_events[i]:,} stored entries  |  "
                              f"accumulated {acc_mb:.1f} MB in custom_chunks")

                # Merge chunks
                for fs_flag in event_flags:
                    if pass_counts[fs_flag] == 0:
                        print(f"  Warning: 0 events pass baseline cuts for '{fs_flag}' in {file_path} "
                              f"({flag_counts[fs_flag]} passed the flag(s) before baseline cuts)")
                        continue
                    if not event_chunks[fs_flag]:
                        continue
                    file_data = self._process_extracted_data(_merge_chunks(event_chunks[fs_flag]))
                    if file_data:
                        event_result[fs_flag] = file_data
                    else:
                        print(f"  Warning: Events passed '{fs_flag}' and baseline cuts but failed "
                              f"mode validation in {file_path}")

                for i in range(len(custom_cuts)):
                    region = f"CustomRegion{i+1}"
                    if not custom_chunks[region]:
                        continue
                    file_data = self._process_extracted_data(_merge_chunks(custom_chunks[region]))
                    if file_data:
                        custom_result[region] = file_data

        except Exception as e:
            print(f"  Error loading {file_path}: {e}")

        return file_path, event_result, custom_result

    def _passes_compressed_plane_cuts(self, var_key, data, idx):
        """
        Apply compressed PtISR/RISR cuts per plotted variable.

        PtISR is plotted with the RISR cut only, RISR is plotted with the PtISR
        cut only, and all other variables get both plane cuts.
        """
        if self.analysis_mode != AnalysisMode.COMPRESSED:
            return True

        if 'rjrIsr_PtIsr' not in data or 'rjrIsr_RIsr' not in data:
            return False

        passes_pt = (
            self.isr_pt_cut is None or
            data['rjrIsr_PtIsr'][idx] >= self.isr_pt_cut
        )
        passes_risr = data['rjrIsr_RIsr'][idx] >= AnalysisConfig.ISR_RISR_CUT

        if var_key == 'rjrIsr_PtIsr':
            return passes_risr
        if var_key == 'rjrIsr_RIsr':
            return passes_pt
        return passes_pt and passes_risr

    def _extract_values(self, data, mask, is_data=False, selected_object_masks=None):
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
                # Compressed mode: require ISR plane variables and nSVisObjects.
                # PtISR/RISR cuts are applied per plotted variable for N-1 behavior.
                if ('rjrIsr_PtIsr' in data and 'rjrIsr_RIsr' in data and
                        'rjrIsr_nSVisObjects' in data):
                    passes_nsv = data['rjrIsr_nSVisObjects'][idx] > 0
                    passes_validation = passes_nsv

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

                if not self._passes_compressed_plane_cuts(var_key, data, idx):
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
                    sv_array = data[var_key][idx]
                    collection = 'HadronicSV' if var_key.startswith('HadronicSV_') else 'LeptonicSV'
                    selected_mask = self._selected_mask_for_event(
                        selected_object_masks, collection, idx, len(sv_array))
                    if selected_mask is not None:
                        for sv_val in np.asarray(sv_array)[selected_mask]:
                            scaled_val = sv_val * var_config['scale']
                            extracted_data[var_key].append(scaled_val)
                            extracted_data[f'{var_key}_weights'].append(base_weight)
                    elif self.analysis_mode == AnalysisMode.COMPRESSED:
                        # Extract only the leading SV to avoid per-event array flattening
                        # on large data files; consistent with custom-cut evaluation.
                        if len(sv_array) > 0:
                            scaled_val = float(sv_array[0]) * var_config['scale']
                            extracted_data[var_key].append(scaled_val)
                            extracted_data[f'{var_key}_weights'].append(base_weight)
                    else:
                        for sv_val in sv_array:
                            scaled_val = sv_val * var_config['scale']
                            extracted_data[var_key].append(scaled_val)
                            extracted_data[f'{var_key}_weights'].append(base_weight)

                elif var_key.startswith('baseLinePhoton_'):
                    photon_array = data[var_key][idx]
                    selected_mask = self._selected_mask_for_event(
                        selected_object_masks, 'baseLinePhoton', idx, len(photon_array))
                    photon_values = (np.asarray(photon_array)[selected_mask]
                                     if selected_mask is not None else photon_array)
                    for ph_val in photon_values:
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

    def _build_custom_cut_variables(self, chunk, n_events):
        """Build the variable namespace used by the custom-cut evaluator."""
        variables = {name: values for name, values in chunk.items()}

        scalar_defaults = {
            'nSelPhotons': np.zeros(n_events, dtype=np.int32),
            'SV_nHadronic': np.zeros(n_events, dtype=np.int32),
            'SV_nLeptonic': np.zeros(n_events, dtype=np.int32),
            'nBaseLinePhotons': np.zeros(n_events, dtype=np.int32),
            'selCMet': np.zeros(n_events, dtype=float),
            'evtFillWgt': np.zeros(n_events, dtype=float),
        }
        for name, default in scalar_defaults.items():
            variables.setdefault(name, default)

        return variables

    @staticmethod
    def _selected_mask_for_event(selected_object_masks, collection, idx, expected_len):
        if not selected_object_masks or collection not in selected_object_masks:
            return None
        if idx >= len(selected_object_masks[collection]):
            return None

        selected = np.asarray(selected_object_masks[collection][idx], dtype=bool)
        if len(selected) != expected_len:
            n = min(len(selected), expected_len)
            padded = np.zeros(expected_len, dtype=bool)
            if n:
                padded[:n] = selected[:n]
            selected = padded
        return selected

    def _parse_simple_cut(self, cut_string, variables):
        """
        Parse custom cut expressions without eval().

        Vector branches are evaluated object-by-object.  Conditions joined by
        AND on the same object collection must pass on the same object; the
        collection mask is then reduced to an event mask with implicit any().
        Explicit reducers are also supported:
          any(expr), all(expr), count(expr) >= N, lead(var) > value.
        """
        result = self._parse_cut_expr(cut_string, variables, reduce_objects=True)
        return self._result_to_event_mask(result, self._event_count_from_variables(variables))

    def _evaluate_custom_cut_with_objects(self, cut_string, variables):
        result = self._parse_cut_expr(cut_string, variables, reduce_objects=True)
        event_mask = self._result_to_event_mask(
            result, self._event_count_from_variables(variables))
        return event_mask, (result.object_masks or {})

    def _parse_cut_expr(self, cut_string, variables, reduce_objects=True):
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
                return self._parse_cut_expr(cut_string[1:-1], variables, reduce_objects=reduce_objects)

        # Normalize logical operators so both "a&b" and "a & b" parse.
        normalized = cut_string.replace('&&', ' & ').replace('||', ' | ')
        normalized = re.sub(r'(?<!&)&(?!&)', ' & ', normalized)
        normalized = re.sub(r'(?<!\|)\|(?!\|)', ' | ', normalized)

        # Split on OR first (lower precedence), respecting parentheses
        or_parts = self._split_respecting_parens(normalized, ' | ')
        if len(or_parts) > 1:
            result_mask = None
            for part in or_parts:
                part_mask = self._parse_cut_expr(part, variables, reduce_objects=reduce_objects)
                if result_mask is None:
                    result_mask = part_mask
                else:
                    result_mask = self._combine_or_results(result_mask, part_mask, variables)
            return result_mask

        # Then split on AND, respecting parentheses
        and_parts = self._split_respecting_parens(normalized, ' & ')
        if len(and_parts) > 1:
            scalar_masks = []
            object_masks = {}
            for part in and_parts:
                part_result = self._parse_cut_expr(part, variables, reduce_objects=False)
                if part_result.kind == 'object':
                    existing = object_masks.get(part_result.collection)
                    object_masks[part_result.collection] = (
                        part_result.values if existing is None
                        else self._combine_object_masks(existing, part_result.values, '&')
                    )
                else:
                    scalar_masks.append(part_result.values)
            return self._finalize_and_result(scalar_masks, object_masks, variables, reduce_objects)

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
        Evaluate a single condition like 'nSelPhotons==1'.
        """
        import re

        reducer_match = self._match_function_call(condition, 'any')
        if reducer_match is not None:
            inner = self._parse_cut_expr(reducer_match, variables, reduce_objects=False)
            return _CutEvalResult(
                'scalar',
                self._reduce_result(inner, 'any', variables),
                object_masks=inner.object_masks)

        reducer_match = self._match_function_call(condition, 'all')
        if reducer_match is not None:
            inner = self._parse_cut_expr(reducer_match, variables, reduce_objects=False)
            return _CutEvalResult(
                'scalar',
                self._reduce_result(inner, 'all', variables),
                object_masks=inner.object_masks)

        count_match = re.match(
            r'count\s*\((.*)\)\s*(==|!=|<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)\s*$',
            condition
        )
        if count_match:
            inner_expr, operator, value_str = count_match.groups()
            inner = self._parse_cut_expr(inner_expr, variables, reduce_objects=False)
            counts = self._count_result(inner, variables)
            value = self._parse_numeric_value(value_str)
            return _CutEvalResult(
                'scalar',
                self._apply_operator(counts, operator, value),
                object_masks=inner.object_masks)

        lead_match = re.match(
            r'lead\s*\(\s*(\w+)\s*\)\s*(==|!=|<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)\s*$',
            condition
        )
        if lead_match:
            var_name, operator, value_str = lead_match.groups()
            values = self._leading_values(var_name, variables)
            value = self._parse_numeric_value(value_str)
            return _CutEvalResult('scalar', self._apply_operator(values, operator, value))

        # Parse condition with regex ($ anchor ensures full match, no trailing text ignored).
        # Supports abs(var) and negative numeric thresholds.
        match = re.match(
            r'(abs\s*\(\s*(\w+)\s*\)|(\w+))\s*(==|!=|<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)\s*$',
            condition
        )
        if not match:
            raise ValueError(f"Cannot parse condition: {condition}")

        left_expr, abs_var, plain_var, operator, value_str = match.groups()
        var_name = abs_var or plain_var

        value = self._parse_numeric_value(value_str)
        use_abs = left_expr.strip().startswith('abs')
        collection = self._collection_for_branch(var_name)

        if collection:
            object_values = self._object_values(var_name, variables, use_abs=use_abs)
            object_mask = [self._apply_operator(values, operator, value) for values in object_values]
            return _CutEvalResult(
                'object',
                object_mask,
                collection,
                {collection: object_mask}
            )

        if var_name not in variables:
            raise ValueError(f"Unknown variable: {var_name}")

        array = np.asarray(variables[var_name])
        if use_abs:
            array = np.abs(array)
        return _CutEvalResult('scalar', self._apply_operator(array, operator, value))

    @staticmethod
    def _parse_numeric_value(value_str):
        return float(value_str) if '.' in value_str else int(value_str)

    @staticmethod
    def _apply_operator(array, operator, value):
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

    def _match_function_call(self, text, name):
        prefix = f'{name}('
        compact = text.replace(' ', '')
        if not compact.startswith(prefix) or not text.endswith(')'):
            return None

        start = text.find('(')
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return text[start + 1:i].strip() if i == len(text) - 1 else None
        return None

    @staticmethod
    def _collection_for_branch(var_name):
        if var_name.startswith('HadronicSV_'):
            return 'HadronicSV'
        if var_name.startswith('LeptonicSV_'):
            return 'LeptonicSV'
        if var_name.startswith('baseLinePhoton_'):
            return 'baseLinePhoton'
        if var_name in {'rjr_Ms', 'rjr_Rs', 'rjrPTS'}:
            return 'rjr'
        return None

    def _event_count_from_variables(self, variables):
        for values in variables.values():
            try:
                return len(values)
            except TypeError:
                continue
        return 0

    def _object_values(self, var_name, variables, use_abs=False):
        n_events = self._event_count_from_variables(variables)
        raw = variables.get(var_name)
        if raw is None:
            return [np.array([], dtype=float) for _ in range(n_events)]

        values = []
        for item in raw:
            arr = np.asarray(item, dtype=float)
            if use_abs:
                arr = np.abs(arr)
            values.append(arr)
        return values

    def _leading_values(self, var_name, variables):
        if var_name not in variables:
            raise ValueError(f"Unknown variable: {var_name}")

        collection = self._collection_for_branch(var_name)
        if not collection:
            return np.asarray(variables[var_name])

        lead = []
        for values in variables[var_name]:
            arr = np.asarray(values, dtype=float)
            lead.append(float(arr[0]) if len(arr) > 0 else np.nan)
        return np.asarray(lead, dtype=float)

    def _combine_object_masks(self, left, right, operator):
        combined = []
        for left_evt, right_evt in zip(left, right):
            left_arr = np.asarray(left_evt, dtype=bool)
            right_arr = np.asarray(right_evt, dtype=bool)
            n = min(len(left_arr), len(right_arr))
            if operator == '&':
                combined.append(left_arr[:n] & right_arr[:n])
            elif operator == '|':
                combined.append(left_arr[:n] | right_arr[:n])
            else:
                raise ValueError(f"Unknown object-mask operator: {operator}")
        return combined

    def _merge_object_mask_maps(self, left, right, operator):
        if not left and not right:
            return None

        merged = {}
        for collection in set((left or {}).keys()) | set((right or {}).keys()):
            if left and collection in left and right and collection in right:
                merged[collection] = self._combine_object_masks(
                    left[collection], right[collection], operator)
            elif left and collection in left:
                merged[collection] = left[collection]
            else:
                merged[collection] = right[collection]
        return merged

    def _gate_and_object_masks(self, object_masks, scalar_mask):
        gated_masks = {}
        event_masks_by_collection = {
            collection: self._reduce_object_mask(mask, 'any')
            for collection, mask in object_masks.items()
        }

        for collection, obj_mask in object_masks.items():
            gated = []
            other_event_mask = np.asarray(scalar_mask, dtype=bool).copy()
            for other_collection, other_mask in event_masks_by_collection.items():
                if other_collection != collection:
                    other_event_mask &= other_mask

            for evt_mask, event_pass in zip(obj_mask, other_event_mask):
                gated.append(np.asarray(evt_mask, dtype=bool) & bool(event_pass))
            gated_masks[collection] = gated
        return gated_masks

    def _combine_or_results(self, left, right, variables):
        if left.kind == 'object' and right.kind == 'object' and left.collection == right.collection:
            object_mask = self._combine_object_masks(left.values, right.values, '|')
            return _CutEvalResult(
                'object',
                object_mask,
                left.collection,
                {left.collection: object_mask}
            )

        left_mask = self._result_to_event_mask(left, self._event_count_from_variables(variables))
        right_mask = self._result_to_event_mask(right, self._event_count_from_variables(variables))
        return _CutEvalResult(
            'scalar',
            left_mask | right_mask,
            object_masks=self._merge_object_mask_maps(left.object_masks, right.object_masks, '|'))

    def _finalize_and_result(self, scalar_masks, object_masks, variables, reduce_objects):
        n_events = self._event_count_from_variables(variables)
        scalar_mask = np.ones(n_events, dtype=bool)
        for mask in scalar_masks:
            scalar_mask &= np.asarray(mask, dtype=bool)

        if not object_masks:
            return _CutEvalResult('scalar', scalar_mask)

        if not reduce_objects and len(object_masks) == 1:
            collection, obj_mask = next(iter(object_masks.items()))
            gated = []
            for evt_mask, scalar_pass in zip(obj_mask, scalar_mask):
                gated.append(np.asarray(evt_mask, dtype=bool) & bool(scalar_pass))
            return _CutEvalResult('object', gated, collection, {collection: gated})

        event_mask = scalar_mask
        for obj_mask in object_masks.values():
            event_mask &= self._reduce_object_mask(obj_mask, 'any')
        return _CutEvalResult(
            'scalar',
            event_mask,
            object_masks=self._gate_and_object_masks(object_masks, scalar_mask))

    def _result_to_event_mask(self, result, n_events):
        if result.kind == 'scalar':
            return np.asarray(result.values, dtype=bool)
        return self._reduce_object_mask(result.values, 'any')

    def _reduce_result(self, result, reducer, variables):
        if result.kind == 'scalar':
            return np.asarray(result.values, dtype=bool)
        return self._reduce_object_mask(result.values, reducer)

    @staticmethod
    def _reduce_object_mask(object_mask, reducer):
        reduced = []
        for evt_mask in object_mask:
            arr = np.asarray(evt_mask, dtype=bool)
            if reducer == 'any':
                reduced.append(bool(np.any(arr)) if len(arr) > 0 else False)
            elif reducer == 'all':
                reduced.append(bool(np.all(arr)) if len(arr) > 0 else False)
            else:
                raise ValueError(f"Unknown reducer: {reducer}")
        return np.asarray(reduced, dtype=bool)

    def _count_result(self, result, variables):
        if result.kind == 'scalar':
            return np.asarray(result.values, dtype=np.int32)
        return np.asarray([np.count_nonzero(evt_mask) for evt_mask in result.values], dtype=np.int32)

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
