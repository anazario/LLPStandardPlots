import re

class FinalStateResolver:
    """Resolves final state flags to LaTeX labels."""
    
    # Ordered photon region keyword → compact LaTeX label.
    # Checked most-specific first so e.g. "EarlyBeamHalo" matches before "BeamHalo".
    _PHOTON_REGION_MAP = [
        ('EarlyBeamHalo',          '#gamma_{t-}^{CR, BH}'),
        ('LateBeamHalo',           '#gamma_{t+}^{CR, BH}'),
        ('BeamHalo',               '#gamma_{t0}^{CR, BH}'),
        ('EarlyNotBH',             '#gamma_{t-}^{CR, !BH}'),
        ('LateNotBH',              '#gamma_{t+}^{SR, !BH}'),
        ('NotBHPrompt',            '#gamma_{t0}^{CR, !BH}'),
        ('NotBH',                  '#gamma^{CR, !BH}'),
        ('PromptLooseNotTightIso1','#gamma_{t0}^{CR, L!T iso, 1}'),
        ('PromptLooseNotTightIso2','#gamma_{t0}^{CR, L!T iso, 2}'),
        ('PromptLooseNotTightIso', '#gamma_{t0}^{CR, L!T iso}'),
        ('PromptTightIso',         '#gamma_{t0}^{SR, tight iso}'),
    ]

    @staticmethod
    def format_sv_label(final_state: str) -> str:
        """
        Format final state labels for SV-based and photon-based selections.

        SV convention: {N}SV_{flavor}^{selection}
          - N: shown only for ≥2 SVs
          - flavor: 'hh' (hadronic) or '\\ell\\ell' (leptonic)
          - selection: CR/SR + L/T

        Photon convention: {N}<region_label>
          - Region label encodes timing and BH/!BH info (see _PHOTON_REGION_MAP)
          - N: "2" prefix for NPhoEq2, empty otherwise

        Mixed photon+SV: {N}<photon_region> + {sv_count}SV_{flavor}

        SV examples:
          passNHad1SelectionSRLoose         -> Region: SV_{hh}^{SR,L}
          passNLep1SelectionSRTight         -> Region: SV_{\\ell\\ell}^{SR,T}
          passNHad2SelectionCRLoose         -> Region: 2SV_{hh}^{CR,L}
          passNLepNHadSelectionSRTight      -> Region: SV_{\\ell\\ell}SV_{hh}^{SR,T}

        Photon examples:
          passNPhoEq1SelectionEarlyBeamHaloCR    -> Region: #gamma_{t-}^{CR, BH}
          passNPhoEq2SelectionLateNotBHSR        -> Region: 2#gamma_{t+}^{SR, !BH}
          passNPhoGe1SelectionPromptTightIsoSR   -> Region: #gamma_{t0}^{SR, tight iso}
          passNPhoEq1SelectionNotBHCR            -> Region: #gamma^{CR, !BH}
        """

        # ------------------------------------------------------------------ #
        # OR-combined flags: label each part and join with ' | '              #
        # ------------------------------------------------------------------ #
        if '|' in final_state:
            parts = [FinalStateResolver.format_sv_label(p.strip()) for p in final_state.split('|')]
            # Keep "Region: " prefix from the first part only
            return parts[0] + ''.join(' | ' + p.replace('Region: ', '') for p in parts[1:])

        # ------------------------------------------------------------------ #
        # Photon path                                                          #
        # ------------------------------------------------------------------ #
        if "NPho" in final_state:
            # Multiplicity prefix: "2" for NPhoEq2, empty for Eq1/Ge1
            pho_count = "2" if "NPhoEq2" in final_state else ""

            # Resolve photon region label (most-specific keyword wins)
            pho_region = None
            for keyword, label in FinalStateResolver._PHOTON_REGION_MAP:
                if keyword in final_state:
                    pho_region = label
                    break
            # Fallback: no photon-specific region keyword found; use SV-style region suffix
            if pho_region is None:
                if "CR" in final_state:
                    sel = "CR,L" if "Loose" in final_state else "CR,T"
                elif "SR" in final_state:
                    sel = "SR,L" if "Loose" in final_state else "SR,T"
                else:
                    sel = ""
                pho_region = f"#gamma^{{{sel}}}" if sel else "#gamma"

            # Optional SV component (mixed photon+SV flags)
            # Parse CR/SR and L/T from the substring *after* the NHad/NLep
            # token so we don't accidentally pick up the photon region's keyword.
            sv_part = ""
            if "NHad" in final_state:
                sv_count = ""
                had_match = re.search(r'NHad(\d+)', final_state)
                if had_match and int(had_match.group(1)) >= 2:
                    sv_count = "2"
                sv_suffix = final_state[had_match.start():]
                if "CR" in sv_suffix:
                    sv_sel = "CR,L" if "Loose" in sv_suffix else "CR,T"
                elif "SR" in sv_suffix:
                    sv_sel = "SR,L" if "Loose" in sv_suffix else "SR,T"
                else:
                    sv_sel = ""
                sv_sel_str = f"^{{{sv_sel}}}" if sv_sel else ""
                sv_part = f" + {sv_count}SV_{{hh}}{sv_sel_str}"
            elif "NLep" in final_state:
                sv_count = ""
                lep_match = re.search(r'NLep(\d+)', final_state)
                if lep_match and int(lep_match.group(1)) >= 2:
                    sv_count = "2"
                sv_suffix = final_state[lep_match.start():]
                if "CR" in sv_suffix:
                    sv_sel = "CR,L" if "Loose" in sv_suffix else "CR,T"
                elif "SR" in sv_suffix:
                    sv_sel = "SR,L" if "Loose" in sv_suffix else "SR,T"
                else:
                    sv_sel = ""
                sv_sel_str = f"^{{{sv_sel}}}" if sv_sel else ""
                sv_part = f" + {sv_count}SV_{{\\ell\\ell}}{sv_sel_str}"

            return f"Region: {pho_count}{pho_region}{sv_part}"

        # ------------------------------------------------------------------ #
        # SV path (original logic preserved exactly)                           #
        # ------------------------------------------------------------------ #
        count = ""
        flavor = "hh"  # default to hadronic
        selection = "SR,T"  # default to signal region tight

        if "HadAndLep" in final_state or "LepAndHad" in final_state:
            if "CR" in final_state:
                selection = "CR,L" if "Loose" in final_state else "CR,T"
            elif "SR" in final_state:
                selection = "SR,L" if "Loose" in final_state else "SR,T"
            return f"Region: SV_{{\\ell\\ell}}SV_{{hh}}^{{{selection}}}"

        elif "NHad" in final_state:
            flavor = "hh"
            if "HadGe2" in final_state:
                count = "2"
            else:
                had_match = re.search(r'NHad(\d+)', final_state)
                if had_match and int(had_match.group(1)) >= 2:
                    count = "2"

        elif "NLep" in final_state:
            flavor = "\\ell\\ell"
            if "LepGe2" in final_state:
                count = "2"
            else:
                lep_match = re.search(r'NLep(\d+)', final_state)
                if lep_match and int(lep_match.group(1)) >= 2:
                    count = "2"

        if "CR" in final_state:
            selection = "CR,L" if "Loose" in final_state else "CR,T"
        elif "SR" in final_state:
            selection = "SR,L" if "Loose" in final_state else "SR,T"

        return f"Region: {count}SV_{{{flavor}}}^{{{selection}}}"

    @staticmethod
    def get_active_flag(tree_branches):
        """
        Scans tree branches to find the active 'pass...' flag.
        Returns the flag name and its formatted label.
        """
        # This logic assumes only ONE of the relevant flags is true per event
        # or that we are just looking for the presence of the branch in the file
        # If this is per-event logic, it needs to be inside the event loop.
        # If this is "what dataset is this?", we check the branches.
        
        # Assuming we are looking for existence of branches for now
        # or user specifies which flag to filter on.
        pass 

class SelectionManager:
    """
    Manages physics selections (Cuts, Filters, Triggers).
    """
    def __init__(self):
        self.common_cuts = [
            "selCMet > 150",
            "rjrPTS < 150"
        ]
        self.flags = [
            "Flag_MetFilters",
            "hlt_flags" 
            # Add other boolean flags here if they are single branches
        ]
        # HLT fallback expression
        self.hlt_fallback_expression = "(Trigger_PFMET120_PFMHT120_IDTight || Trigger_PFMETNoMu120_PFMHTNoMu120_IDTight || Trigger_PFMET120_PFMHT120_IDTight_PFHT60 || Trigger_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60)"

    def get_combined_selection_string(self, final_state_flag: str = None):
        """Returns a string representation of cuts for uproot.filter/cut."""
        cuts = list(self.common_cuts)
        
        # Add boolean flags (assumed to be == 1)
        for flag in self.flags:
            cuts.append(f"({flag} == 1)")
            
        if final_state_flag:
            cuts.append(f"({final_state_flag} == 1)")
            
        return " & ".join(cuts)
