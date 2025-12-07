import re

class FinalStateResolver:
    """Resolves final state flags to LaTeX labels."""
    
    @staticmethod
    def format_sv_label(final_state: str) -> str:
        """
        Format final state labels according to SV convention: {N}SV_{flavor}^{selection}
        
        Convention:
        - N: total count of SVs (only shown for 2 or more, just shows "2")
        - flavor: 'hh' (hadronic) or '\ell\ell' (leptonic)
        - selection: CRL, SRL, CRT, SRT
        - mixed case: SV_{\ell\ell}SV_{hh}^{selection}
        
        Examples:
        - passNHad1SelectionSRLoose -> SV_{hh}^{SR,L}
        - passNLep1SelectionSRTight -> SV_{\ell\ell}^{SR,T}
        - passNHad2SelectionCRLoose -> 2SV_{hh}^{CR,L}
        - passNLepNHadSelectionSRTight -> SV_{\ell\ell}SV_{hh}^{SR,T}
        """
        
        # Default values
        count = ""
        flavor = "hh"  # default to hadronic
        selection = "SR,T"  # default to signal region tight
        
        # Parse the branch name pattern
        
        # Extract flavor and count
        if "HadAndLep" in final_state or "LepAndHad" in final_state:
            # Mixed case
            if "CR" in final_state:
                if "Loose" in final_state:
                    selection = "CR,L"
                else:
                    selection = "CR,T"
            elif "SR" in final_state:
                if "Loose" in final_state:
                    selection = "SR,L"
                else:
                    selection = "SR,T"
            return f"SV_{{\ell\ell}}SV_{{hh}}^{{{selection}}}"
            
        elif "NHad" in final_state:
            flavor = "hh"
            if "HadGe2" in final_state:
                count = "2"
            else:
                had_match = re.search(r'NHad(\d+)', final_state)
                if had_match:
                    sv_count = int(had_match.group(1))
                    if sv_count >= 2:
                        count = "2"
                        
        elif "NLep" in final_state:
            flavor = "\\ell\\ell"
            if "LepGe2" in final_state:
                count = "2"
            else:
                lep_match = re.search(r'NLep(\d+)', final_state)
                if lep_match:
                    sv_count = int(lep_match.group(1))
                    if sv_count >= 2:
                        count = "2"
        
        # Extract selection region
        if "CR" in final_state:
            if "Loose" in final_state:
                selection = "CR,L"
            else:
                selection = "CR,T"
        elif "SR" in final_state:
            if "Loose" in final_state:
                selection = "SR,L"
            else:
                selection = "SR,T"
                
        return f"{count}SV_{{{flavor}}}^{{{selection}}}"

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

    def get_combined_selection_string(self, final_state_flag: str = None):
        """Returns a string representation of cuts for uproot.filter/cut."""
        cuts = list(self.common_cuts)
        
        # Add boolean flags (assumed to be == 1)
        for flag in self.flags:
            cuts.append(f"({flag} == 1)")
            
        if final_state_flag:
            cuts.append(f"({final_state_flag} == 1)")
            
        return " & ".join(cuts)
