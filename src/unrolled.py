import numpy as np
from typing import Dict, List, Tuple

class UnrolledBinning:
    """
    Handles the mathematics of unrolling 2D (Ms, Rs) data into 1D arrays for plotting.
    Supports flexible binning schemes including 'legacy' (separate) and 'merged' configurations.
    """

    def __init__(self, scheme: str = 'merged_rs'):
        self.scheme = scheme
        
        # Define base bin edges (in TeV for Ms, unitless for Rs)
        # Ms: [1.0, 2.0, 3.0, inf] TeV (convert from GeV to match framework)
        self.ms_bins = [1.0, 2.0, 3.0, float('inf')]
        
        # Rs: [0.15, 0.3, 0.4, inf]
        self.rs_bins = [0.15, 0.3, 0.4, float('inf')]
        
        # Display constants
        self.inf_string = "#scale[1.5]{#infty}"

        # Labels for base bins (already in TeV)
        self.ms_labels = ["[1.0, 2.0]", "[2.0, 3.0]", f"[3.0, {self.inf_string}]"]
        self.rs_labels = ["[0.15, 0.3]", "[0.3, 0.4]", f"[0.4, {self.inf_string}]"]

    def calculate_2d_yields(self, ms_values: np.ndarray, rs_values: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the 3x3 yield and error matrices from raw data arrays.
        """
        yields = np.zeros((3, 3))
        sum_w2 = np.zeros((3, 3))

        # Digitize (convert values to bin indices 0, 1, 2)
        # np.digitize returns 1-based indices (1, 2, 3), so we subtract 1
        ms_indices = np.digitize(ms_values, self.ms_bins) - 1
        rs_indices = np.digitize(rs_values, self.rs_bins) - 1

        # Fill matrix
        for i in range(len(weights)):
            m_idx = ms_indices[i]
            r_idx = rs_indices[i]

            # Filter out overflow/underflow if any (though cuts should handle this)
            if 0 <= m_idx < 3 and 0 <= r_idx < 3:
                w = weights[i]
                yields[m_idx, r_idx] += w
                sum_w2[m_idx, r_idx] += w * w
        
        errors = np.sqrt(sum_w2)
        return yields, errors

    def unroll(self, yields_2d: np.ndarray, errors_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict]]:
        """
        Unrolls 2D matrices into 1D arrays based on the selected scheme.
        
        Returns:
            yields_1d (np.ndarray): The unrolled yields
            errors_1d (np.ndarray): The unrolled errors
            bin_labels (List[str]): Labels for the x-axis bins
            decorations (List[Dict]): Info for drawing lines and group labels
        """
        if self.scheme == 'merged_rs':
            return self._unroll_merged_rs(yields_2d, errors_2d)
        elif self.scheme == 'merged_ms':
            return self._unroll_merged_ms(yields_2d, errors_2d)
        else:
            raise ValueError(f"Unsupported unrolling scheme: {self.scheme}")

    def _unroll_merged_rs(self, yields_2d, errors_2d):
        """
        Scheme 'merged_rs':
        - Group 1 (Bins 1-3): Ms in [1000, 2000]. Rs varies [0.15,0.3], [0.3,0.4], [0.4,inf]
        - Group 2 (Bins 4-5): Ms in [2000, 3000]. Rs varies [0.15,0.3], [0.3, inf] <- Wait, let's check standard
          Actually usually it's:
          - G1 (3 bins): Fixed Low Ms. Full Rs scan.
          - G2 (2 bins): Fixed Med Ms. Low Rs, High Rs (merged 2 & 3).
          - G3 (1 bin):  Fixed High Ms. Full Rs (merged).
          
          Let's verify the user's provided logic in `unrolled_data_processor.py`.
          It says for `merged_rs`:
          Group 1 (3 bins): Ms[0] (1000-2000). Rs varies [0-2] (indices 0,0; 0,1; 0,2)
          Group 2 (2 bins): Ms[1] (2000-3000). Rs varies? Wait.
          
          In `unrolled_data_processor.py`:
          G1 (3 bins): (0,0), (0,1), (0,2)  -> Ms 1-2TeV, Rs scans
          G2 (2 bins): (1,0), (2,0)         -> Rs 0.15-0.3, Ms scans 2-3, 3-inf.
          G3 (1 bin):  (1,1)+(1,2)+(2,1)+(2,2) -> High Ms & High Rs merged.
          
          Wait, the logic in `unrolled_data_processor.py` `unroll_2d_to_1d_merged_rs` is:
          y[0] = (0,0) -> Ms[1,2], Rs[0.15,0.3]
          y[1] = (0,1) -> Ms[1,2], Rs[0.3,0.4]
          y[2] = (0,2) -> Ms[1,2], Rs[0.4,inf]
          
          y[3] = (1,0) -> Ms[2,3], Rs[0.15,0.3]
          y[4] = (2,0) -> Ms[3,inf], Rs[0.15,0.3]
          
          y[5] = Sum((1,1),(1,2),(2,1),(2,2)) -> Ms>2 & Rs>0.3
          
          This matches the "reversed_rs" logic mentioned in the prompts usually, 
          but `unrolled_data_processor.py` calls this `merged_rs`. 
          Let's stick to the implementation in `unrolled_data_processor.py` exactly.
        """
        n_bins = 6
        yields_1d = np.zeros(n_bins)
        errors_1d = np.zeros(n_bins)
        bin_labels = []

        # --- Group 1: Fixed Ms [1.0, 2.0] ---
        # Bin 1: Rs [0.15, 0.3]
        yields_1d[0] = yields_2d[0, 0]
        errors_1d[0] = errors_2d[0, 0]
        bin_labels.append(self.rs_labels[0])

        # Bin 2: Rs [0.3, 0.4]
        yields_1d[1] = yields_2d[0, 1]
        errors_1d[1] = errors_2d[0, 1]
        bin_labels.append(self.rs_labels[1])

        # Bin 3: Rs [0.4, inf]
        yields_1d[2] = yields_2d[0, 2]
        errors_1d[2] = errors_2d[0, 2]
        bin_labels.append(self.rs_labels[2])

        # --- Group 2: Fixed Rs [0.15, 0.3] ---
        # Bin 4: Ms [2.0, 3.0]
        yields_1d[3] = yields_2d[1, 0]
        errors_1d[3] = errors_2d[1, 0]
        bin_labels.append(self.ms_labels[1])

        # Bin 5: Ms [3.0, inf]
        yields_1d[4] = yields_2d[2, 0]
        errors_1d[4] = errors_2d[2, 0]
        bin_labels.append(self.ms_labels[2])

        # --- Group 3: Merged High Signal Region ---
        # Bin 6: Ms > 2.0 AND Rs > 0.3
        merged_val = (yields_2d[1, 1] + yields_2d[1, 2] + 
                      yields_2d[2, 1] + yields_2d[2, 2])
        merged_err = np.sqrt(errors_2d[1, 1]**2 + errors_2d[1, 2]**2 + 
                             errors_2d[2, 1]**2 + errors_2d[2, 2]**2)
        
        yields_1d[5] = merged_val
        errors_1d[5] = merged_err
        bin_labels.append(f"[0.3, {self.inf_string}]") # Label indicating merged Rs range

        # --- Decorations ---
        # Define where vertical lines go (after bin index, 1-based)
        # And labels describing the groups
        # Group labels: Group 1 and 3 normal, Group 2 gets individual labels (handled separately)

        group_labels = [
            {"text": "M_{S} #in [1.0,2.0]", "x_range": (0, 3)},   # Group 1
            {"text": "", "x_range": (3, 5)},                      # Group 2: empty, individual labels instead
            {"text": "M_{S} #geq 2.0", "x_range": (5, 6)}        # Group 3
        ]
        
        # Individual labels for Group 2 (at group label height) 
        individual_labels = [
            {"text": "", "bin": 0},                                    # Group 1 bins get no individual labels
            {"text": "", "bin": 1},
            {"text": "", "bin": 2},
            {"text": "R_{S} #in [0.3,0.4]", "bin": 3},                 # Group 2: individual Rs labels
            {"text": f"R_{{S}} #in [0.4,#scale[1.5]{{#infty}}]", "bin": 4},
            {"text": "", "bin": 5}                                     # Group 3 gets no individual labels
        ]
        
        # Bin labels: normal for Groups 1&3, empty for Group 2 + centered label
        bin_labels = [
            "[1.0, 2.0]",      # Bin 0
            "[2.0, 3.0]",      # Bin 1
            f"[3.0, {self.inf_string}]",  # Bin 2
            "",                # Bin 3: empty
            "",                # Bin 4: empty
            f"[0.4, {self.inf_string}]"   # Bin 5
        ]
        
        # Centered label for Group 2 (at bin label height, spanning bins 3-4)
        centered_label = {"text": "[1.0,2.0]", "x_range": (3, 5)}
        
        decorations = {
            "lines": [3, 5], # Vertical lines after bin 3 and bin 5
            "group_labels": group_labels,
            "individual_labels": individual_labels,
            "bin_labels": bin_labels,
            "centered_label": centered_label
        }
        
        return yields_1d, errors_1d, bin_labels, decorations

    def _unroll_merged_ms(self, yields_2d, errors_2d):
        """
        Scheme 'merged_ms': Symmetric to merged_rs.
        - Group 1 (Bins 1-3): Fixed Rs [0.15, 0.3]. Ms varies.
        - Group 2 (Bins 4-5): Fixed Ms [1.0, 2.0]. Rs varies.
        - Group 3 (Bin 6): Merged High.
        """
        n_bins = 6
        yields_1d = np.zeros(n_bins)
        errors_1d = np.zeros(n_bins)
        bin_labels = []

        # --- Group 1: Fixed Rs [0.15, 0.3] ---
        # Bin 1: Ms [1.0, 2.0]
        yields_1d[0] = yields_2d[0, 0]
        errors_1d[0] = errors_2d[0, 0]
        bin_labels.append(self.ms_labels[0])

        # Bin 2: Ms [2.0, 3.0]
        yields_1d[1] = yields_2d[1, 0]
        errors_1d[1] = errors_2d[1, 0]
        bin_labels.append(self.ms_labels[1])

        # Bin 3: Ms [3.0, inf]
        yields_1d[2] = yields_2d[2, 0]
        errors_1d[2] = errors_2d[2, 0]
        bin_labels.append(self.ms_labels[2])

        # --- Group 2: Fixed Ms [1.0, 2.0] ---
        # Bin 4: Rs [0.3, 0.4]
        yields_1d[3] = yields_2d[0, 1]
        errors_1d[3] = errors_2d[0, 1]
        bin_labels.append(self.rs_labels[1])

        # Bin 5: Rs [0.4, inf]
        yields_1d[4] = yields_2d[0, 2]
        errors_1d[4] = errors_2d[0, 2]
        bin_labels.append(self.rs_labels[2])

        # --- Group 3: Merged High Signal Region ---
        # Bin 6: Ms > 2.0 AND Rs > 0.3 (Same as above)
        merged_val = (yields_2d[1, 1] + yields_2d[1, 2] + 
                      yields_2d[2, 1] + yields_2d[2, 2])
        merged_err = np.sqrt(errors_2d[1, 1]**2 + errors_2d[1, 2]**2 + 
                             errors_2d[2, 1]**2 + errors_2d[2, 2]**2)
        
        yields_1d[5] = merged_val
        errors_1d[5] = merged_err
        bin_labels.append(f"[2.0, {self.inf_string}]")

        # Group labels: Group 1 and 3 normal, Group 2 gets individual labels (handled separately)
        group_labels = [
            {"text": "R_{S} #in [0.15,0.3]", "x_range": (0, 3)},   # Group 1
            {"text": "", "x_range": (3, 5)},                        # Group 2: empty, individual labels instead
            {"text": "R_{S} #geq 0.3", "x_range": (5, 6)}          # Group 3
        ]
        
        # Individual labels for Group 2 (at group label height)
        individual_labels = [
            {"text": "", "bin": 0},                                    # Group 1 bins get no individual labels
            {"text": "", "bin": 1},
            {"text": "", "bin": 2},
            {"text": "M_{S} #in [2.0,3.0]", "bin": 3},                # Group 2: individual Ms labels
            {"text": f"M_{{S}} #in [3.0,#scale[1.5]{{#infty}}]", "bin": 4},
            {"text": "", "bin": 5}                                     # Group 3 gets no individual labels
        ]
        
        # Bin labels: normal for Groups 1&3, empty for Group 2 + centered label
        bin_labels = [
            "[1.0, 2.0]",     # Bin 0
            "[2.0, 3.0]",      # Bin 1  
            f"[3.0, {self.inf_string}]",  # Bin 2
            "",                # Bin 3: empty
            "",                # Bin 4: empty  
            f"[2.0, {self.inf_string}]"   # Bin 5
        ]
        
        # Centered label for Group 2 (at bin label height, spanning bins 3-4)
        centered_label = {"text": "[0.15,0.3]", "x_range": (3, 5)}
        
        decorations = {
            "lines": [3, 5],
            "group_labels": group_labels,
            "individual_labels": individual_labels,
            "bin_labels": bin_labels,
            "centered_label": centered_label
        }

        return yields_1d, errors_1d, decorations["bin_labels"], decorations

    def add_separator_lines(self, canvas, hist, scheme):
        """Copy of working implementation from unrolled_canvas_maker.py"""
        import ROOT
        
        # Define separator bin positions for each scheme (from unrolled_canvas_maker.py)
        scheme_config = {
            'merged_rs': {'total_bins': 6, 'separator_bins': [3, 5]},
            'merged_ms': {'total_bins': 6, 'separator_bins': [3, 5]},
        }
        
        if scheme not in scheme_config:
            return []
            
        config = scheme_config[scheme]
        separator_bins = config['separator_bins']
        total_bins = config['total_bins']
                                  
        canvas.cd()
        canvas.Update()
        
        # === Create an overlay pad (fully visual coordinate system, NDC) ===
        overlay = ROOT.TPad("overlay","overlay",0,0,1,1)
        overlay.SetFillStyle(0)
        overlay.SetFrameFillStyle(0)
        overlay.SetBorderSize(0)
        overlay.SetBorderMode(0)
        overlay.SetMargin(0,0,0,0)
        overlay.SetBit(ROOT.kCannotPick)  # Make transparent to mouse events
        overlay.Draw()
        overlay.cd()
        
        # === Convert x positions to NDC within the *main* pad ===
        # The pads exist as primitives, not as numbered GetPad() objects
        # Look for the main plotting pad in the canvas primitives
        main_pad = None
        primitives = canvas.GetListOfPrimitives()
        for i in range(primitives.GetSize()):
            obj = primitives.At(i)
            if obj and obj.InheritsFrom("TPad"):
                # Check if this looks like the main pad (top pad in ratio plot)
                pad = obj
                # Main pad should have larger height (top pad in ratio layout)
                if pad.GetY2() > 0.5:  # Top pad
                    main_pad = pad
                    break
        
        if main_pad is None:
            # Fallback to canvas itself
            main_pad = canvas
        
        main_pad.Update()
        
        x_axis = hist.GetXaxis()
        x_min = x_axis.GetXmin()
        x_max = x_axis.GetXmax()
        
        # Pad margins
        left_ndc = main_pad.GetLeftMargin()
        right_ndc = 1.0 - main_pad.GetRightMargin()
        data_ndc_width = right_ndc - left_ndc
        
        # Use actual histogram axis range
        hist_axis_range = x_max - x_min
        
        def x_to_ndc_from_bin_edge(bin_edge_index):
            # Convert bin edge index to histogram x coordinate, then to NDC
            x_hist = bin_edge_index  # Since bins are numbered 0, 1, 2, ... nbins-1
            norm = (x_hist - x_min) / hist_axis_range
            return left_ndc + norm * data_ndc_width

        # === Now draw the vertical lines in PURE NDC ===
        y_bottom = 0.07   # Bottom of canvas (bottom of ratio pad)
        y_top = 0.93     # Near top of distribution pad (below group labels)
        
        lines = []
        for b_edge in separator_bins:
            x_ndc = x_to_ndc_from_bin_edge(b_edge)
            line = ROOT.TLine()
            line.SetNDC(True)
            line.SetLineColor(ROOT.kBlack)
            line.SetLineWidth(2)
            line.SetLineStyle(1)
            line.DrawLine(x_ndc, y_bottom, x_ndc, y_top)
            lines.append(line)

        canvas.Modified()
        canvas.Update()
        return lines

    def add_individual_labels(self, canvas, scheme):
        """Add individual labels for merged schemes (copied from unrolled_canvas_maker.py)"""
        import ROOT
        
        if scheme == 'merged_rs':
            # Add individual Ms labels at top for merged_rs (bins 3-4 need individual Ms labels)
            individual_ms_labels = ["M_{S} #in [2.0,3.0]", f"M_{{S}} #in [3.0,#scale[1.5]{{#infty}}]"]
            individual_bins = [3, 4]  # Bins that need individual Ms labels (0-indexed)
        elif scheme == 'merged_ms':
            # Add individual Rs labels at top for merged_ms (bins 3-4 need individual Rs labels)  
            individual_rs_labels = ["R_{S} #in [0.3,0.4]", f"R_{{S}} #in [0.4,#scale[1.5]{{#infty}}]"]
            individual_bins = [3, 4]  # Bins that need individual Rs labels (0-indexed)
            labels = individual_rs_labels
        else:
            return []
        
        # Get labels based on scheme
        if scheme == 'merged_rs':
            labels = individual_ms_labels
        else:
            labels = individual_rs_labels
        
        # Create overlay pad for individual labels
        overlay = ROOT.TPad("individual_overlay", "individual_overlay", 0, 0, 1, 1)
        overlay.SetFillStyle(0)
        overlay.SetFrameFillStyle(0)
        overlay.SetBorderSize(0)
        overlay.SetBorderMode(0)
        overlay.SetMargin(0, 0, 0, 0)
        overlay.SetBit(ROOT.kCannotPick)
        overlay.Draw()
        overlay.cd()
        
        # Get pad margins (same logic as separator lines)
        main_pad = None
        primitives = canvas.GetListOfPrimitives()
        for i in range(primitives.GetSize()):
            obj = primitives.At(i)
            if obj and obj.InheritsFrom("TPad"):
                pad = obj
                if pad.GetY2() > 0.5:
                    main_pad = pad
                    break
        
        if main_pad is None:
            main_pad = canvas
        
        left_ndc = main_pad.GetLeftMargin()
        right_ndc = 1.0 - main_pad.GetRightMargin()
        data_ndc_width = right_ndc - left_ndc
        
        # Position individual labels
        latex_objects = []
        total_bins = 6
        
        for i, (bin_idx, label) in enumerate(zip(individual_bins, labels)):
            # Calculate x position for center of this bin
            bin_center = bin_idx + 0.5  # Center of bin
            x_normalized = bin_center / total_bins
            x_ndc = left_ndc + x_normalized * data_ndc_width
            
            # Y position
            y_ndc = 0.83  # Same as group labels in datamc (hardcoded like original)
            
            # Create and position label
            latex_individual = ROOT.TLatex()
            latex_individual.SetNDC(True)
            latex_individual.SetTextFont(42)
            latex_individual.SetTextSize(0.05)
            latex_individual.SetTextAlign(22)  # Center alignment
            latex_individual.DrawLatex(x_ndc, y_ndc, label)
            latex_objects.append(latex_individual)
        
        canvas.Modified()
        canvas.Update()
        return latex_objects

    def add_merged_centered_labels(self, canvas, scheme):
        """Add centered labels for merged schemes (copied from unrolled_canvas_maker.py)"""
        import ROOT
        
        if scheme not in ['merged_rs', 'merged_ms']:
            return []
        
        # Get scheme config
        scheme_config = {
            'merged_rs': {'total_bins': 6, 'group_widths': [3, 2, 1]},
            'merged_ms': {'total_bins': 6, 'group_widths': [3, 2, 1]},
        }
        
        config = scheme_config[scheme]
        total_bins = config['total_bins']
        group_widths = config['group_widths']
        
        # Create overlay pad
        overlay = ROOT.TPad("centered_overlay", "centered_overlay", 0, 0, 1, 1)
        overlay.SetFillStyle(0)
        overlay.SetFrameFillStyle(0)
        overlay.SetBorderSize(0)
        overlay.SetBorderMode(0)
        overlay.SetMargin(0, 0, 0, 0)
        overlay.SetBit(ROOT.kCannotPick)
        overlay.Draw()
        overlay.cd()
        
        # Get pad margins
        main_pad = None
        primitives = canvas.GetListOfPrimitives()
        for i in range(primitives.GetSize()):
            obj = primitives.At(i)
            if obj and obj.InheritsFrom("TPad"):
                pad = obj
                if pad.GetY2() > 0.5:
                    main_pad = pad
                    break
        
        if main_pad is None:
            main_pad = canvas
        
        left_ndc = main_pad.GetLeftMargin()
        right_ndc = 1.0 - main_pad.GetRightMargin()
        data_ndc_width = right_ndc - left_ndc
        
        centered_text_objects = []
        
        if scheme == 'merged_rs':
            # Center [0.15,0.3] over Group 2 (bins 3-4, which is group_widths[1] = 2 bins)
            group_start_bin = group_widths[0]  # 3
            group_width = group_widths[1]      # 2
            group_center = group_start_bin + group_width / 2.0  # 4.0
            
            x_normalized = group_center / total_bins
            x_position = left_ndc + x_normalized * data_ndc_width
            y_position_bottom = 0.31  # Bottom position for the centered label
            
            # Create centered label
            latex_bottom = ROOT.TLatex()
            latex_bottom.SetNDC(True)
            latex_bottom.SetTextFont(42)
            latex_bottom.SetTextSize(0.16)
            latex_bottom.SetTextAlign(22)
            latex_bottom.DrawLatex(x_position, y_position_bottom, "[0.15,0.3]")
            centered_text_objects.append(latex_bottom)
            
        elif scheme == 'merged_ms':
            # Center [1.0,2.0] over Group 2 (bins 3-4, which is group_widths[1] = 2 bins) 
            group_start_bin = group_widths[0]  # 3 (after Group 1)
            group_width = group_widths[1]      # 2
            group_center = group_start_bin + group_width / 2.0  # 4.0
            
            x_normalized = group_center / total_bins
            x_position = left_ndc + x_normalized * data_ndc_width
            y_position_bottom = 0.31
            
            # Create centered label
            latex_bottom = ROOT.TLatex()
            latex_bottom.SetNDC(True)
            latex_bottom.SetTextFont(42)
            latex_bottom.SetTextSize(0.16)
            latex_bottom.SetTextAlign(22)
            latex_bottom.DrawLatex(x_position, y_position_bottom, "[1.0,2.0]")
            centered_text_objects.append(latex_bottom)
        
        canvas.Modified()
        canvas.Update()
        return centered_text_objects
