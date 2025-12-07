import ROOT
import numpy as np
import cmsstyle as CMS
from src.style import StyleManager
from src.utils import parse_signal_name, parse_background_name
from src.config import AnalysisConfig

ROOT.gROOT.ForceStyle(False)

class PlotterBase:
    def __init__(self, style_manager):
        self.style = style_manager

    def _draw_region_label(self, canvas, label, x_pos=0.48, y_pos=None, textsize=None, plot_type="default"):
        """Helper to draw the region/final state label."""
        if label:
            y_pos = y_pos if y_pos is not None else self.style.cms_y_pos
            textsize = textsize if textsize is not None else self.style.sample_label_size
            
            # Use style manager to handle complex label rendering
            latex_objects = self.style.draw_region_label(canvas, label, x_pos, y_pos, textsize, plot_type)
            
            # Keep objects alive by storing them on the canvas
            if isinstance(latex_objects, list):
                canvas.region_latex_objects = latex_objects
            else:
                canvas.region_latex_objects = [latex_objects]
            
            return latex_objects

    def _map_var_name(self, var_name):
        """Map short variable names to full data keys."""
        var_map = {
            'ms': 'rjr_Ms', 
            'rs': 'rjr_Rs',
            'met': 'selCMet',
            'had_mass': 'HadronicSV_mass',
            'had_dxy': 'HadronicSV_dxy', 
            'had_dxysig': 'HadronicSV_dxySig',
            'had_povere': 'HadronicSV_pOverE',
            'had_decayangle': 'HadronicSV_decayAngle',
            'had_costheta': 'HadronicSV_cosTheta',
            'lep_mass': 'LeptonicSV_mass',
            'lep_dxy': 'LeptonicSV_dxy',
            'lep_dxysig': 'LeptonicSV_dxySig', 
            'lep_povere': 'LeptonicSV_pOverE',
            'lep_decayangle': 'LeptonicSV_decayAngle',
            'lep_costheta': 'LeptonicSV_cosTheta'
        }
        return var_map.get(var_name, var_name)

class Plotter1D(PlotterBase):
    def __init__(self, style_manager):
        super().__init__(style_manager)

    def create_histogram(self, data, weights, bins, x_min, x_max, title, color=ROOT.kBlack, style=1):
        hist = ROOT.TH1F(f"h_{title}_{np.random.randint(0, 10000)}", title, bins, x_min, x_max)
        hist.SetDirectory(0)
        hist.Sumw2()
        
        # Efficient filling
        for val, w in zip(data, weights):
            hist.Fill(val, w)
            
        hist.SetLineColor(color)
        hist.SetLineStyle(style)
        hist.SetLineWidth(2)
        hist.SetFillColor(color)
        hist.SetFillStyle(3003)
        hist.SetStats(0)
        return hist

    def setup_axes(self, hist, x_label, y_label="Events", normalized=False):
        """Configures axis properties for visibility and consistent styling."""
        hist.SetStats(0)
        hist.SetTitle("")
        hist.GetXaxis().SetTitle(x_label)
        hist.GetXaxis().SetTitleOffset(self.style.axis_title_offset)
        hist.GetXaxis().SetTitleSize(self.style.axis_title_size)
        hist.GetXaxis().SetLabelSize(self.style.axis_label_size)
        hist.GetXaxis().CenterTitle(True)
        
        # Set y-axis title based on normalization
        if normalized:
            # Extract just the variable name without units for the normalized y-axis title
            # Remove units in brackets like [TeV] and clean up the variable name
            import re
            clean_var = re.sub(r'\s*\[.*?\]', '', x_label).strip()
            y_axis_title = f"#frac{{1}}{{N}}  #frac{{dN}}{{d({clean_var})}}"
        else:
            y_axis_title = y_label
            
        hist.GetYaxis().SetTitle(y_axis_title)
        hist.GetYaxis().SetTitleOffset(1.5 if normalized else self.style.axis_title_offset)
        hist.GetYaxis().SetTitleSize(self.style.axis_title_size)
        hist.GetYaxis().SetLabelSize(self.style.axis_label_size)
        hist.GetYaxis().CenterTitle(True)

    def _initialize_canvas(self, name, x_min, x_max, var_label, y_label="Events"):
        """Helper to initialize a consistent CMS canvas."""
        canvas = CMS.cmsCanvas(name, x_min, x_max, 0, 1, var_label, y_label, square=False, extraSpace=0.01, iPos=0)
        canvas.SetCanvasSize(self.style.canvas_width, self.style.canvas_height)
        canvas.SetGridx(True)
        canvas.SetGridy(True)
        canvas.SetLeftMargin(self.style.margin_left+0.04)
        canvas.SetRightMargin(self.style.margin_right-0.02)
        return canvas

    def plot_collection(self, data_collection, var_name, var_label, bins, x_min, x_max, collection_type="Signal", normalized=False, suffix="", final_state_label=None):
        """Creates a canvas with distributions from a collection of files (signals or backgrounds)."""
        canvas_name = f"{collection_type.lower()}s_{var_name}_{suffix}"
        canvas = self._initialize_canvas(canvas_name, x_min, x_max, var_label)
        
        # Use different legend position for background plots
        if collection_type.lower() == "background":
            legend = CMS.cmsLeg(0.63, 0.63, 0.94, 0.88, textSize=0.035)
        else:
            legend = CMS.cmsLeg(0.35, 0.675, 0.65, 0.874, textSize=0.035)
        
        hists = []
        max_y = 0
        
        for i, (filename, data) in enumerate(data_collection.items()):
            color = self.style.get_color(i)
            
            h = self.create_histogram(data[self._map_var_name(var_name)], data['weights'], bins, x_min, x_max, filename, color=color)
            if normalized and h.Integral() > 0:
                h.Scale(1.0 / h.Integral())
            
            if h.GetMaximum() > max_y:
                max_y = h.GetMaximum()
                
            hists.append(h)
            
            # Determine label based on collection type
            if collection_type.lower() == "background":
                label = parse_background_name(filename)
            else:
                label = parse_signal_name(filename)
                
            legend.AddEntry(h, label, "fl")

        if hists:
            # Use helper for axis setup
            self.setup_axes(hists[0], var_label, normalized=normalized)

            hists[0].GetYaxis().SetRangeUser(0.0001, max_y * 100 if normalized else max_y * 1000)
            hists[0].Draw("HIST")
            for h in hists[1:]:
                h.Draw("HIST SAME")
        
        legend.Draw()
        canvas.SetLogy()
        # Use common CMS label drawing, optimized for 1D
        self.style.draw_cms_labels(cms_x=0.16, cms_y=self.style.cms_y_pos, prelim_x=0.26, lumi_x=0.9, cms_text_size_mult=1.1)
        
        # Draw Region Label
        self._draw_region_label(canvas, final_state_label, plot_type="1d")

        canvas.Update() # Ensure all drawing is updated

        # Keep objects alive
        canvas.hists = hists
        canvas.legend = legend
        
        return canvas

    def plot_signals_vs_net_background(self, signals_data, background_combined, var_name, var_label, bins, x_min, x_max, normalized=False, suffix="", final_state_label=None):
        """Plots combined background (shaded) vs individual signals."""
        canvas_name = f"signals_vs_net_background_{var_name}_{suffix}"
        canvas = self._initialize_canvas(canvas_name, x_min, x_max, var_label)
        
        legend = CMS.cmsLeg(0.35, 0.675, 0.65, 0.874, textSize=0.035)
        
        # Background
        bg_hist = self.create_histogram(background_combined[self._map_var_name(var_name)], background_combined['weights'], bins, x_min, x_max, "Total Background")
        bg_hist.SetFillColor(ROOT.kGray+1)
        bg_hist.SetLineColor(ROOT.kGray+2)
        bg_hist.SetFillStyle(3004)
        
        if normalized and bg_hist.Integral() > 0:
            bg_hist.Scale(1.0 / bg_hist.Integral())
            
        max_y = bg_hist.GetMaximum()
        
        # Signals
        sig_hists = []
        for i, (filename, data) in enumerate(signals_data.items()):
            # Use distinct colors for signals
            color = self.style.get_color(i)
            
            h = self.create_histogram(data[self._map_var_name(var_name)], data['weights'], bins, x_min, x_max, filename, color=color)
            if normalized and h.Integral() > 0:
                h.Scale(1.0 / h.Integral())
            
            if h.GetMaximum() > max_y:
                max_y = h.GetMaximum()
            
            sig_hists.append(h)
            label = parse_signal_name(filename)
            legend.AddEntry(h, label, "fl")
            
        # Draw
        legend.AddEntry(bg_hist, "Total Background", "f")
        
        # Use helper for axis setup
        self.setup_axes(bg_hist, var_label, normalized=normalized)
        
        # Set Range
        bg_hist.GetYaxis().SetRangeUser(0.0001, max_y * 100 if normalized else max_y * 1000)
        
        bg_hist.Draw("HIST")
        for h in sig_hists:
            h.Draw("HIST SAME")
            
        legend.Draw()
        canvas.SetLogy()
        # Use common CMS label drawing, optimized for 1D
        self.style.draw_cms_labels(cms_x=0.16, cms_y=self.style.cms_y_pos, prelim_x=0.26, lumi_x=0.9, cms_text_size_mult=1.1)
        
        # Draw Region Label
        self._draw_region_label(canvas, final_state_label, plot_type="1d")

        canvas.Update() # Ensure all drawing is updated
        
        # Keep objects alive
        canvas.bg_hist = bg_hist
        canvas.sig_hists = sig_hists
        canvas.legend = legend
        
        return canvas, bg_hist, sig_hists # Return objs to keep in memory

class Plotter2D(PlotterBase):
    def __init__(self, style_manager):
        super().__init__(style_manager)
        
    def create_2d_histogram(self, ms, rs, weights, nbins_x, x_min, x_max, nbins_y, y_min, y_max, title):
        hist = ROOT.TH2F(f"h2d_{title}_{np.random.randint(0,10000)}", title, nbins_x, x_min, x_max, nbins_y, y_min, y_max)
        hist.SetDirectory(0)
        
        for m, r, w in zip(ms, rs, weights):
            hist.Fill(m, r, w)
            
        return hist

    def plot_2d(self, data, name, sample_label, final_state_label, sample_label_x_pos=0.65):
        """Creates a CMS styled 2D plot."""
        # Load ranges from config
        ms_conf = AnalysisConfig.VARIABLES['rjr_Ms']
        rs_conf = AnalysisConfig.VARIABLES['rjr_Rs']
        
        ms_min, ms_max = ms_conf['range']
        rs_min, rs_max = rs_conf['range']
        ms_label = ms_conf['label']
        rs_label = rs_conf['label']
        
        hist = self.create_2d_histogram(data['rjr_Ms'], data['rjr_Rs'], data['weights'], 
                                      ms_conf['bins'], ms_min, ms_max, 
                                      rs_conf['bins'], rs_min, rs_max, name)
        
        canvas = CMS.cmsCanvas(name, ms_min, ms_max, rs_min, rs_max, ms_label, rs_label, 
                              square=False, extraSpace=0.01, iPos=0, with_z_axis=True)
        canvas.SetLogz(True)
        canvas.SetGridx(True)
        canvas.SetGridy(True)
        canvas.SetLeftMargin(self.style.margin_left)
        canvas.SetRightMargin(self.style.margin_right+0.06)
        canvas.cd() # Explicitly change to this canvas
        
        # Re-apply palette (CMS style might reset it)
        ROOT.gStyle.SetPalette(self.style.color_palette)
        
        # Set histogram formatting (restored from original)
        hist.SetStats(0)
        hist.SetTitle("")
        hist.GetXaxis().SetTitle(ms_label)
        hist.GetYaxis().SetTitle(rs_label)
        hist.GetZaxis().SetTitle("Events")
        hist.GetXaxis().CenterTitle(True)
        hist.GetYaxis().CenterTitle(True)
        hist.GetZaxis().CenterTitle(True)
        hist.GetXaxis().SetTitleSize(self.style.axis_title_size)
        hist.GetYaxis().SetTitleSize(self.style.axis_title_size)
        hist.GetZaxis().SetTitleSize(self.style.axis_title_size)
        hist.GetXaxis().SetTitleOffset(1.1)
        hist.GetYaxis().SetTitleOffset(1.15)
        hist.GetZaxis().SetTitleOffset(1.3)
        hist.GetXaxis().SetLabelSize(self.style.axis_label_size)
        hist.GetYaxis().SetLabelSize(self.style.axis_label_size)
        hist.GetZaxis().SetLabelSize(self.style.axis_label_size)
        
        # Re-apply palette immediately before drawing
        ROOT.gStyle.SetPalette(self.style.color_palette)
        
        hist.Draw("COLZ")
        canvas.Update() # Update after drawing histogram
        
        # Use common CMS label drawing, using default positions
        self.style.draw_cms_labels()
        self.style.draw_process_label(sample_label, x_pos=sample_label_x_pos, y_pos=0.87)
        self._draw_region_label(canvas, final_state_label, x_pos=0.435, plot_type="2d")
        
        canvas.Update() # Update after adding labels
        
        return canvas, hist

class PlotterDataMC(PlotterBase):
    def __init__(self, style_manager):
        super().__init__(style_manager)
        
        # Define MC colors - indices for colors created by _ensure_mc_colors()
        self.mc_colors = [1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186]
        self.data_color = ROOT.kBlack
    
    def _clean_mc_label(self, label):
        """Extract clean physics process name from file-based label."""
        # Remove common suffixes
        clean_label = label.replace("Skim_v43", "").replace("Skim", "").replace("_v43", "")
        
        # Map to standard physics process names
        label_mapping = {
            'QCD': 'QCD multijets',
            'WJets': 'W + jets', 
            'ZJets': 'Z + jets',
            'GJets': '#gamma + jets',
            'TTXJets': 't#bar{t} + X',
            'TTJets': 't#bar{t} + jets'
        }
        
        # Find matching process
        for key, clean_name in label_mapping.items():
            if key in clean_label:
                return clean_name
        
        # Fallback to cleaned label if no mapping found
        return clean_label.strip('_')

    def create_histogram(self, data, weights, bins, x_min, x_max, title, color=ROOT.kBlack, style=1):
        """Create and fill histogram with proper styling."""
        # Create the histogram
        name = f"h_{title}_{np.random.randint(0, 10000)}"
        hist = ROOT.TH1F(name, title, bins, x_min, x_max)
        hist.SetDirectory(0)
        hist.Sumw2()
        
        # Safety check and fill histogram
        if len(data) > 0 and len(weights) > 0:
            # Ensure arrays are same length
            min_len = min(len(data), len(weights))
            if min_len != len(data) or min_len != len(weights):
                print(f"Warning: Mismatched array lengths - data: {len(data)}, weights: {len(weights)}")
            
            # Use FillN for efficient filling
            data_array = np.array(data[:min_len], dtype=np.float64)
            weights_array = np.array(weights[:min_len], dtype=np.float64)
            hist.FillN(min_len, data_array, weights_array)

        # Styling
        hist.SetLineColor(ROOT.kBlack)
        hist.SetLineStyle(style)
        hist.SetLineWidth(1)
        hist.SetFillColor(color) 
        hist.SetStats(0)
        return hist

    
    def create_data_mc_comparison(self, data_collection, mc_collection, var_name, var_label, bins, x_min, x_max, blind_data=False, final_state_label=None, suffix="", normalized=False):
        """Create data/MC comparison plot with ratio panel using CMS styling."""
        canvas_name = f"datamc_{var_name}_{suffix}"
        if normalized:
            canvas_name = f"datamc_norm_{var_name}_{suffix}"
        
        # Ensure our MC colors are properly defined before creating histograms
        self._ensure_mc_colors()
        
        # Use CMS canvas with ratio panel setup
        canvas = CMS.cmsCanvas(canvas_name, x_min, x_max, 0, 1, var_label, "Events", square=False, extraSpace=0.01, iPos=0)
        canvas.SetCanvasSize(1100, 800)
        
        # Create pads for main plot and ratio
        pad1 = ROOT.TPad("pad1", "pad1", 0, 0.3, 1, 1.0)
        pad2 = ROOT.TPad("pad2", "pad2", 0, 0.0, 1, 0.3)
        
        pad1.SetBottomMargin(0.01)
        pad1.SetTopMargin(0.1)  # Increase top margin for more space above plot

        pad2.SetBottomMargin(0.4)
        
        canvas.cd()
        pad1.Draw()
        pad2.Draw()
        
        # Draw top pad with distributions
        pad1.cd()
        pad1.SetGridx(True)
        pad1.SetGridy(True)
        pad1.SetLogy(True)
        pad1.SetLeftMargin(self.style.margin_left+0.04)
        pad1.SetRightMargin(self.style.margin_right_ratio)
        
        # Create MC histograms
        mc_histograms = []
        for i, (filename, data) in enumerate(mc_collection.items()):
            color = self.mc_colors[i % len(self.mc_colors)]
            
            # Use the correct weights for this specific variable
            mapped_var = self._map_var_name(var_name)
            var_weights_key = f'{mapped_var}_weights'
            weights_to_use = data.get(var_weights_key, data.get('weights', []))
            
            h = self.create_histogram(data[mapped_var], weights_to_use, bins, x_min, x_max, filename, color=color)
            
            bg_name = self._clean_mc_label(parse_background_name(filename))
            mc_histograms.append((h, bg_name))
        
        # Sort MC histograms by yield (ascending order)
        mc_histograms.sort(key=lambda x: x[0].Integral())
        
        # Create THStack for MC
        stack = ROOT.THStack("stack", "")
        for mc_hist, _ in mc_histograms:
            stack.Add(mc_hist)
            
        # Apply normalization after histograms are created but before drawing
        if normalized:
            # Get total MC integral for normalization
            total_mc_integral = 0
            for mc_hist, _ in mc_histograms:
                total_mc_integral += mc_hist.Integral()
            
            # Normalize each MC histogram by the total MC integral
            if total_mc_integral > 0:
                for mc_hist, _ in mc_histograms:
                    mc_hist.Scale(1.0 / total_mc_integral)
        
        # Create data histogram (if not blinded)
        data_hist = None
        if not blind_data and data_collection:
            # Combine all data files
            combined_data = self._combine_data_collections(data_collection)
            if combined_data:
                # Use the correct weights for this specific variable
                mapped_var = self._map_var_name(var_name)
                var_weights_key = f'{mapped_var}_weights'
                weights_to_use = combined_data.get(var_weights_key, combined_data.get('weights', []))
                data_hist = self.create_histogram(combined_data[mapped_var], weights_to_use, bins, x_min, x_max, "data", color=self.data_color)
                data_hist.SetMarkerStyle(20)
                data_hist.SetMarkerSize(self.style.data_marker_size)
                data_hist.SetLineWidth(self.style.data_line_width)
                
                # Normalize data if requested
                if normalized and data_hist.Integral() > 0:
                    data_hist.Scale(1.0 / data_hist.Integral())
        
        # Set axis ranges
        data_max = data_hist.GetMaximum() if data_hist else 0
        stack_max = stack.GetMaximum()
        max_val = max(data_max, stack_max)
        
        if normalized:
            # For normalized plots, use fixed range that works with log scale
            stack.SetMinimum(2e-5)
            stack.SetMaximum(max_val * 10)
        else:
            # For regular plots, use the original scaling
            stack.SetMinimum(0.5)
            stack.SetMaximum(max_val * 10)
        
        # Draw stack and data with grid behind histograms
        stack.Draw("HIST")
        pad1.RedrawAxis("G")  # Draw grid lines behind everything  
        stack.Draw("HIST SAME")  # Redraw histogram content on top of grid
        stack.GetXaxis().SetLabelSize(0)
        
        # Set y-axis title based on normalization
        if normalized:
            # Extract just the variable name without units for the normalized y-axis title
            import re
            clean_var = re.sub(r'\s*\[.*?\]', '', var_label).strip()
            y_axis_title = f"#frac{{1}}{{N}}  #frac{{dN}}{{d({clean_var})}}"
        else:
            y_axis_title = "Events"
            
        stack.GetYaxis().SetTitle(y_axis_title)
        stack.GetYaxis().SetTitleSize(0.06)
        stack.GetYaxis().SetTitleOffset(1.1)
        stack.GetYaxis().SetLabelSize(0.05)
        stack.GetYaxis().CenterTitle(True)
        
        if data_hist:
            data_hist.Draw("PEX0 SAME")
        
        # Create CMS legend
        legend = CMS.cmsLeg(0.77, 0.5, 1., 0.9, textSize=0.045)
        
        if data_hist:
            legend.AddEntry(data_hist, "data", "lp")
        
        # Add MC backgrounds to legend (reverse order for stacking)
        for mc_hist, bg_name in reversed(mc_histograms):
            legend.AddEntry(mc_hist, bg_name, "f")
        
        legend.Draw()
        
        # Use common CMS label drawing with positions optimized for data/MC plots
        self.style.draw_cms_labels(cms_x=0.16, cms_y=0.93, prelim_str="Preliminary", prelim_x=0.235, lumi_x=0.75, cms_text_size_mult=1.25)
        
        # Draw region label
        if final_state_label:
            region_latex = self._draw_region_label(canvas, final_state_label, x_pos=0.4, y_pos=0.93, textsize=0.05, plot_type="datamc")
        
        # Draw bottom pad (ratio) only if data is available
        if data_hist:
            pad2.cd()
            pad2.SetGridx(True)
            pad2.SetGridy(True)
            pad2.SetLeftMargin(self.style.margin_left+0.04)
            pad2.SetRightMargin(self.style.margin_right_ratio)
            
            # Create total MC histogram for ratio
            total_mc_hist = mc_histograms[0][0].Clone("total_mc")
            total_mc_hist.Reset()
            for mc_hist, _ in mc_histograms:
                total_mc_hist.Add(mc_hist)
            
            # Create ratio histogram
            ratio_hist = data_hist.Clone("ratio")
            ratio_hist.Divide(total_mc_hist)
            
            # Style ratio plot
            ratio_hist.SetLineColor(self.data_color)
            ratio_hist.SetLineWidth(self.style.data_line_width)
            ratio_hist.SetMarkerColor(self.data_color)
            ratio_hist.SetMarkerStyle(20)
            ratio_hist.SetMarkerSize(self.style.data_marker_size)

            ratio_hist.SetTitle("")
            ratio_hist.GetXaxis().SetTitle(var_label)
            ratio_hist.GetYaxis().SetTitle("#frac{data}{model}")
            ratio_hist.GetYaxis().SetRangeUser(0.5, 1.5)
            ratio_hist.GetXaxis().SetTitleSize(0.14)
            ratio_hist.GetYaxis().SetTitleSize(0.14)
            ratio_hist.GetXaxis().SetLabelSize(0.12)
            ratio_hist.GetYaxis().SetLabelSize(0.12)
            ratio_hist.GetYaxis().SetTitleOffset(0.45)
            ratio_hist.GetXaxis().SetTitleOffset(1.15)
            ratio_hist.GetYaxis().SetNdivisions(505)
            ratio_hist.GetXaxis().CenterTitle(True)
            ratio_hist.GetYaxis().CenterTitle(True)
            ratio_hist.Draw("PEX0")
            
            # Reference line at 1
            x_min_ratio = ratio_hist.GetXaxis().GetXmin()
            x_max_ratio = ratio_hist.GetXaxis().GetXmax()
            line = ROOT.TLine(x_min_ratio, 1, x_max_ratio, 1)
            line.SetLineStyle(2)
            line.Draw()
            
            # Keep objects alive
            canvas.ratio_hist = ratio_hist
            canvas.total_mc_hist = total_mc_hist
            canvas.line = line

        pad1.SetFixedAspectRatio()
        pad2.SetFixedAspectRatio()
            
        canvas.Modified()
        canvas.Update()
        
        # Keep objects alive
        canvas.pad1 = pad1
        canvas.pad2 = pad2
        canvas.stack = stack
        canvas.mc_histograms = mc_histograms
        canvas.data_hist = data_hist
        canvas.legend = legend
        
        return canvas
    
    def _ensure_mc_colors(self):
        """Force recreation of MC colors at specific indices to override palette interference."""
        # Define the hex colors and expected indices from rootlogon.C
        hex_colors = ["#5A4484", "#347889", "#F4B240", "#E54B26", "#C05780", "#7A68A6", "#2E8B57", "#8B4513"]
        expected_indices = [1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186]
        
        for i, hex_color in enumerate(hex_colors):
            expected_index = expected_indices[i]
            
            # Parse hex color to RGB
            hex_clean = hex_color.replace("#", "")
            r = int(hex_clean[0:2], 16) / 255.0
            g = int(hex_clean[2:4], 16) / 255.0  
            b = int(hex_clean[4:6], 16) / 255.0
            
            # Get existing color or create new one
            existing_color = ROOT.gROOT.GetColor(expected_index)
            if existing_color:
                # Update existing color's RGB values
                existing_color.SetRGB(r, g, b)
            else:
                # Create new color at specific index
                ROOT.TColor(expected_index, r, g, b)
    
    def _combine_data_collections(self, data_collection):
        """Combine data from multiple files into single dataset."""
        if not data_collection:
            return None
            
        # Get the first file's data structure
        first_file_data = next(iter(data_collection.values()))
        combined_data = {}
        
        # Initialize with empty arrays
        for key in first_file_data.keys():
            combined_data[key] = []
        
        # Combine data from all files
        for file_data in data_collection.values():
            for key, values in file_data.items():
                if key in combined_data:
                    combined_data[key].extend(values)
        
        # Convert to numpy arrays
        for key in combined_data:
            combined_data[key] = np.array(combined_data[key])
        
        return combined_data
