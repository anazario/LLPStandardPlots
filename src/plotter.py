import ROOT
import numpy as np
import cmsstyle as CMS
from src.style import StyleManager
from src.utils import parse_signal_name, parse_background_name
from src.config import AnalysisConfig
from src.unrolled import UnrolledBinning

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
        self.style.draw_cms_labels(cms_x=0.16, cms_y=self.style.cms_y_pos, prelim_x=0.248, lumi_x=0.9, cms_text_size_mult=1.)
        
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
        self.style.draw_cms_labels(cms_x=0.16, cms_y=self.style.cms_y_pos, prelim_x=0.248, lumi_x=0.9, cms_text_size_mult=1.)
        
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
        hist.GetXaxis().SetTitleOffset(1.25)
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
        self._draw_region_label(canvas, final_state_label, x_pos=0.425, plot_type="2d")
        
        canvas.Update() # Update after adding labels
        
        return canvas, hist

class PlotterDataMC(PlotterBase):
    def __init__(self, style_manager):
        super().__init__(style_manager)
        # Define MC colors - indices for colors created by _ensure_mc_colors()
        self.mc_colors = [1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186]
        self.data_color = ROOT.kBlack
    
    def _setup_comparison_canvas(self, canvas_name, x_min, x_max, x_label="", canvas_width=1100, canvas_height=800):
        """Shared canvas setup for data/MC comparison plots."""
        canvas = CMS.cmsCanvas(canvas_name, x_min, x_max, 0, 1, x_label, "Events", square=False, extraSpace=0.01, iPos=0)
        canvas.SetCanvasSize(canvas_width, canvas_height)
        
        # Create pads for main plot and ratio
        pad1 = ROOT.TPad("pad1", "pad1", 0, 0.3, 1, 1.0)
        pad2 = ROOT.TPad("pad2", "pad2", 0, 0.0, 1, 0.3)
        
        pad1.SetBottomMargin(0.01)
        pad1.SetTopMargin(0.1)
        pad2.SetBottomMargin(0.4)
        
        canvas.cd()
        pad1.Draw()
        pad2.Draw()
        
        return canvas, pad1, pad2
    
    def _setup_main_pad(self, pad1, use_custom_grid=False):
        """Shared main pad setup."""
        pad1.cd()
        if use_custom_grid:
            pad1.SetGridx(False)  # Custom grid for unrolled
        else:
            pad1.SetGridx(True)
        pad1.SetGridy(True)
        pad1.SetLogy(True)
        pad1.SetLeftMargin(self.style.margin_left+0.04)
        pad1.SetRightMargin(self.style.margin_right_ratio)
    
    
    def _create_mc_uncertainty_band(self, mc_histograms):
        """Create MC uncertainty band with existing data/MC styling."""
        if not mc_histograms:
            return None
        
        # Get the sum of all MC histograms for uncertainty band (like original code)
        total_mc = None
        for mc_hist, _ in mc_histograms:
            if total_mc is None:
                total_mc = mc_hist.Clone("total_mc_for_uncertainty")
            else:
                total_mc.Add(mc_hist)
        
        # Create and style MC uncertainty band (EXACT existing styling)
        mc_uncertainty = total_mc.Clone("mc_uncertainty")
        mc_uncertainty.SetFillStyle(3244)
        ROOT.gStyle.SetHatchesLineWidth(1)
        mc_uncertainty.SetFillColor(ROOT.kBlack)
        mc_uncertainty.SetLineColor(ROOT.kBlack)
        mc_uncertainty.SetLineWidth(1)
        mc_uncertainty.SetMarkerSize(0)
        mc_uncertainty.SetMarkerStyle(0)
        # Don't draw here - caller will draw at proper time
        
        return mc_uncertainty
    
    def _create_standard_legend(self, data_hist, mc_uncertainty, mc_histograms, x1=0.77, x2=1., y1=0.4, y2=0.9, text_size=0.05):
        """Create legend with existing data/MC ordering and styling."""
        legend = CMS.cmsLeg(x1, y1, x2, y2, textSize=text_size)
        
        if data_hist:
            legend.AddEntry(data_hist, "data", "lp")
        
        # Add MC uncertainty band (second after data - existing order)
        if mc_uncertainty:
            legend.AddEntry(mc_uncertainty, "total uncertainty", "f")
        
        # Add MC backgrounds (reverse order for stacking - existing order)
        for mc_hist, bg_name in reversed(mc_histograms):
            legend.AddEntry(mc_hist, bg_name, "f")
        
        return legend
    
    def _draw_standard_labels(self, canvas, final_state_label):
        self.style.draw_cms_labels(cms_x=0.16, cms_y=0.93, prelim_str="Preliminary", prelim_x=0.235, lumi_x=0.75, cms_text_size_mult=1.25)
        
        if final_state_label:
            self._draw_region_label(canvas, final_state_label, x_pos=0.4, y_pos=0.93, textsize=0.05, plot_type="datamc")
    
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
        
        # Use shared canvas setup
        canvas, pad1, pad2 = self._setup_comparison_canvas(canvas_name, x_min, x_max, var_label)
        
        # Setup main pad with standard data/MC grid
        pad1.cd() 
        pad1.SetGridx(True)
        pad1.SetGridy(True)
        pad1.SetLogy(True)
        pad1.SetLeftMargin(self.style.margin_left+0.04)
        pad1.SetRightMargin(self.style.margin_right_ratio)
        
        # Create MC histograms
        mc_histograms = []
        for i, (filename, data) in enumerate(mc_collection.items()):
            color = self._get_background_color_index(filename)
            
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
        
        # Draw stack and data with grid behind histograms
        stack.Draw("HIST")
        pad1.RedrawAxis("G")  # Draw grid lines behind everything  
        stack.Draw("HIST SAME")  # Redraw histogram content on top of grid
        
        # Set axis ranges after all drawing is complete
        if normalized:
            # For normalized plots, use fixed range that works with log scale
            stack.SetMaximum(max_val * 5.)
            stack.SetMinimum(2e-4)
            stack.GetHistogram().GetYaxis().SetRangeUser(2e-4, max_val * 5.)
        else:
            # For regular plots, use the original scaling
            stack.GetHistogram().GetYaxis().SetRangeUser(0.5, max_val * 10)
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
        
        # Create and add MC uncertainty band using helper
        mc_uncertainty = self._create_mc_uncertainty_band(mc_histograms)
        if mc_uncertainty:
            mc_uncertainty.Draw("E2 SAME")  # E2 = error band only (no markers)
        
        if data_hist:
            data_hist.Draw("PEX0 SAME")
        
        # Create legend using helper
        legend = self._create_standard_legend(data_hist, mc_uncertainty, mc_histograms)
        legend.Draw()
        
        # Draw standard labels using helper
        #self._draw_standard_labels(canvas, final_state_label)
        self.style.draw_cms_labels(cms_x=0.16, cms_y=0.93, prelim_str="Preliminary", prelim_x=0.235, lumi_x=0.75, cms_text_size_mult=1.25)

        if final_state_label:
            self._draw_region_label(canvas, final_state_label, x_pos=0.4, y_pos=0.93, textsize=0.05, plot_type="datamc")
        
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
        canvas.mc_uncertainty = mc_uncertainty
        canvas.data_hist = data_hist
        canvas.legend = legend
        
        return canvas
    
    def _get_background_color_index(self, filename):
        """Get the color index for a specific background based on physics process."""
        from src.utils import parse_background_name
        
        bg_name = parse_background_name(filename)
        
        # Background to color mapping (using original hex color indices)
        # QCD=purple, WJets=teal, ZJets=yellow/gold, TTX=red/orange, GJets=pink/rose
        background_color_map = {
            'QCD multijets': 1179,        # #5A4484 - Purple
            'W + jets': 1180,             # #347889 - Teal/Blue-green
            'Z + jets': 1181,             # #F4B240 - Yellow/Gold
            't#bar{t} + X': 1182,         # #E54B26 - Red/Orange
            't#bar{t} + jets': 1182,      # #E54B26 - Red/Orange (same as TTX)
            '#gamma + jets': 1183,        # #C05780 - Pink/Rose
            # Assign remaining backgrounds to remaining colors
            'Drell-Yan': 1184,            # #7A68A6 - Light purple
            'Diboson': 1185,              # #2E8B57 - Sea green
            'Single top': 1186,           # #8B4513 - Saddle brown
        }
        
        return background_color_map.get(bg_name, 1179)  # Default to purple if not found

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

    def create_unrolled_comparison(self, data_collection, mc_collection, scheme="merged_rs", blind_data=False, final_state_label=None, suffix="", normalized=False):
        """
        Creates an unrolled Data/MC comparison plot using the UnrolledBinning logic.
        Reuse as much of standard comparison logic as possible.
        """
        unroller = UnrolledBinning(scheme=scheme)

        right_margin = 0.22
        left_margin = 0.13
        
        # --- 1. Process MC Data ---
        mc_histograms = []
        for filename, data in mc_collection.items():
            # Calculate 3x3 yields
            y2d, e2d = unroller.calculate_2d_yields(data['rjr_Ms'], data['rjr_Rs'], data['weights'])
            # Unroll
            y1d, e1d, bin_labels, decorations = unroller.unroll(y2d, e2d)
            
            # Create Histogram - create empty histogram and set bin contents directly
            color = self._get_background_color_index(filename)
            nbins = len(y1d)
            
            # Create empty histogram with proper binning
            name = f"h_unrolled_{filename}_{np.random.randint(0, 10000)}"
            h = ROOT.TH1F(name, filename, nbins, 0, nbins)
            h.SetDirectory(0)
            h.Sumw2()
            
            # Set bin contents and errors directly from unrolled yields
            for i in range(nbins):
                h.SetBinContent(i+1, y1d[i])
                h.SetBinError(i+1, e1d[i])
            
            # Set colors
            h.SetFillColor(color)
            h.SetLineColor(ROOT.kBlack)
            h.SetLineWidth(1)
            h.SetStats(0)
            
            bg_name = self._clean_mc_label(parse_background_name(filename))
            mc_histograms.append((h, bg_name))
        
        # Sort MC by yield
        mc_histograms.sort(key=lambda x: x[0].Integral())
        
        # --- 2. Process Data ---
        data_hist = None
        if not blind_data and data_collection:
            combined_data = self._combine_data_collections(data_collection)
            if combined_data:
                y2d, e2d = unroller.calculate_2d_yields(combined_data['rjr_Ms'], combined_data['rjr_Rs'], combined_data['weights'])
                y1d, e1d, bin_labels, decorations = unroller.unroll(y2d, e2d)
                
                nbins = len(y1d)
                
                # Create empty data histogram and set bin contents directly  
                name = f"h_unrolled_data_{np.random.randint(0, 10000)}"
                data_hist = ROOT.TH1F(name, "data", nbins, 0, nbins)
                data_hist.SetDirectory(0)
                data_hist.Sumw2()
                
                # Set bin contents and errors directly from unrolled yields
                for i in range(nbins):
                    data_hist.SetBinContent(i+1, y1d[i])
                    data_hist.SetBinError(i+1, e1d[i])
                
                # Set data styling
                data_hist.SetLineColor(self.data_color)
                data_hist.SetMarkerColor(self.data_color)
                
                data_hist.SetMarkerStyle(20)
                data_hist.SetMarkerSize(self.style.data_marker_size)
                data_hist.SetLineWidth(self.style.data_line_width)

        # --- 3. Normalization ---
        if normalized:
             # Normalize MC Stack
            total_mc_integral = sum(h[0].Integral() for h in mc_histograms)
            if total_mc_integral > 0:
                for h, _ in mc_histograms:
                    h.Scale(1.0 / total_mc_integral)
            
            # Normalize Data
            if data_hist and data_hist.Integral() > 0:
                data_hist.Scale(1.0 / data_hist.Integral())

        # --- 4. Plotting Infrastructure (Reuse internal logic if possible, or replicate essential parts) ---
        # We replicate essential parts of create_data_mc_comparison but adapt for custom axis labels and decorations
        
        canvas_name = f"unrolled_{scheme}_{suffix}"
        if normalized:
            canvas_name = f"unrolled_{scheme}_norm_{suffix}"
            
        self._ensure_mc_colors()
        
        # Use shared canvas setup with original unrolled dimensions
        canvas, pad1, pad2 = self._setup_comparison_canvas(canvas_name, 0, nbins, "", canvas_width=1200, canvas_height=600)
        
        # Setup main pad with exact original unrolled margins
        pad1.cd() 
        pad1.SetGridx(True)
        pad1.SetGridy(True)
        pad1.SetLogy(True)
        pad1.SetLeftMargin(left_margin)
        pad1.SetRightMargin(right_margin)
        pad1.SetBottomMargin(0.0)
        
        # Stack
        stack = ROOT.THStack("stack", "")
        for h, _ in mc_histograms:
            stack.Add(h)

        stack.Draw("HIST")
        pad1.RedrawAxis("G")  # Draw grid lines behind everything
        stack.Draw("HIST SAME")  # Redraw histogram content on top of grid
            
        # Range
        data_max = data_hist.GetMaximum() if data_hist else 0
        stack_max = stack.GetMaximum()
        max_val = max(data_max, stack_max)
        
        # Ensure positive max_val for log scale
        if max_val <= 0:
            max_val = 1.0
        
        # Apply EXACT same range logic as data/MC
        if normalized:
            # Use same normalized y-title format as data/MC 
            if "rs" in scheme.lower():
                import re
                clean_var = "R_{S}"  
                y_title = f"#frac{{1}}{{N}}  #frac{{dN}}{{d({clean_var})}}"
            else:  # ms scheme
                import re  
                clean_var = "M_{S}"
                y_title = f"#frac{{1}}{{N}}  #frac{{dN}}{{d({clean_var})}}"
            stack.SetMaximum(max_val * 5.)
            stack.SetMinimum(2e-4) 
            stack.GetHistogram().GetYaxis().SetRangeUser(2e-4, max_val * 5.)
        else:
            y_title = "number of events"
            stack.GetHistogram().GetYaxis().SetRangeUser(0.5, max_val * 10)
            stack.SetMaximum(max_val * 10.)
            stack.SetMinimum(0.5)
            stack.GetHistogram().GetYaxis().SetRangeUser(0.5, max_val * 10.)
            
        # Apply original unrolled axis formatting
        stack.GetYaxis().SetTitle(y_title)
        stack.GetYaxis().SetTitleSize(0.075)
        stack.GetYaxis().SetTitleOffset(0.75)
        stack.GetYaxis().SetLabelSize(0.07)
        stack.GetYaxis().CenterTitle(True)
        stack.GetXaxis().SetLabelSize(0)      # Hide default numbers
        
        # Create MC uncertainty using shared helper (EXACT same styling as data/MC)
        mc_uncertainty = self._create_mc_uncertainty_band(mc_histograms)
        if mc_uncertainty:
            mc_uncertainty.Draw("E2 SAME")

        if data_hist:
            data_hist.Draw("PEX0 SAME")
            
        # --- Unrolled Decorations ---
        # 1. Vertical Separator Lines (using proper NDC implementation)
        line_objs = unroller.add_separator_lines(canvas, mc_histograms[0][0] if mc_histograms else data_hist, scheme)
             
        # Switch to pad1 for all text elements (same as regular data/MC plots)
        pad1.cd()
        pad1.SetGridy(True)
        
        # 2. Group Labels (Top of pad) - original unrolled formatting
        latex = ROOT.TLatex()
        latex.SetNDC(True)  # Use NDC coordinates
        latex.SetTextFont(42)
        latex.SetTextSize(0.05)  # Original group label size
        latex.SetTextAlign(22)  # Center alignment
        
        # Get pad margins for NDC conversion (same as standard data/MC)
        main_pad = pad1
        left_ndc = main_pad.GetLeftMargin()
        right_ndc = 1.0 - main_pad.GetRightMargin()
        data_ndc_width = right_ndc - left_ndc
        
        label_objs = []
        for grp in decorations['group_labels']:
            x_start, x_end = grp['x_range']
            x_center = (x_start + x_end) / 2.0
            
            # Convert bin center to NDC coordinates 
            x_ndc = left_ndc + (x_center / nbins) * data_ndc_width
            y_ndc = 0.83  # Original unrolled group label y-position
            
            label_text = grp['text']
            if label_text:  # Only draw non-empty labels
                latex.DrawLatex(x_ndc, y_ndc, label_text)
        
        # 3. Add individual labels for merged schemes (exactly like original)
        individual_label_objs = unroller.add_individual_labels(canvas, scheme)
        
        # Create legend using original unrolled positioning
        legend = self._create_standard_legend(data_hist, mc_uncertainty, mc_histograms, x1=0.8, x2=1.0, y1=0.25, y2=0.9, text_size=0.065)
        legend.Draw()
        
        # Draw labels using original unrolled positioning
        self.style.draw_cms_labels(cms_x=0.13, cms_y=0.93, prelim_str="Preliminary", prelim_x=0.202, lumi_x=0.78, cms_text_size_mult=1.75)

        if final_state_label:
            self._draw_region_label(canvas, final_state_label, x_pos=0.46, y_pos=0.93, textsize=0.065, plot_type="unrolled")

        # --- Draw Bottom Pad (Ratio) ---
        if data_hist:
            pad2.cd()
            pad2.SetGridx(True)
            pad2.SetGridy(True)
            pad2.SetLeftMargin(left_margin)
            pad2.SetRightMargin(right_margin)
            pad2.SetTopMargin(0.0)
            pad2.SetBottomMargin(0.4)
            
            total_mc_hist = mc_histograms[0][0].Clone("total_mc_ratio")
            total_mc_hist.Reset()
            for h, _ in mc_histograms:
                total_mc_hist.Add(h)
                
            ratio_hist = data_hist.Clone("ratio")
            ratio_hist.SetStats(0)
            ratio_hist.Divide(total_mc_hist)
            
            # Style Ratio
            ratio_hist.SetLineColor(self.data_color)
            ratio_hist.SetLineWidth(self.style.data_line_width)
            ratio_hist.SetMarkerColor(self.data_color)
            ratio_hist.SetMarkerStyle(20)
            ratio_hist.SetMarkerSize(self.style.data_marker_size)
            
            ratio_hist.SetTitle("")
            ratio_hist.GetYaxis().SetTitle("#frac{data}{model}")
            ratio_hist.GetYaxis().SetRangeUser(0.0, 1.9) # Wider range for unrolled usually
            
            # Setup X-axis Labels (Custom Bins)
            ax = ratio_hist.GetXaxis()
            for i, label in enumerate(bin_labels):
                ax.SetBinLabel(i+1, label)
            
            # Set bin label size - use only one method
            ax.SetLabelSize(0.25)  # Increased size for better visibility
            
            ratio_hist.GetYaxis().SetTitleSize(0.18)
            ratio_hist.GetYaxis().SetLabelSize(0.16)
            ratio_hist.GetYaxis().SetTitleOffset(0.32)
            ratio_hist.GetYaxis().SetNdivisions(505)
            ratio_hist.GetYaxis().CenterTitle(True)

            ax.CenterTitle(True)
            
            # Add x-axis title based on scheme
            x_title = "R_{S}" if "rs" in scheme else "M_{S} [TeV]"
            ax.SetTitle(x_title)
            ax.SetTitleSize(0.18)
            ax.SetTitleOffset(1.)
            
            ratio_hist.Draw("PEX0")
            
            # Force update to apply text changes
            pad2.Modified()
            pad2.Update()
            
            # Line at 1
            line = ROOT.TLine(0, 1, nbins, 1)
            line.SetLineStyle(2)
            line.Draw()
            
            ratio_lines = []

        # NOW add centered labels after both pads are drawn (overlay on top)
        centered_label_objs = unroller.add_merged_centered_labels(canvas, scheme)

        if data_hist:
            # Keep objects
            canvas.ratio_hist = ratio_hist
            canvas.ratio_lines = ratio_lines
            canvas.line = line

        canvas.Update()
        
        # Keep references
        canvas.pad1 = pad1
        canvas.pad2 = pad2
        canvas.stack = stack
        canvas.mc_histograms = mc_histograms
        canvas.mc_uncertainty = mc_uncertainty
        canvas.data_hist = data_hist
        canvas.legend = legend
        canvas.line_objs = line_objs
        canvas.individual_label_objs = individual_label_objs
        canvas.centered_label_objs = centered_label_objs
        
        return canvas
