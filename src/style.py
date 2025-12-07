import ROOT
import cmsstyle as CMS

class StyleManager:
    """Manages CMS plotting style configuration."""
    
    def __init__(self, luminosity=400, energy=13):
        self.luminosity = luminosity
        self.energy = energy
        self.color_palette = ROOT.kViridis
        
        # Text sizes and positions (consistent with original script)
        self.cms_text_size = 0.052
        self.prelim_text_size = 0.04
        self.lumi_text_size = 0.04
        self.sample_label_size = 0.04
        
        self.cms_x_pos = 0.122
        self.cms_y_pos = 0.955
        self.prelim_x_pos = 0.205
        self.lumi_x_pos = 0.819
        self.lumi_y_pos = 0.955

        # Axis & Canvas Defaults
        self.axis_title_offset = 1.3
        self.axis_title_size = 0.045
        self.axis_label_size = 0.04
        self.canvas_width = 800
        self.canvas_height = 600
        self.margin_left = 0.12
        self.margin_right = 0.12
        self.margin_right_ratio = 0.25

        # Data MC
        self.data_marker_size = 1.3
        self.data_line_width = 2
        
        # Colors from plotting.py
        self.colors = [
            ROOT.kBlue - 4,
            ROOT.kGreen - 3,
            ROOT.kPink - 8,
            ROOT.kOrange + 7,
            ROOT.kMagenta - 2,
            ROOT.kBlack,
            ROOT.kAzure + 2,
            ROOT.kGray
        ]

    def get_color(self, index):
        """Return a color from the defined palette."""
        return self.colors[index % len(self.colors)]

    def set_style(self):
        """Apply global CMS style settings."""
        CMS.SetExtraText("Preliminary")
        CMS.SetLumi(self.luminosity)
        ROOT.gROOT.SetBatch(True)
    
    def reset_palette_for_1d(self):
        """Reset palette to standard colors for 1D/data-MC plots."""
        ROOT.gStyle.SetPalette(ROOT.kBird)  # Reset to a standard palette

    def draw_cms_labels(self, cms_x=None, cms_y=None, prelim_str="Simulation", prelim_x=None, prelim_y=None, lumi_x=None, lumi_y=None, cms_text_size_mult=1.0):
        """
        Draw CMS, Preliminary, and Luminosity labels with customizable positions.
        Defaults are optimized for 2D plots.
        """
        # Use defaults if not provided
        cms_x = cms_x if cms_x is not None else self.cms_x_pos
        cms_y = cms_y if cms_y is not None else self.cms_y_pos
        prelim_x = prelim_x if prelim_x is not None else self.prelim_x_pos
        prelim_y = prelim_y if prelim_y is not None else cms_y # usually same y as CMS label
        lumi_x = lumi_x if lumi_x is not None else self.lumi_x_pos
        lumi_y = lumi_y if lumi_y is not None else cms_y

        # CMS Label
        cms_label = ROOT.TLatex()
        cms_label.SetNDC()
        cms_label.SetTextSize(self.cms_text_size * cms_text_size_mult)
        cms_label.SetTextFont(61)
        cms_label.DrawLatex(cms_x, cms_y, "CMS")
        
        # Preliminary Label
        cms_label.SetTextFont(52)
        cms_label.SetTextSize(self.prelim_text_size * cms_text_size_mult)
        cms_label.DrawLatex(prelim_x, prelim_y, prelim_str)
        
        # Luminosity Label
        lumi_label = ROOT.TLatex()
        lumi_label.SetNDC()
        lumi_label.SetTextSize(self.lumi_text_size * cms_text_size_mult)
        lumi_label.SetTextFont(42)
        lumi_label.SetTextAlign(31)  # Right-aligned
        lumi_label.DrawLatex(lumi_x, lumi_y, f"{self.luminosity} fb^{{-1}} ({self.energy} TeV)")

    def draw_process_label(self, label_text, x_pos=0.65, y_pos=0.85, align=11):
        """Draw the process/sample label on the plot."""
        latex = ROOT.TLatex()
        latex.SetTextFont(42)
        latex.SetNDC()
        latex.SetTextSize(self.sample_label_size)
        latex.SetTextAlign(align)
        latex.DrawLatex(x_pos, y_pos, label_text)
    
    def draw_region_label(self, canvas, label, x_pos, y_pos, textsize, plot_type="default"):
        """
        Draw region label with proper handling of \\ell\\ell symbols.
        plot_type: "1d", "2d", "datamc", or "default"
        Returns latex objects to keep alive.
        """
        if "\\ell\\ell" in label:
            # Split approach: draw main label with placeholder, then overlay \\ell\\ell
            main_label = label.replace("\\ell\\ell", "  ")  # Two spaces as placeholder
            
            # Draw main label
            main_latex = ROOT.TLatex()
            main_latex.SetNDC()
            main_latex.SetTextSize(textsize)
            main_latex.SetTextFont(42)
            main_latex.SetTextAlign(11)  # Left aligned
            main_text = f"Region: {main_label}"
            main_latex.DrawLatex(x_pos, y_pos, main_text)
            
            # Calculate position for \\ell\\ell using plot-type-specific offsets
            if plot_type == "1d":
                ell_x_offset = 0.14
                ell_y_offset = -0.014
                ell_size_offset = -0.013
            elif plot_type == "2d":
                ell_x_offset = 0.13  # Slightly different for 2D layouts
                ell_y_offset = -0.014
                ell_size_offset = -0.013
            elif plot_type == "datamc":
                ell_x_offset = 0.122  # Adjust for data/MC plot layout
                ell_y_offset = -0.014
                ell_size_offset = -0.013
            else:  # default
                ell_x_offset = 0.126
                ell_y_offset = -0.014
                ell_size_offset = -0.013
                
            ell_x_pos = x_pos + ell_x_offset
            ell_y_pos = y_pos + ell_y_offset
            ell_textsize = textsize + ell_size_offset
            
            # Draw \\ell\\ell with TMathText for proper italic rendering
            ell_latex = ROOT.TMathText()
            ell_latex.SetNDC() 
            ell_latex.SetTextSize(ell_textsize)
            ell_latex.DrawMathText(ell_x_pos, ell_y_pos, "\\ell\\ell")
            ell_latex.Paint()
            
            return [main_latex, ell_latex]  # Return both objects to keep alive
        else:
            # Simple case: no \\ell symbols
            fs_latex = ROOT.TLatex()
            fs_latex.SetNDC()
            fs_latex.SetTextSize(textsize)
            fs_latex.SetTextFont(42)
            fs_latex.SetTextAlign(11)  # Left aligned
            fs_latex.DrawLatex(x_pos, y_pos, f"Region: {label}")
            return fs_latex
