class AnalysisConfig:
    """Central configuration for analysis constants, cuts, and variable definitions."""
    
    # Selection Constants
    MET_CUT = 150.0
    RJR_PTS_CUT = 200.0      # rjrPTS[0] < 200, both modes
    EVT_WGT_CUT = 10.0

    # Kinematic baseline (lowest bin edges of the LLPCombine analysis configs), in branch units.
    # Applied with the N-1 treatment: a variable is plotted without its own cut.
    MS_CUT = 2000.0          # uncompressed: rjr_Ms[0] >= 2000 GeV
    RS_CUT = 0.15            # uncompressed: rjr_Rs[0] >= 0.15
    ISR_PT_CUT = 300.0       # compressed:   rjrIsr_PtIsr > 300 GeV (per-region override possible)
    ISR_RISR_CUT = 0.4       # compressed:   rjrIsr_RIsr >= 0.4
    
    # Variable Definitions
    # Key: Branch Name (in ROOT tree)
    # Value: Dict with plotting properties
    VARIABLES = {
        'rjr_Ms': {
            'name': 'ms',                 # Internal short name
            'label': 'M_{S} [TeV]',       # LaTeX label
            'bins': 30,
            'range': (0, 10),             # (min, max)
            'scale': 0.001,               # Scale factor (e.g. GeV -> TeV)
            'is_vector': True,            # If it requires the vector loop extraction
        },
        'rjr_Rs': {
            'name': 'rs',
            'label': 'R_{S}',
            'bins': 30,
            'range': (0, 1.0),
            'scale': 1.0,
            'is_vector': True,
        },
        # SV variables for data/MC comparison plots
        'HadronicSV_mass': {
            'name': 'had_mass',
            'label': 'mass [GeV]',
            'bins': 25,
            'range': (0, 100),
            'scale': 1.0,
            'is_vector': True
        },
        'HadronicSV_dxy': {
            'name': 'had_dxy',
            'label': 'd_{xy} [cm]',
            'bins': 15,
            'range': (0, 50),
            'scale': 1.0,
            'is_vector': True
        },
        'HadronicSV_dxySig': {
            'name': 'had_dxysig',
            'label': 'd_{xy}/#sigma_{d_{xy}}',
            'bins': 25,
            'range': (0, 800),
            'scale': 1.0,
            'is_vector': True
        },
        'HadronicSV_pOverE': {
            'name': 'had_povere',
            'label': 'p/E',
            'bins': 25,
            'range': (0.6, 1),
            'scale': 1.0,
            'is_vector': True
        },
        'HadronicSV_decayAngle': {
            'name': 'had_decayangle',
            'label': 'cos#theta_{CM}^{*}',
            'bins': 25,
            'range': (-1, 1),
            'scale': 1.0,
            'is_vector': True
        },
        'HadronicSV_cosTheta': {
            'name': 'had_costheta',
            'label': 'cos#theta',
            'bins': 25,
            'range': (0, 1),
            'scale': 1.0,
            'is_vector': True
        },
         'HadronicSV_nTracks': {
            'name': 'ntracks',
            'label': 'N_{tracks}',
            'bins': 15,
            'range': (5, 20),
            'scale': 1.0,
            'is_vector': True
        },
        'LeptonicSV_mass': {
            'name': 'lep_mass',
            'label': 'mass [GeV]',
            'bins': 25,
            'range': (0, 100),
            'scale': 1.0,
            'is_vector': True
        },
        'LeptonicSV_dxy': {
            'name': 'lep_dxy',
            'label': 'd_{xy} [cm]',
            'bins': 25,
            'range': (0, 50),
            'scale': 1.0,
            'is_vector': True
        },
        'LeptonicSV_dxySig': {
            'name': 'lep_dxysig',
            'label': 'd_{xy}/#sigma_{d_{xy}}',
            'bins': 25,
            'range': (0, 500),
            'scale': 1.0,
            'is_vector': True
        },
        'LeptonicSV_pOverE': {
            'name': 'lep_povere',
            'label': 'p/E',
            'bins': 25,
            'range': (0.6, 1),
            'scale': 1.0,
            'is_vector': True
        },
        'LeptonicSV_decayAngle': {
            'name': 'lep_decayangle',
            'label': 'cos#theta_{CM}^{*}',
            'bins': 25,
            'range': (-1, 1),
            'scale': 1.0,
            'is_vector': True
        },
        'LeptonicSV_cosTheta': {
            'name': 'lep_costheta',
            'label': 'cos#theta',
            'bins': 25,
            'range': (0.75, 1),
            'scale': 1.0,
            'is_vector': True
        },
        'InclusiveSV_mass': {
            'name': 'sv_mass',
            'label': 'mass [GeV]',
            'bins': 25,
            'range': (0, 100),
            'scale': 1.0,
            'is_vector': True
        },
        'InclusiveSV_dxy': {
            'name': 'sv_dxy',
            'label': 'd_{xy} [cm]',
            'bins': 25,
            'range': (0, 50),
            'scale': 1.0,
            'is_vector': True
        },
        'InclusiveSV_dxySig': {
            'name': 'sv_dxysig',
            'label': 'd_{xy}/#sigma_{d_{xy}}',
            'bins': 25,
            'range': (0, 800),
            'scale': 1.0,
            'is_vector': True
        },
        'InclusiveSV_pOverE': {
            'name': 'sv_povere',
            'label': 'p/E',
            'bins': 25,
            'range': (0.6, 1),
            'scale': 1.0,
            'is_vector': True
        },
        'InclusiveSV_decayAngle': {
            'name': 'sv_decayangle',
            'label': 'cos#theta_{CM}^{*}',
            'bins': 25,
            'range': (-1, 1),
            'scale': 1.0,
            'is_vector': True
        },
        'InclusiveSV_cosTheta': {
            'name': 'sv_costheta',
            'label': 'cos#theta',
            'bins': 25,
            'range': (0, 1),
            'scale': 1.0,
            'is_vector': True
        },
        'selCMet': {
            'name': 'met',
            'label': 'p_{T}^{miss} [GeV]',
            'bins': 50,
            'range': (0, 1000),
            'scale': 1.0,
            'is_vector': False
        },
        'nBaseLinePhotons': {
            'name': 'n_baseline_photons',
            'label': 'N_{#gamma}',
            'bins': 3,
            'range': (0, 3),
            'scale': 1.0,
            'is_vector': False
        },
        # ISR variables (compressed scenario)
        'rjrIsr_Ms': {
            'name': 'isr_ms',
            'label': 'M_{S}^{ISR} [GeV]',
            'bins': 50,
            'range': (0, 1600),
            'scale': 1.0,
            'is_vector': False
        },
        'rjrIsr_MsPerp': {
            'name': 'isr_msperp',
            'label': 'M_{#scale[0.7]{#perp}}^{S} [GeV]',
            'bins': 50,
            'range': (0, 1600),
            'scale': 1.0,
            'is_vector': False
        },
        'rjrIsr_PtIsr': {
            'name': 'isr_ptisr',
            'label': 'p_{T}^{ISR} [GeV]',
            'bins': 50,
            'range': (0, 2800),
            'scale': 1.0,
            'is_vector': False
        },
        'rjrIsr_RIsr': {
            'name': 'isr_risr',
            'label': 'R_{ISR}',
            'bins': 50,
            'range': (0, 1.0),
            'scale': 1.0,
            'is_vector': False
        },
        'rjrIsr_Rs': {
            'name': 'isr_rs',
            'label': 'R_{S}^{ISR}',
            'bins': 50,
            'range': (0, 1.0),
            'scale': 1.0,
            'is_vector': False
        },
        'rjrIsr_nSVisObjects': {
            'name': 'isr_nsvisobjects',
            'label': 'N_{S}^{vis}',
            'bins': 10,
            'range': (0, 10),
            'scale': 1.0,
            'is_vector': False
        },
        'rjrIsr_nIsrVisObjects': {
            'name': 'isr_nisrvisobjects',
            'label': 'N_{ISR}^{vis}',
            'bins': 12,
            'range': (0, 12),
            'scale': 1.0,
            'is_vector': False
        },
        'rjrIsrPTS': {
            'name': 'isr_pts',
            'label': 'p_{T}^{S, ISR} [GeV]',
            'bins': 50,
            'range': (0, 500),
            'scale': 1.0,
            'is_vector': False
        },
        'rjrIsrSdphiBV': {
            'name': 'isr_sdphibv',
            'label': '#Delta#phi_{BV}^{S, ISR}',
            'bins': 50,
            'range': (0, 3.15),
            'scale': 1.0,
            'is_vector': False
        },
        # Photon timing variables
        # baseLinePhoton_WTimeSig: available for both data and MC
        # baseLinePhoton_GenTimeSig, baseLinePhoton_GenLabTimeSig: MC only (no gen info in data)
        'baseLinePhoton_WTimeSig': {
            'name': 'photon_wtimesig',
            'label': 't_{#gamma}^{w}/#sigma_{t}',
            'bins': 60,
            'range': (-2.5, 2.5),
            'scale': 1.0,
            'is_vector': True,
            'mc_only': False
        },
        'baseLinePhoton_Pt': {
            'name': 'photon_pt',
            'label': 'p_{T}^{#gamma} [GeV]',
            'bins': 50,
            'range': (0, 1000),
            'scale': 1.0,
            'is_vector': True,
            'mc_only': False
        },
        'baseLinePhoton_Eta': {
            'name': 'photon_eta',
            'label': '#eta_{#gamma}',
            'bins': 50,
            'range': (-2.5, 2.5),
            'scale': 1.0,
            'is_vector': True,
            'mc_only': False
        },
        'baseLinePhoton_isoANNScore': {
            'name': 'photon_isoann',
            'label': 'Photon iso ANN score',
            'bins': 50,
            'range': (0, 1),
            'scale': 1.0,
            'is_vector': True,
            'mc_only': False
        },
        'baseLinePhoton_GenTimeSig': {
            'name': 'photon_gentimesig',
            'label': 't_{#gamma}^{gen}/#sigma_{t}',
            'bins': 60,
            'range': (-5, 10),
            'scale': 1.0,
            'is_vector': True,
            'mc_only': True
        },
        'baseLinePhoton_GenLabTimeSig': {
            'name': 'photon_genlabsig',
            'label': 't_{#gamma,lab}^{gen}/#sigma_{t}',
            'bins': 60,
            'range': (-5, 10),
            'scale': 1.0,
            'is_vector': True,
            'mc_only': True
        }
    }

    @staticmethod
    def get_var_config(short_name):
        """Helper to find config by short name (e.g. 'ms')"""
        for branch, conf in AnalysisConfig.VARIABLES.items():
            if conf['name'] == short_name:
                return conf
        return None


class AnalysisMode:
    """Analysis mode constants."""
    UNCOMPRESSED = 'uncompressed'
    COMPRESSED = 'compressed'


class ModeConfig:
    """Mode-specific configuration for analysis types."""

    UNCOMPRESSED = {
        'default_vars': ['rjr_Ms', 'rjr_Rs'],
        'plot_2d_configs': [
            {'x_var': 'rjr_Ms', 'y_var': 'rjr_Rs', 'suffix': 'Ms_vs_Rs'}
        ],
        'branches': ['rjr_Ms', 'rjr_Rs', 'rjrPTS'],
        'baseline_var': 'rjrPTS',
    }

    COMPRESSED = {
        'default_vars': ['rjrIsr_PtIsr', 'rjrIsr_RIsr'],
        'plot_2d_configs': [
            {'x_var': 'rjrIsr_PtIsr', 'y_var': 'rjrIsr_RIsr', 'suffix': 'ptISR_vs_RISR'},
        ],
        'branches': ['rjrIsr_PtIsr', 'rjrIsr_RIsr', 'rjrIsr_nSVisObjects', 'rjrPTS'],
        'isr_pt_cut_default': AnalysisConfig.ISR_PT_CUT,
    }

    @classmethod
    def get(cls, mode):
        """Get configuration for the specified analysis mode."""
        return cls.COMPRESSED if mode == AnalysisMode.COMPRESSED else cls.UNCOMPRESSED


def kinematic_cuts(mode, isr_pt_cut=None):
    """
    The kinematic baseline of a mode as [(branch, op, threshold)], threshold in plotted
    units (branch value x VARIABLES[branch]['scale']), as compared by kinematic_pass.

    Every plotted variable gets all of these cuts except the one on itself (N-1).
    isr_pt_cut overrides the compressed PtISR threshold (None -> AnalysisConfig.ISR_PT_CUT).
    """
    V = AnalysisConfig.VARIABLES
    if mode == AnalysisMode.COMPRESSED:
        pt = AnalysisConfig.ISR_PT_CUT if isr_pt_cut is None else float(isr_pt_cut)
        return [('rjrIsr_PtIsr', '>', pt * V['rjrIsr_PtIsr']['scale']),
                ('rjrIsr_RIsr', '>=', AnalysisConfig.ISR_RISR_CUT * V['rjrIsr_RIsr']['scale'])]
    return [('rjr_Ms', '>=', AnalysisConfig.MS_CUT * V['rjr_Ms']['scale']),
            ('rjr_Rs', '>=', AnalysisConfig.RS_CUT * V['rjr_Rs']['scale'])]


def kinematic_pass(value, op, threshold):
    """value <op> threshold; a missing value (None / NaN) fails. Works element-wise on arrays."""
    import numpy as np
    if value is None:
        return False
    if np.ndim(value) == 0:   # per-event scalar: plain comparison (NaN compares False)
        return bool(value > threshold) if op == '>' else bool(value >= threshold)
    value = np.asarray(value, dtype=float)
    return value > threshold if op == '>' else value >= threshold


def describe_kinematic_cuts(mode, isr_pt_cut=None):
    """Human-readable cut strings in branch units, e.g. 'rjr_Ms[0] >= 2000'."""
    V = AnalysisConfig.VARIABLES
    out = []
    for branch, op, thr in kinematic_cuts(mode, isr_pt_cut):
        name = f"{branch}[0]" if V[branch]['is_vector'] else branch
        out.append(f"{name} {op} {thr / V[branch]['scale']:g}")
    return out


def plane_key(x_var, y_var):
    """Prefix of the paired 2D-plane arrays the loader stores: <prefix>x, <prefix>y, <prefix>w."""
    return f"plane2d__{x_var}__{y_var}__"
