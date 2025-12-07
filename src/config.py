class AnalysisConfig:
    """Central configuration for analysis constants, cuts, and variable definitions."""
    
    # Selection Constants
    MET_CUT = 150.0
    RJR_PTS_CUT = 150.0
    EVT_WGT_CUT = 10.0
    
    # Variable Definitions
    # Key: Branch Name (in ROOT tree)
    # Value: Dict with plotting properties
    VARIABLES = {
        'rjr_Ms': {
            'name': 'ms',                 # Internal short name
            'label': 'M_{S} [TeV]',       # LaTeX label
            'bins': 50, 
            'range': (0, 10),             # (min, max)
            'scale': 0.001,               # Scale factor (e.g. GeV -> TeV)
            'is_vector': True             # If it requires the vector loop extraction
        },
        'rjr_Rs': {
            'name': 'rs',
            'label': 'R_{S}',
            'bins': 50,
            'range': (0, 1.0),
            'scale': 1.0,
            'is_vector': True
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
            'bins': 50,
            'range': (0, 50),
            'scale': 1.0,
            'is_vector': True
        },
        'HadronicSV_dxySig': {
            'name': 'had_dxysig',
            'label': 'd_{xy}/#sigma_{d_{xy}}',
            'bins': 25,
            'range': (0, 1000),
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
            'bins': 50,
            'range': (0, 50),
            'scale': 1.0,
            'is_vector': True
        },
        'LeptonicSV_dxySig': {
            'name': 'lep_dxysig',
            'label': 'd_{xy}/#sigma_{d_{xy}}',
            'bins': 25,
            'range': (0, 1000),
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
        'selCMet': {
            'name': 'met',
            'label': 'p_{T}^{miss} [GeV]',
            'bins': 50,
            'range': (0, 1000),
            'scale': 1.0,
            'is_vector': False
        }
    }

    @staticmethod
    def get_var_config(short_name):
        """Helper to find config by short name (e.g. 'ms')"""
        for branch, conf in AnalysisConfig.VARIABLES.items():
            if conf['name'] == short_name:
                return conf
        return None
