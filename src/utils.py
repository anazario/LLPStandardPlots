import re
from pathlib import Path

def parse_signal_name(filename):
    """Parse signal file name to extract physics parameters"""
    stem = Path(filename).stem
    
    # Extract parameters using regex
    mgl_match = re.search(r'mGl-(\d+)', stem)
    mn2_match = re.search(r'mN2-(\d+)', stem)
    mn1_match = re.search(r'mN1-(\d+)', stem)
    ct_match = re.search(r'ct(\d+(?:p\d+)?)', stem)
    
    if all([mgl_match, mn2_match, mn1_match, ct_match]):
        mgl = mgl_match.group(1)
        mn2 = mn2_match.group(1)
        mn1 = mn1_match.group(1)
        ct = ct_match.group(1).replace('p', '.')
        
        return f"m_{{#tilde{{g}}}}({mgl})-m_{{#tilde{{#chi}}_{{2}}^{{0}}}}({mn2})-m_{{#tilde{{#chi}}_{{1}}^{{0}}}}({mn1}), c#tau={ct} m"
    else:
        return stem

def parse_background_name(filename):
    """Parse background file name to extract clean physics process name"""
    # Extract filename without path and extension
    stem = Path(filename).stem
    
    # Remove common suffixes (following unrolled plotting framework)
    clean_label = stem.replace("Skim_v43", "").replace("Skim", "").replace("_v43", "")
    
    # Map to standard physics process names
    label_mapping = {
        'QCD': 'QCD multijets',
        'WJets': 'W + jets', 
        'ZJets': 'Z + jets',
        'GJets': '#gamma + jets',
        'TTXJets': 't#bar{t} + X',
        'TTJets': 't#bar{t} + jets',
        'DYJets': 'Drell-Yan',
        'VV': 'Diboson',
        'SingleTop': 'Single top',
        'ST': 'Single top'
    }
    
    # Find matching process
    for key, clean_name in label_mapping.items():
        if key in clean_label:
            return clean_name
    
    # Fallback to cleaned label if no mapping found
    return clean_label.strip('_')
