import re


# ---------------------------------------------------------------------------- #
# Region classification / blinding                                              #
# ---------------------------------------------------------------------------- #
# Skimmer selection flags follow the naming convention (see KUCMSSkimmer
# src/KUCMSAodSkimmer_SV.cc and src/KUCMSAodSkimmer_Photons.cc):
#   ...SR             signal region               (blind)
#   ...CR             control region              (unblind)
#   ...ValSR/ValCR    validation region, always a CR by construction (unblind)
#   no suffix         inclusive selection that overlaps the SRs, e.g.
#                     passNSVGe1Selection, passNSVEq0Selection,
#                     passNPhoGe1NonPrompt, passNPhoGe1SelectionLateSignal
# A region is the AND of the flags it requires; it is blinded when it requires
# an SR flag, or when it requires an inclusive flag without also requiring a CR
# flag (e.g. LateSignal alone overlaps the delayed-photon SR, while
# LowDxySigCR + LateSignal is the MixDel anchor CR). Vetoed flags (== 0) never
# make a region an SR.

FLAG_SR = 'sr'
FLAG_CR = 'cr'
FLAG_INCLUSIVE = 'inclusive'

_FLAG_TOKEN_RE = re.compile(
    r'(!\s*)?\b(pass\w+)\b(?:\s*(==|!=|>=|<=|>|<)\s*(-?\d+(?:\.\d+)?))?')


def classify_selection_flag(flag):
    """Return FLAG_SR, FLAG_CR or FLAG_INCLUSIVE for a skimmer selection flag."""
    if re.search(r'Val(SR|CR)', flag):
        return FLAG_CR
    if 'SR' in flag:
        return FLAG_SR
    if 'CR' in flag:
        return FLAG_CR
    return FLAG_INCLUSIVE


def _required_flags_in_cut(cut_string):
    """Return the selection flags a custom cut string requires to be true."""
    required = []
    for negated, flag, op, value in _FLAG_TOKEN_RE.findall(cut_string):
        if op:
            value = float(value)
            requires = ((op == '==' and value == 1) or (op == '!=' and value == 0) or
                        (op == '>=' and value == 1) or (op == '>' and value == 0))
        else:
            requires = True
        if negated:
            requires = not requires
        if requires:
            required.append(flag)
    return required


def _and_group_is_blinded(flags):
    kinds = {classify_selection_flag(flag) for flag in flags}
    if FLAG_SR in kinds:
        return True
    return FLAG_INCLUSIVE in kinds and FLAG_CR not in kinds


def _split_top_level(expr, op):
    """Split expr on the single-character operator op outside parentheses."""
    parts, depth, current = [], 0, []
    for ch in expr:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == op and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    parts.append(''.join(current))
    return [p.strip() for p in parts if p.strip()]


def _strip_outer_parens(expr):
    expr = expr.strip()
    while expr.startswith('(') and expr.endswith(')'):
        depth = 0
        for i, ch in enumerate(expr):
            depth += (ch == '(') - (ch == ')')
            if depth == 0 and i < len(expr) - 1:
                return expr
        expr = expr[1:-1].strip()
    return expr


def _cut_and_groups(expr):
    """Expand a cut string into its OR of AND-groups of required flags."""
    expr = _strip_outer_parens(expr)
    or_parts = _split_top_level(expr, '|')
    if len(or_parts) > 1:
        return [group for part in or_parts for group in _cut_and_groups(part)]
    and_parts = _split_top_level(expr, '&')
    if len(and_parts) > 1:
        groups = [[]]
        for part in and_parts:
            groups = [a + b for a in groups for b in _cut_and_groups(part)]
        return groups
    if expr.startswith('!') and _strip_outer_parens(expr[1:]) != expr[1:].strip():
        return [[]]  # negated sub-expression: a veto never requires a flag
    return [_required_flags_in_cut(expr)]


def is_blinded_selection(selection):
    """
    Decide whether data must be blinded for an event-flag expression
    ('A+B|C', '+' = AND, '|' = OR) or a custom cut string.

    The selection is expanded into an OR of AND-groups of required flags and
    is blinded if any group is. Custom cuts that reference no selection flags
    are not blinded here; use 'blind: true' in the input config for those.
    """
    if selection.startswith('pass'):
        return any(
            _and_group_is_blinded([f.strip() for f in or_part.split('+')])
            for or_part in selection.split('|')
        )

    normalized = selection.replace('&&', '&').replace('||', '|')
    return any(_and_group_is_blinded(group) for group in _cut_and_groups(normalized))


class FinalStateResolver:
    """Resolves final state flags to LaTeX labels."""

    # Ordered photon keyword -> (timing subscript, descriptors). Checked most
    # specific first so e.g. "LateNotBHTightIso" matches before "LateNotBH".
    _PHOTON_REGION_MAP = [
        ('EarlyBeamHalo',     't-',   ['BH']),
        ('LateBeamHalo',      't+',   ['BH']),
        ('BeamHalo',          '',     ['BH']),
        ('EarlyNotBH',        't-',   ['!BH']),
        ('LateNotBHTightIso', 't+',   ['!BH', 'tight iso']),
        ('LateSignal',        't+',   ['!BH', 'tight iso']),
        ('EarlyMedIso',       't-',   ['med iso']),
        ('LateMedIso',        't+',   ['med iso']),
        ('EarlyTightIso',     't-',   ['tight iso']),
        ('PromptMedIso',      't0',   ['med iso']),
        ('PromptTightIso',    't0',   ['tight iso']),
        ('NonPrompt',         't#pm', []),
    ]

    _SV_DXYSIG_BANDS = [
        ('LowDxySig',  'low d_{xy}/#sigma'),
        ('MidDxySig',  'mid d_{xy}/#sigma'),
        ('HighDxySig', 'high d_{xy}/#sigma'),
    ]

    @staticmethod
    def _region_tag(flag):
        """SR/CR/VR tag for a flag, or '' for inclusive selections."""
        if re.search(r'Val(SR|CR)', flag):
            return 'VR'
        kind = classify_selection_flag(flag)
        if kind == FLAG_SR:
            return 'SR'
        if kind == FLAG_CR:
            return 'CR'
        return ''

    @staticmethod
    def _photon_label(flag):
        count = "2" if "NPhoEq2" in flag else ""
        timing, descriptors = '', []
        for keyword, kw_timing, kw_descriptors in FinalStateResolver._PHOTON_REGION_MAP:
            if keyword in flag:
                timing, descriptors = kw_timing, kw_descriptors
                break
        tags = [t for t in [FinalStateResolver._region_tag(flag)] + descriptors if t]
        sub = f"_{{{timing}}}" if timing else ""
        sup = f"^{{{', '.join(tags)}}}" if tags else ""
        return f"{count}#gamma{sub}{sup}"

    @staticmethod
    def _sv_label(flag):
        if "NSVEq0" in flag:
            return "0SV"
        if "NHad" in flag and "NLep" not in flag:
            flavor = "_{hh}"
        elif "NLep" in flag and "NHad" not in flag:
            flavor = "_{\\ell\\ell}"
        else:
            flavor = ""
        count_match = re.search(r'N(?:SV|Had|Lep)(?:Ge|Eq)?(\d+)', flag)
        count = "2" if count_match and int(count_match.group(1)) >= 2 else ""

        tags = [FinalStateResolver._region_tag(flag)]
        for keyword, band_label in FinalStateResolver._SV_DXYSIG_BANDS:
            if keyword in flag:
                tags.append(band_label)
                break
        tags = [t for t in tags if t]
        sup = f"^{{{', '.join(tags)}}}" if tags else ""
        return f"{count}SV{flavor}{sup}"

    @staticmethod
    def format_sv_label(final_state: str) -> str:
        """
        Format final state labels for SV- and photon-based selection flags.

        '|' (OR) and '+' (AND) combinations are labelled part by part.

        Examples:
          passNHadGe1SelectionHighDxySigSR     -> Region: SV_{hh}^{SR, high d_{xy}/#sigma}
          passNSVGe1SelectionLowDxySigValCR    -> Region: SV^{VR, low d_{xy}/#sigma}
          passNPhoGe1SelectionEarlyBeamHaloCR  -> Region: #gamma_{t-}^{CR, BH}
          passNPhoEq2SelectionPromptTightIsoSR -> Region: 2#gamma_{t0}^{SR, tight iso}
          passNSVGe1SelectionLowDxySigCR+passNPhoGe1SelectionLateSignal
              -> Region: SV^{CR, low d_{xy}/#sigma} + #gamma_{t+}^{!BH, tight iso}
        """
        or_labels = []
        for or_part in final_state.split('|'):
            and_labels = []
            for flag in (f.strip() for f in or_part.split('+')):
                if "NPho" in flag:
                    and_labels.append(FinalStateResolver._photon_label(flag))
                else:
                    and_labels.append(FinalStateResolver._sv_label(flag))
            or_labels.append(' + '.join(and_labels))
        return "Region: " + ' | '.join(or_labels)


class SelectionManager:
    """
    Manages the baseline event filters and triggers.
    """
    def __init__(self):
        self.flags = [
            "Flag_MetFilters",
            # Add other boolean flags here if they are single branches
        ]
        self.inverted_flags = [
            "Flag_hemVeto"
        ]
        # Baseline trigger requirement: OR of the MET triggers
        self.hlt_triggers = [
            "Trigger_PFMET120_PFMHT120_IDTight",
            "Trigger_PFMETNoMu120_PFMHTNoMu120_IDTight",
            "Trigger_PFMET120_PFMHT120_IDTight_PFHT60",
            "Trigger_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60",
        ]
