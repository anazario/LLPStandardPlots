"""
Inputs of the SF-fit diagnostics: config resolution, input audit, and histogram sets.

Everything selection-related goes through the production code:
  src.input_config.load_input_config   YAML, globs, background groups
  src.lumi.resolve_run_lumis           per-run MC luminosity, file eras
  src.loader.DataLoader                baseline cuts, triggers, custom cuts, weights, N-1 plane cuts
  src.yields.build_yields              per-process x era histograms ("native" selection)

DiagnosticLoader only *adds* to what DataLoader extracts: for every event that the
loader keeps in a region, one event-level record (weight, MET, the mode's RJR
variables, object counts, an in-pass event key). The records give the common-selection
histograms, region overlaps, photon-veto accounting and the event-level partition,
without a second read of the inputs.
"""

import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import yaml

from src.config import AnalysisConfig, AnalysisMode, ModeConfig
from src.input_config import load_input_config, unique_files_from_groups
from src.loader import DataLoader
from src.lumi import resolve_run_lumis, detect_years, LumiError
from src.selections import SelectionManager, is_blinded_selection, _strip_outer_parens
from src.yields import build_yields, data_era, process_label, _fill

VETO_TERM = "nBaseLinePhotons == 0"

# Candidate fit observables per analysis (branch names of src/config.py)
CANDIDATES = {
    AnalysisMode.UNCOMPRESSED: ['selCMet', 'rjr_Ms', 'rjr_Rs'],
    AnalysisMode.COMPRESSED: ['selCMet', 'rjrIsr_RIsr', 'rjrIsr_PtIsr'],
}
REGION_KINDS = ('photon', 'hadronic', 'leptonic')

# Event-identifier branch triplets tried in order (run, lumi, event)
DEFAULT_ID_BRANCHES = [('run', 'lumi', 'event'), ('evtRun', 'evtLumi', 'evtNum'),
                       ('Evt_run', 'Evt_luminosityBlock', 'Evt_event'),
                       ('runNumber', 'lumiBlock', 'eventNumber')]

REC = 'evt__'   # prefix of the record keys added to the loader's per-file data
# Version of the cached analysis data (records, partition rule, histograms); bump on any change
# so that caches written by an older diagnostic are not reused.
RECORD_FORMAT = 3


class DiagInputError(Exception):
    pass


# --------------------------------------------------------------------------- #
# Config resolution                                                            #
# --------------------------------------------------------------------------- #
def _cut_terms(cut):
    """Top-level AND terms of a cut string (the parenthesis-aware splitters of the production code)."""
    s = _strip_outer_parens(cut)
    norm = s.replace('&&', ' & ')
    norm = re.sub(r'(?<!&)&(?!&)', ' & ', norm)
    return [t.strip() for t in DataLoader._split_respecting_parens(None, norm, ' & ') if t.strip()]


def _is_veto_term(term):
    return re.sub(r'[\s()]', '', term) == re.sub(r'\s', '', VETO_TERM)


def veto_variants(cut):
    """(cut without the photon veto, cut with it, whether the original had it)."""
    terms = _cut_terms(cut)
    had = any(_is_veto_term(t) for t in terms)
    base = [t for t in terms if not _is_veto_term(t)]
    fmt = lambda ts: " & ".join(t if t.startswith('(') and t.endswith(')') else f"({t})" for t in ts)
    return fmt(base), fmt(base + [VETO_TERM]), had


def classify_region(cut, region_type):
    if cut in (DataLoader._PROMPT_PHO_CR_COMPRESSED_GE1, DataLoader._PROMPT_PHO_CR_UNCOMPRESSED_GE1) \
            or region_type == 'pho':
        return 'photon'
    if 'SV_nHadronic' in cut and 'SV_nLeptonic' not in cut:
        return 'hadronic'
    if 'SV_nLeptonic' in cut and 'SV_nHadronic' not in cut:
        return 'leptonic'
    return None


def resolve_config(path):
    """
    Everything the diagnostics need from one main.py YAML, resolved with the production helpers.
    """
    path = str(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    conf = load_input_config(path)
    ov = conf['overrides']
    mode = ov.get('analysis_type', 'uncompressed')
    mode_cfg = ModeConfig.get(mode)
    isr_pt_cut = None
    if mode == AnalysisMode.COMPRESSED:
        isr_pt_cut = float(ov['isr_pt_cut']) if ov.get('isr_pt_cut') is not None \
            else float(mode_cfg.get('isr_pt_cut_default', 700.0))

    bg_groups, data_groups = conf['bg_groups'], conf['data_groups']
    lumi_info = None
    if ov.get('lumi') is None:
        try:
            lumi_info = resolve_run_lumis(ov.get('years'), [], bg_groups, data_groups)
        except LumiError as e:
            raise DiagInputError(f"{path}: {e}")
    lumi_total = float(ov['lumi']) if ov.get('lumi') is not None else (
        lumi_info['total'] if lumi_info else 400.0)

    flags = ov.get('flags') or []
    rtypes = ov.get('region_types') or [None] * len(flags)
    blinds = ov.get('blind_cuts') or [False] * len(flags)
    prod = OrderedDict()
    for i, (cut, rt, bl) in enumerate(zip(flags, rtypes, blinds)):
        kind = classify_region(cut, rt)
        if kind is None or kind in prod:
            continue
        prod[kind] = {'cut': cut, 'region_type': rt, 'production_name': f"CustomRegion{i + 1}",
                      'blind': bool(bl) or is_blinded_selection(cut)}
    missing = [k for k in REGION_KINDS if k not in prod]
    if missing:
        raise DiagInputError(f"{path}: could not identify the {', '.join(missing)} region(s) among the flags")

    regions = OrderedDict()
    regions['photon'] = dict(prod['photon'], kind='photon', veto=None, cut_as_configured=prod['photon']['cut'],
                             production=True)
    for kind in ('hadronic', 'leptonic'):
        p = prod[kind]
        no_veto, with_veto, had = veto_variants(p['cut'])
        # the variant equal to the configured selection is passed to the loader verbatim
        regions[kind] = dict(p, kind=kind, veto=True, cut=p['cut'] if had else with_veto,
                             cut_as_configured=p['cut'], config_has_veto=had, production=had)
        regions[f"{kind}_noveto"] = dict(p, kind=kind, veto=False, cut=no_veto if had else p['cut'],
                                         cut_as_configured=p['cut'], config_has_veto=had, production=not had)

    file_eras = lumi_info['file_eras'] if lumi_info else {}
    file_lumis = lumi_info['file_lumis'] if lumi_info else {}
    files = OrderedDict()
    for g in bg_groups:
        for f in g['files']:
            files.setdefault(f, {'is_data': False, 'process': process_label(g, f), 'group': g.get('name'),
                                 'scale': float(g.get('scale', 1.0)), 'era': file_eras.get(f, 'all'),
                                 'lumi': float(file_lumis.get(f, lumi_total)), 'campaign': campaign_of(f),
                                 'years': detect_years(f)})
    for g in data_groups:
        for f in g['files']:
            files.setdefault(f, {'is_data': True, 'process': 'data', 'group': g.get('name'), 'scale': 1.0,
                                 'era': data_era(f), 'lumi': None, 'campaign': None,
                                 'years': detect_years(f)})

    return {
        'config': path,
        'name': mode,
        'mode': mode,
        'isr_pt_cut': isr_pt_cut,
        'tree': ov.get('tree', 'kuSkimTree'),
        'energy': float(ov.get('energy', 13.6)),
        'lumi_total': lumi_total,
        'run_lumis': dict(lumi_info['run_lumis']) if lumi_info else None,
        'energy_label': lumi_info['energy_label'] if lumi_info else None,
        'years': lumi_info['years'] if lumi_info else None,
        'bg_groups': bg_groups,
        'data_groups': data_groups,
        'bg_files': unique_files_from_groups(bg_groups),
        'data_files': unique_files_from_groups(data_groups),
        'files': files,
        'file_eras': file_eras,
        'regions': regions,
        'sf_fit': raw.get('sf_fit') or {},
        'raw_patterns': {'background': raw.get('background'), 'data': raw.get('data'),
                         'base_dir': raw.get('base_dir')},
        'candidates': CANDIDATES[mode],
    }


_CAMPAIGNS = [('RunIISummer20UL16APV', r'RunIISummer20UL16APV'), ('RunIISummer20UL16', r'RunIISummer20UL16(?!APV)'),
              ('RunIISummer20UL17', r'RunIISummer20UL17'), ('RunIISummer20UL18', r'RunIISummer20UL18'),
              ('Run3Summer23BPix', r'Run3Summer23BPix'), ('Run3Summer23', r'Run3Summer23(?!BPix)|Rn3Sm23'),
              ('Run3Summer22EE', r'Run3Summer22EE'), ('Run3Summer22', r'Run3Summer22(?!EE)')]


def campaign_of(path):
    name = Path(path).name
    for label, pat in _CAMPAIGNS:
        if re.search(pat, name):
            return label
    return None


def data_run_tags(path):
    """Data-taking era tokens in a file name, e.g. ['2023C'], for the duplicate-coverage audit."""
    name = Path(path).name
    tags = re.findall(r'Run(20\d\d[A-Z]?)', name)
    if not tags:
        tags = [f"20{y}" for y in re.findall(r'(?<![A-Za-z0-9])R(\d\d)(?!\d)', name)]
    return sorted(set(tags))


# --------------------------------------------------------------------------- #
# Input audit: light pass over scalar branches                                #
# --------------------------------------------------------------------------- #
def _cut_branches(cut):
    """Branches a region cut reads (named cuts: the loader's own requirement list)."""
    loader = DataLoader.__new__(DataLoader)
    if loader._is_named_custom_cut(cut):
        return sorted(loader._missing_named_cut_branches(cut, set()))
    return sorted(DataLoader._extract_cut_tokens(cut))


def _sel_manager():
    return SelectionManager()


def scan_file(path, tree_name, is_data, region_cuts, id_branches):
    """
    One light pass over a file: entries, weights, branch validation and the
    pre-region cutflow of both analysis modes. Returns a plain dict (picklable).

    MC sums are of the raw evtFillWgt; the caller multiplies by the luminosity x group
    scale of the config at hand.

    Steps 0-6 follow DataLoader._load_one_file / _extract_values: MET and weight cuts, the
    baseline flags and the trigger OR restricted to the branches present in the file, then
    the mode validation. The order of the steps is for display only (the loader applies them
    together). Steps "7 ... approx." are object-count proxies of the regions, NOT the region
    selections; the exact region yields come from the loader records (cutflow steps 8.x).
    """
    import uproot
    try:
        import awkward as ak
    except ImportError:  # pragma: no cover
        ak = None
    sm = _sel_manager()
    out = {'path': path, 'is_data': is_data, 'error': None, 'entries': 0, 'missing': {},
           'id_branches': None, 'weights': {}, 'cutflow': {}}
    try:
        f = uproot.open(path)
    except Exception as e:
        out['error'] = f"cannot open: {e}"
        return out
    with f:
        if tree_name not in f:
            out['error'] = f"tree {tree_name} not found"
            return out
        t = f[tree_name]
        keys = set(t.keys())
        out['entries'] = int(t.num_entries)
        req = {
            'weight': ['evtFillWgt'], 'met': ['selCMet'],
            'baseline_flags': sm.flags + sm.inverted_flags,
            'triggers': list(sm.hlt_triggers),
            'compressed': ModeConfig.COMPRESSED['branches'],
            'uncompressed': ModeConfig.UNCOMPRESSED['branches'],
            'photon_veto': ['nBaseLinePhotons'],
        }
        for name, cut in region_cuts.items():
            req[f"region:{name}"] = _cut_branches(cut)
        out['missing'] = {k: sorted(set(v) - keys) for k, v in req.items()}
        out['triggers_available'] = [b for b in sm.hlt_triggers if b in keys]
        for trip in id_branches:
            if all(b in keys for b in trip):
                out['id_branches'] = list(trip)
                break

        scalars = ['evtFillWgt', 'selCMet', 'SV_nHadronic', 'SV_nLeptonic', 'nBaseLinePhotons',
                   'rjrIsr_nSVisObjects'] + sm.flags + sm.inverted_flags + out['triggers_available']
        jagged = ['rjrPTS', 'rjr_Ms', 'rjr_Rs']
        read = [b for b in scalars + jagged if b in keys]
        if 'evtFillWgt' not in keys or 'selCMet' not in keys:
            out['error'] = "evtFillWgt/selCMet missing"
            return out

        W = {'sumw': 0.0, 'sumw2': 0.0, 'n_neg': 0, 'sumw_neg': 0.0, 'wmax': -np.inf, 'wmin': np.inf,
             'n_wcut': 0, 'sumw_wcut': 0.0}
        flows = {}

        def add(step, m, w):
            s = flows.setdefault(step, [0, 0.0, 0.0])
            s[0] += int(m.sum())
            s[1] += float(w[m].sum())
            s[2] += float((w[m] ** 2).sum())

        for arr in t.iterate(read, step_size=500_000, library='ak' if ak else 'np'):
            def col(b):
                a = arr[b]
                return ak.to_numpy(a) if ak else np.asarray(a)
            raw_w = col('evtFillWgt').astype(np.float64)
            n = len(raw_w)
            w = np.ones(n) if is_data else raw_w
            if not is_data:
                W['sumw'] += float(raw_w.sum())
                W['sumw2'] += float((raw_w ** 2).sum())
                neg = raw_w < 0
                W['n_neg'] += int(neg.sum())
                W['sumw_neg'] += float(raw_w[neg].sum())
                if n:
                    W['wmax'] = max(W['wmax'], float(raw_w.max()))
                    W['wmin'] = min(W['wmin'], float(raw_w.min()))
                big = raw_w >= AnalysisConfig.EVT_WGT_CUT
                W['n_wcut'] += int(big.sum())
                W['sumw_wcut'] += float(raw_w[big].sum())
            m = np.ones(n, dtype=bool)
            add('0 all entries', m, w)
            m &= col('selCMet') > AnalysisConfig.MET_CUT
            add(f"1 MET > {AnalysisConfig.MET_CUT:g}", m, w)
            m &= raw_w < AnalysisConfig.EVT_WGT_CUT
            add(f"2 evtFillWgt < {AnalysisConfig.EVT_WGT_CUT:g}", m, w)
            for fl in sm.flags:
                if fl in keys:
                    m &= col(fl) == 1
            add("3 MET filters", m, w)
            for fl in sm.inverted_flags:
                if fl in keys:
                    m &= col(fl) == 0
            add("4 HEM veto", m, w)
            if out['triggers_available']:
                trig = np.zeros(n, dtype=bool)
                for tb in out['triggers_available']:
                    trig |= col(tb) == 1
                m &= trig
            add("5 MET trigger OR", m, w)
            base = m.copy()
            # compressed mode requirement (the loader also needs the ISR plane branches)
            if all(b in keys for b in ModeConfig.COMPRESSED['branches']):
                mc = base & (col('rjrIsr_nSVisObjects') > 0)
            else:
                mc = np.zeros(n, dtype=bool)
            add("6 compressed: N_S^vis > 0", mc, w)
            # uncompressed mode requirement
            if all(b in keys for b in jagged):
                if ak:
                    has = (ak.to_numpy(ak.num(arr['rjr_Ms'])) > 0) & (ak.to_numpy(ak.num(arr['rjr_Rs'])) > 0) \
                          & (ak.to_numpy(ak.num(arr['rjrPTS'])) > 0)
                    pts0 = ak.to_numpy(ak.fill_none(ak.firsts(arr['rjrPTS']), np.inf))
                else:
                    has = np.array([len(a) > 0 and len(b) > 0 and len(c) > 0 for a, b, c in
                                    zip(arr['rjr_Ms'], arr['rjr_Rs'], arr['rjrPTS'])])
                    pts0 = np.array([c[0] if len(c) else np.inf for c in arr['rjrPTS']])
                mu = base & has & (pts0 < AnalysisConfig.RJR_PTS_CUT)
            else:
                mu = np.zeros(n, dtype=bool)
            add(f"6 uncompressed: PTS < {AnalysisConfig.RJR_PTS_CUT:g}", mu, w)
            for tag, mm in (('compressed', mc), ('uncompressed', mu)):
                nh = col('SV_nHadronic') if 'SV_nHadronic' in keys else np.zeros(n)
                nl = col('SV_nLeptonic') if 'SV_nLeptonic' in keys else np.zeros(n)
                npho = col('nBaseLinePhotons') if 'nBaseLinePhotons' in keys else np.full(n, -1)
                # object-count proxies only; the exact region selections are the 8.x steps (records)
                add(f"7 {tag}: approx. >=1 hadronic SV", mm & (nh >= 1), w)
                add(f"7 {tag}: approx. >=1 leptonic SV", mm & (nl >= 1), w)
                add(f"7 {tag}: approx. >=1 photon, no SV", mm & (npho >= 1) & (nh == 0) & (nl == 0), w)
        if not is_data and W['wmax'] == -np.inf:
            W['wmax'] = W['wmin'] = 0.0
        out['weights'] = W if not is_data else {}
        out['cutflow'] = {k: v for k, v in sorted(flows.items())}
    return out


def scan_inputs(files, tree_name, region_cuts, id_branches, workers=1):
    """scan_file over {path: is_data}; parallel over files when workers > 1."""
    results = OrderedDict()
    items = list(files.items())
    if workers > 1 and len(items) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(scan_file, p, tree_name, d, region_cuts, id_branches) for p, d in items]
            for (p, _), fut in zip(items, futs):
                results[p] = fut.result()
    else:
        for i, (p, d) in enumerate(items):
            print(f"  audit [{i + 1}/{len(items)}] {Path(p).name}")
            results[p] = scan_file(p, tree_name, d, region_cuts, id_branches)
    return results


# --------------------------------------------------------------------------- #
# DataLoader extension: event-level records                                    #
# --------------------------------------------------------------------------- #
class DiagnosticLoader(DataLoader):
    """
    DataLoader plus one event-level record per stored event (keys prefixed 'evt__').

    The records are taken from the same chunk, mask and luminosity the loader uses, after
    the same mode validation, so record weights match the loader's MET weights exactly
    (checked at run time by check_records). The event key (chunk serial, row in chunk) is
    unique within one pass over a file and identical for every region of that pass.
    """

    def __init__(self, *a, id_branches=None, **kw):
        super().__init__(*a, **kw)
        self.id_branches = [tuple(t) for t in (id_branches or DEFAULT_ID_BRANCHES)]
        self._last_chunk = None
        self._chunk_serial = -1

    def _get_branches_for_mode(self):
        branches = super()._get_branches_for_mode()
        for trip in self.id_branches:
            branches.extend(trip)
        return branches

    def _load_one_file(self, file_path, branches, event_flags, custom_cuts, is_data):
        self._last_chunk, self._chunk_serial = None, -1
        try:
            return super()._load_one_file(file_path, branches, event_flags, custom_cuts, is_data)
        finally:
            self._last_chunk = None

    def _validated(self, data, indices):
        if self.analysis_mode == AnalysisMode.UNCOMPRESSED:
            if not all(k in data for k in ('rjr_Ms', 'rjr_Rs', 'rjrPTS')):
                return np.zeros(len(indices), dtype=bool)
            return np.array([len(data['rjr_Ms'][i]) > 0 and len(data['rjr_Rs'][i]) > 0 and
                             len(data['rjrPTS'][i]) > 0 and data['rjrPTS'][i][0] < AnalysisConfig.RJR_PTS_CUT
                             for i in indices], dtype=bool)
        if not all(k in data for k in ('rjrIsr_PtIsr', 'rjrIsr_RIsr', 'rjrIsr_nSVisObjects')):
            return np.zeros(len(indices), dtype=bool)
        return np.asarray(data['rjrIsr_nSVisObjects'][indices] > 0, dtype=bool)

    def _extract_values(self, data, mask, is_data=False, selected_object_masks=None, luminosity=None):
        out = super()._extract_values(data, mask, is_data, selected_object_masks, luminosity)
        if data is not self._last_chunk:
            self._last_chunk = data
            self._chunk_serial += 1
        lumi = self.luminosity if luminosity is None else luminosity
        idx = np.where(mask)[0]
        idx = idx[self._validated(data, idx)] if len(idx) else idx
        if len(idx) == 0:
            return out
        rec = {
            f'{REC}chunk': np.full(len(idx), self._chunk_serial, dtype=np.int64),
            f'{REC}row': idx.astype(np.int64),
            f'{REC}w': (np.ones(len(idx)) if is_data else
                        np.asarray(data['evtFillWgt'][idx], dtype=np.float64) * lumi),
            f'{REC}met': np.asarray(data['selCMet'][idx], dtype=np.float64),
            f'{REC}nhad': np.asarray(data['SV_nHadronic'][idx], dtype=np.int64) if 'SV_nHadronic' in data
            else np.full(len(idx), -1),
            f'{REC}nlep': np.asarray(data['SV_nLeptonic'][idx], dtype=np.int64) if 'SV_nLeptonic' in data
            else np.full(len(idx), -1),
            f'{REC}npho': np.asarray(data['nBaseLinePhotons'][idx], dtype=np.int64) if 'nBaseLinePhotons' in data
            else np.full(len(idx), -1),
        }
        V = AnalysisConfig.VARIABLES
        if self.analysis_mode == AnalysisMode.UNCOMPRESSED:
            rec[f'{REC}ms'] = np.array([data['rjr_Ms'][i][0] for i in idx], dtype=np.float64) * V['rjr_Ms']['scale']
            rec[f'{REC}rs'] = np.array([data['rjr_Rs'][i][0] for i in idx], dtype=np.float64) * V['rjr_Rs']['scale']
        else:
            rec[f'{REC}risr'] = np.asarray(data['rjrIsr_RIsr'][idx], dtype=np.float64)
            rec[f'{REC}ptisr'] = np.asarray(data['rjrIsr_PtIsr'][idx], dtype=np.float64)
        for trip in self.id_branches:
            if all(b in data for b in trip):
                for tag, b in zip(('run', 'lumi', 'event'), trip):
                    rec[f'{REC}{tag}'] = np.asarray(data[b][idx]).astype(np.int64)
                break
        out = out if out is not None else {}
        out.update(rec)
        return out


def records_of(fdata):
    return {k[len(REC):]: v for k, v in fdata.items() if k.startswith(REC)}


# --------------------------------------------------------------------------- #
# Selections on the records                                                   #
# --------------------------------------------------------------------------- #
def common_mask(rec, mode, isr_pt_cut):
    """Events passing the cuts of *every* candidate variable (the common selection)."""
    if mode == AnalysisMode.UNCOMPRESSED:
        # every cross cut of the loader (Rs plotted with the Ms cut, Ms with the Rs cut)
        m = np.ones(len(rec['w']), dtype=bool)
        for b in ('rjr_Ms', 'rjr_Rs'):
            other, op, thr = AnalysisConfig.VARIABLES[b]['cross_cut']
            vals = rec[RECORD_VALUE[other]]
            m &= vals > thr if op == '>' else vals < thr
        return m
    m = rec['risr'] >= AnalysisConfig.ISR_RISR_CUT
    if isr_pt_cut is not None:
        m &= rec['ptisr'] >= isr_pt_cut
    return m


def native_mask(rec, mode, isr_pt_cut, branch):
    """The loader's per-variable selection, reproduced on the records (for cross-checks)."""
    n = len(rec['w'])
    if mode == AnalysisMode.UNCOMPRESSED:
        cc = AnalysisConfig.VARIABLES.get(branch, {}).get('cross_cut')
        if cc:
            other, op, thr = cc
            vals = rec[RECORD_VALUE[other]]
            return vals > thr if op == '>' else vals < thr
        return np.ones(n, dtype=bool)
    pt = np.ones(n, dtype=bool) if isr_pt_cut is None else rec['ptisr'] >= isr_pt_cut
    ri = rec['risr'] >= AnalysisConfig.ISR_RISR_CUT
    if branch == 'rjrIsr_PtIsr':
        return ri
    if branch == 'rjrIsr_RIsr':
        return pt
    return pt & ri


RECORD_VALUE = {'selCMet': 'met', 'rjr_Ms': 'ms', 'rjr_Rs': 'rs', 'rjrIsr_RIsr': 'risr', 'rjrIsr_PtIsr': 'ptisr'}


def _mix64(x):
    """splitmix64 finalizer on uint64 arrays (wrapping arithmetic)."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def has_event_ids(rec):
    return all(k in rec for k in ('run', 'lumi', 'event'))


def event_hash(path, rec, use_ids=True, salt="sfdiag"):
    """
    64-bit event hash: of (run, lumi, event) when this file's records have them (the same
    event gets the same hash in every file and region, so a data event present in two input
    files falls into the same partition), else of the in-pass key (file, chunk, row), which is
    the same for every region of one loader pass over that file but not across files.
    """
    with np.errstate(over='ignore'):
        if use_ids and has_event_ids(rec):
            seed = np.uint64(int.from_bytes(hashlib.blake2b(salt.encode(), digest_size=8).digest(), 'little'))
            x = _mix64(rec['run'].astype(np.uint64) ^ seed)
            x = _mix64(x ^ rec['lumi'].astype(np.uint64))
            return _mix64(x ^ rec['event'].astype(np.uint64))
        seed = np.uint64(int.from_bytes(hashlib.blake2b(f"{salt}|{path}".encode(), digest_size=8).digest(),
                                        'little'))
        x = _mix64(rec['chunk'].astype(np.uint64) ^ seed)
        return _mix64(x ^ rec['row'].astype(np.uint64))


def in_pass_key(rec):
    """Integer event key within one loader pass over a file (identical across its regions)."""
    return rec['chunk'].astype(np.int64) * (1 << 32) + rec['row'].astype(np.int64)


def partition_of(path, rec, use_ids=True, salt="sfdiag"):
    """Reproducible event-level 0/1 partition (lowest bit of the event hash)."""
    return (event_hash(path, rec, use_ids, salt) & np.uint64(1)).astype(np.int8)


# --------------------------------------------------------------------------- #
# Histogram containers                                                         #
# --------------------------------------------------------------------------- #
def h_from_th1(h):
    n = h.GetNbinsX()
    ax = h.GetXaxis()
    edges = np.array([ax.GetBinLowEdge(i) for i in range(1, n + 2)])
    w = np.array([h.GetBinContent(i) for i in range(0, n + 2)])
    w2 = np.array([h.GetBinError(i) ** 2 for i in range(0, n + 2)])
    return {'edges': edges, 'w': w, 'w2': w2}


_TH1_COUNTER = [0]


def th1_from(hd, name, title=""):
    import ROOT
    _TH1_COUNTER[0] += 1
    edges = np.asarray(hd['edges'], dtype=np.float64)
    h = ROOT.TH1D(f"{name}__{_TH1_COUNTER[0]}", title, len(edges) - 1, edges)
    h.SetDirectory(0)
    h.Sumw2()
    for i, (c, v) in enumerate(zip(hd['w'], hd['w2'])):
        h.SetBinContent(i, float(c))
        h.SetBinError(i, float(np.sqrt(max(v, 0.0))))
    h.SetEntries(max(float(np.sum(hd['w'])), 0.0))
    return h


def fill_h(values, weights, conf):
    xmin, xmax = conf['range']
    h = _fill(np.asarray(values, dtype=np.float64), np.asarray(weights, dtype=np.float64),
              conf['bins'], xmin, xmax, "fill_tmp", "")
    out = h_from_th1(h)
    return out


def h_add(a, b):
    return {'edges': a['edges'], 'w': a['w'] + b['w'], 'w2': a['w2'] + b['w2']}


def as_yields(entries, var_meta, processes, eras, regions):
    """
    (meta, hists) in the structure of src.yields.read_yields (TH1D), so fit_sfs.build_model
    can be used unchanged. entries: {(region, var): {'bkg': {(proc, era): H}, 'data': {era: H}}}.
    """
    hists = OrderedDict()
    for (reg, var), e in entries.items():
        if regions is not None and reg not in regions:
            continue
        hists[(reg, var)] = {
            'bkg': OrderedDict(((p, er), th1_from(h, f"bkg_{reg}_{var}_{p}_{er}", p))
                               for (p, er), h in e['bkg'].items()),
            'data': OrderedDict((er, th1_from(h, f"data_{reg}_{var}_{er}", "data"))
                                for er, h in e['data'].items()),
        }
    reg_names = list(OrderedDict.fromkeys(r for r, _ in hists))
    meta = {'processes': list(processes), 'eras': list(eras), 'variables': var_meta,
            'regions': [{'name': r, 'vars': [v for rr, v in hists if rr == r], 'blind': False}
                        for r in reg_names]}
    return meta, hists


# --------------------------------------------------------------------------- #
# Loading one analysis                                                         #
# --------------------------------------------------------------------------- #
def _datamc_vars(mode, region_type, cut):
    import main as _main  # the production list of data/MC variables per region type
    return _main._datamc_vars(mode, region_type, 'custom_cut', cut)


def load_analysis(acfg, audit, workers=1, verbose=False, id_branches=None, data=True):
    """
    One pass of DiagnosticLoader over all files of an analysis, all regions at once.

    Returns the compact analysis data (histogram arrays + event records), dropping
    the loader's per-variable arrays. Region x file combinations whose cut reads a
    branch missing from the file are excluded (never interpreted as passing).
    """
    mode = acfg['mode']
    loader = DiagnosticLoader(acfg['tree'], luminosity=acfg['lumi_total'], analysis_mode=mode,
                              isr_pt_cut=acfg['isr_pt_cut'], n_workers=workers, verbose=verbose,
                              id_branches=id_branches)
    loader.file_luminosities = {f: m['lumi'] for f, m in acfg['files'].items() if not m['is_data']}
    region_names = list(acfg['regions'])
    cuts = [acfg['regions'][r]['cut'] for r in region_names]

    print(f"\n[{acfg['name']}] loading {len(acfg['bg_files'])} MC files, {len(region_names)} regions")
    _, bg_custom = loader.load_data_unified(acfg['bg_files'], [], cuts)
    data_regions = [r for r in region_names if not acfg['regions'][r]['blind']]
    data_custom = {}
    if data and acfg['data_files'] and data_regions:
        print(f"[{acfg['name']}] loading {len(acfg['data_files'])} data files")
        _, data_custom = loader.load_data_unified(
            acfg['data_files'], [], [acfg['regions'][r]['cut'] for r in data_regions], is_data=True)

    # region name -> {file: fdata}
    bg_flat = {r: bg_custom.get(f"CustomRegion{i + 1}", {}) for i, r in enumerate(region_names)}
    data_flat = {r: data_custom.get(f"CustomRegion{i + 1}", {}) for i, r in enumerate(data_regions)}

    excluded = []
    for r in region_names:
        for flat in (bg_flat, data_flat):
            for f in list(flat.get(r, {})):
                miss = (audit.get(f) or {}).get('missing', {}).get(f"region:{r}") or []
                if miss:
                    excluded.append({'region': r, 'file': f, 'missing': miss})
                    del flat[r][f]
    # files with missing branches that passed nothing are still reported
    for f, a in audit.items():
        if f not in acfg['files']:
            continue
        for r in region_names:
            miss = (a.get('missing') or {}).get(f"region:{r}") or []
            if miss and not any(e['file'] == f and e['region'] == r for e in excluded):
                excluded.append({'region': r, 'file': f, 'missing': miss})

    return build_analysis_data(acfg, bg_flat, data_flat, excluded)


def build_analysis_data(acfg, bg_flat, data_flat, excluded=()):
    """
    Histogram sets and records from the loader's per-file data of every region.

      native : build_yields() on the loader output (exactly the production histograms)
      common : candidate variables filled from the records with the common selection
      halves : the common histograms split by the event-level partition (A = 0, B = 1), keyed
               per file by (run, lumi, event) when that file has them, else by the in-pass key
    """
    mode = acfg['mode']
    V = AnalysisConfig.VARIABLES
    cands = acfg['candidates']
    var_meta = OrderedDict()
    regions = []
    for r, rc in acfg['regions'].items():
        vars_ = list(OrderedDict.fromkeys(cands + _datamc_vars(mode, rc.get('region_type') or
                                                              ('pho' if rc['kind'] == 'photon' else 'sv'),
                                                              rc['cut'])))
        regions.append({'name': r, 'label': rc['kind'], 'region_type': rc.get('region_type'),
                        'blind': rc['blind'], 'cut': rc['cut'], 'vars': vars_,
                        'bg_flat': bg_flat.get(r, {}), 'data_flat': data_flat.get(r, {})})
    meta_in = {'analysis_type': mode, 'lumi': acfg['lumi_total'], 'run_lumis': acfg['run_lumis']}
    meta, hists = build_yields(regions, acfg['bg_groups'], acfg['file_eras'] or None, V, meta_in)
    native = OrderedDict()
    for key, e in hists.items():
        native[key] = {'bkg': OrderedDict((k, h_from_th1(h)) for k, h in e['bkg'].items()),
                       'data': OrderedDict((k, h_from_th1(h)) for k, h in e['data'].items())}
    for b in cands:
        c = V[b]
        var_meta[c['name']] = {'branch': b, 'label': c['label'], 'bins': c['bins'], 'range': list(c['range'])}
    for v, c in meta['variables'].items():
        var_meta.setdefault(v, c)

    files = acfg['files']
    records = OrderedDict()
    id_strategy = OrderedDict()   # file -> 'event_id' | 'in_pass' (per file, not global)
    objects = OrderedDict()
    common = OrderedDict()
    halves = {'A': OrderedDict(), 'B': OrderedDict()}
    for reg in regions:
        r = reg['name']
        records[r] = {'mc': OrderedDict(), 'data': OrderedDict()}
        objects[r] = {}
        for kind, flat in (('mc', reg['bg_flat']), ('data', reg['data_flat'])):
            for f, fd in flat.items():
                rec = records_of(fd)
                if not rec:
                    continue
                if kind == 'mc':
                    rec['w'] = rec['w'] * files[f]['scale']
                rec['part'] = partition_of(f, rec)
                id_strategy[f] = 'event_id' if has_event_ids(rec) else 'in_pass'
                records[r][kind][f] = rec
                # object (candidate) multiplicity of the SV / photon variables: entries per event
                # entering that variable (the loader's per-variable N-1 plane cuts applied)
                for vk in ('HadronicSV_mass', 'LeptonicSV_mass', 'baseLinePhoton_WTimeSig'):
                    if vk in fd:
                        o = objects[r].setdefault(vk, {'entries': 0.0, 'events': 0.0})
                        o['entries'] += len(fd[vk])
                        o['events'] += int(native_mask(rec, mode, acfg['isr_pt_cut'], vk).sum())
        # common-selection + partition histograms of the candidates
        for b in cands:
            c = V[b]
            vname = c['name']
            ent = {'bkg': OrderedDict(), 'data': OrderedDict()}
            entA = {'bkg': OrderedDict(), 'data': OrderedDict()}
            entB = {'bkg': OrderedDict(), 'data': OrderedDict()}
            for kind in ('mc', 'data'):
                for f, rec in records[r][kind].items():
                    m = common_mask(rec, mode, acfg['isr_pt_cut'])
                    vals = rec[RECORD_VALUE[b]]
                    if kind == 'mc':
                        key, slot = (files[f]['process'], files[f]['era']), 'bkg'
                    else:
                        key, slot = files[f]['era'], 'data'
                    for e_, mm in ((ent, m), (entA, m & (rec['part'] == 0)), (entB, m & (rec['part'] == 1))):
                        h = fill_h(vals[mm], rec['w'][mm], c)
                        e_[slot][key] = h_add(e_[slot][key], h) if key in e_[slot] else h
            if ent['bkg']:
                common[(r, vname)] = ent
                halves['A'][(r, vname)] = entA
                halves['B'][(r, vname)] = entB
            if reg['blind']:
                for e_ in (ent, entA, entB):
                    e_['data'] = OrderedDict()

    id_summary = id_strategy_summary(records, id_strategy, files)
    procs = meta['processes']
    eras = [e for e in ('run2', 'run3', 'all') if e in meta['eras']] + \
           [e for e in meta['eras'] if e not in ('run2', 'run3', 'all')]
    return {
        'name': acfg['name'], 'mode': mode, 'config': acfg['config'], 'isr_pt_cut': acfg['isr_pt_cut'],
        'regions': {r: {k: v for k, v in rc.items()} for r, rc in acfg['regions'].items()},
        'processes': procs, 'eras': eras, 'run_lumis': acfg['run_lumis'], 'lumi_total': acfg['lumi_total'],
        'energy_label': acfg['energy_label'], 'candidates': [V[b]['name'] for b in cands],
        'native': native, 'common': common, 'halves': halves, 'records': records, 'objects': objects,
        'var_meta': var_meta, 'files': files, 'excluded': list(excluded),
        # event-identity-safe: every selected data file has (run, lumi, event)
        'event_ids': id_summary['data_event_identity_safe'], 'id_strategy': id_strategy, 'id_summary': id_summary,
        'region_vars': {reg['name']: [V[v]['name'] for v in reg['vars'] if v in V] for reg in regions},
    }


def id_strategy_summary(records, id_strategy, files):
    """Files and selected events per event-identifier strategy, for MC and data and per region."""
    out = OrderedDict()
    for kind in ('mc', 'data'):
        fs = [f for f in id_strategy if files[f]['is_data'] == (kind == 'data')]
        row = OrderedDict(files_event_id=sum(id_strategy[f] == 'event_id' for f in fs),
                          files_in_pass=sum(id_strategy[f] == 'in_pass' for f in fs), events_by_region=OrderedDict())
        for r, rr in records.items():
            ev = {'event_id': 0, 'in_pass': 0}
            for f, rec in rr[kind].items():
                ev[id_strategy[f]] += len(rec['w'])
            row['events_by_region'][r] = ev
        out[kind] = row
    out['data_event_identity_safe'] = bool(out['data']['files_event_id'] > 0 and out['data']['files_in_pass'] == 0)
    return out


def check_records(adata):
    """
    Consistency of the records with the loader's own histograms: per region, process and era
    the record weight sum with the native-selection mask must equal the native histogram sum
    of every candidate variable (to rounding). Returns a list of discrepancies.
    """
    mode = adata['mode']
    issues = []
    for r, rr in adata['records'].items():
        for b, vname in ((b, AnalysisConfig.VARIABLES[b]['name']) for b in RECORD_VALUE
                         if AnalysisConfig.VARIABLES[b]['name'] in adata['candidates']):
            entry = adata['native'].get((r, vname))
            if entry is None:
                continue
            sums = {}
            for f, rec in rr['mc'].items():
                key = (adata['files'][f]['process'], adata['files'][f]['era'])
                m = native_mask(rec, mode, adata['isr_pt_cut'], b)
                sums[key] = sums.get(key, 0.0) + rec['w'][m].sum()
            for key, h in entry['bkg'].items():
                ref = h['w'].sum()
                got = sums.get(key, 0.0)
                if abs(ref - got) > 1e-6 * max(1.0, abs(ref)):
                    issues.append({'region': r, 'var': vname, 'process': key[0], 'era': key[1],
                                   'native': float(ref), 'records': float(got)})
            dsum = {}
            for f, rec in rr['data'].items():
                m = native_mask(rec, mode, adata['isr_pt_cut'], b)
                era = adata['files'][f]['era']
                dsum[era] = dsum.get(era, 0.0) + rec['w'][m].sum()
            for era, h in entry['data'].items():
                if abs(h['w'].sum() - dsum.get(era, 0.0)) > 1e-6 * max(1.0, h['w'].sum()):
                    issues.append({'region': r, 'var': vname, 'process': 'data', 'era': era,
                                   'native': float(h['w'].sum()), 'records': float(dsum.get(era, 0.0))})
    return issues


def compare_reference_yields(adata, ref_path, region_map=None):
    """
    Regression against an existing yields file (main.py --yields) of the same config:
    max relative difference per (region, var, process, era). region_map maps the
    diagnostic region names to the yields-file names (default: production CustomRegionN).
    """
    from src.yields import read_yields
    meta, hists = read_yields(ref_path)
    region_map = region_map or {r: rc['production_name'] for r, rc in adata['regions'].items()
                                if rc.get('production')}
    rows = []
    for r, ref_r in region_map.items():
        for (rr, var), e in adata['native'].items():
            if rr != r or (ref_r, var) not in hists:
                continue
            ref = hists[(ref_r, var)]
            for key, h in e['bkg'].items():
                if key not in ref['bkg']:
                    rows.append({'region': r, 'var': var, 'sample': f"{key[0]}:{key[1]}", 'status': 'missing in reference'})
                    continue
                rh = h_from_th1(ref['bkg'][key])
                d = np.max(np.abs(rh['w'] - h['w']) / np.maximum(np.abs(rh['w']), 1e-9))
                rows.append({'region': r, 'var': var, 'sample': f"{key[0]}:{key[1]}", 'max_rel_diff': float(d),
                             'status': 'ok' if d < 1e-6 else 'DIFFERENT'})
            for era, h in e['data'].items():
                if era not in ref['data']:
                    continue
                rh = h_from_th1(ref['data'][era])
                d = np.max(np.abs(rh['w'] - h['w']) / np.maximum(np.abs(rh['w']), 1e-9))
                rows.append({'region': r, 'var': var, 'sample': f"data:{era}", 'max_rel_diff': float(d),
                             'status': 'ok' if d < 1e-6 else 'DIFFERENT'})
    return rows


def config_fingerprint(acfg, id_branches=None):
    """Hash of the resolved inputs (files + cuts + lumis + cut constants + record format) for the cache."""
    payload = {
        'record_format': RECORD_FORMAT, 'id_branches': [list(t) for t in (id_branches or DEFAULT_ID_BRANCHES)],
        'files': sorted(acfg['files']), 'regions': {r: rc['cut'] for r, rc in acfg['regions'].items()},
        'lumis': {f: m['lumi'] for f, m in acfg['files'].items()}, 'mode': acfg['mode'],
        'isr': acfg['isr_pt_cut'], 'consts': [AnalysisConfig.MET_CUT, AnalysisConfig.EVT_WGT_CUT,
                                              AnalysisConfig.RJR_PTS_CUT, AnalysisConfig.ISR_RISR_CUT],
        'mtimes': {f: (Path(f).stat().st_mtime if Path(f).exists() else None) for f in acfg['files']},
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
