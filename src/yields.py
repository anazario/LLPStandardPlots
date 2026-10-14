"""
Per-process x era yields file: every data/MC histogram of a run, kept unmerged.

main.py (--yields / YAML `yields:`) fills one TH1D per (region, variable,
process, era) from the per-file maps -- before the files of a stack entry are
combined, before Run 2 and Run 3 are summed, and before the QCD + gamma+jets
merge of the SV regions -- plus the data per era. replot_yields.py and
fit_sfs.py rebuild the plots / fit scale factors from it in seconds.

File layout
-----------
  meta                                  TNamed, JSON: processes, eras, regions, variables, lumi
  <region>/<var>/bkg__<proc>__<era>     TH1D, title = process label
  <region>/<var>/data__<era>            TH1D (absent in blinded regions)

<proc> is the process label with non-alphanumerics replaced by '_'; the label
itself is in the histogram title and in meta['processes'].
"""

import json
import re
from collections import OrderedDict

import numpy as np
import ROOT

from src.lumi import detect_years, run_of_year
from src.utils import parse_background_name

YIELDS_VERSION = 1


def safe_name(label):
    return re.sub(r'[^A-Za-z0-9]+', '_', label).strip('_')


def normalize_label(s):
    """'W + jets' == 'W+jets' == 'w+Jets' for matching user-written process names."""
    return re.sub(r'\s+', '', s).lower()


def process_label(group, file_path):
    """Stack-entry label of a background file: the group name for combined groups, else from the file name."""
    if group.get('combine') and group.get('name'):
        return group['name']
    return parse_background_name(file_path)


def data_era(file_path):
    runs = {run_of_year(y) for y in detect_years(file_path)}
    return runs.pop() if len(runs) == 1 else 'unknown'


def _fill(values, weights, nbins, xmin, xmax, name, title):
    h = ROOT.TH1D(name, title, nbins, xmin, xmax)
    h.SetDirectory(0)
    h.Sumw2()
    n = min(len(values), len(weights))
    if n:
        h.FillN(n, np.asarray(values[:n], dtype=np.float64), np.asarray(weights[:n], dtype=np.float64))
    return h


def _var_arrays(data, branch):
    values = data.get(branch)
    if values is None or len(values) == 0:
        return None, None
    return values, data.get(f'{branch}_weights', data.get('weights', []))


def build_yields(regions, bg_groups, file_eras, variables, meta):
    """
    regions   : list of dicts {name, label, region_type, blind, cut, vars: [var_key],
                               bg_flat: {file: data}, data_flat: {file: data}}
    bg_groups : config background groups (name, files, combine, scale)
    file_eras : {file: 'run2'|'run3'|'all'} for MC (None -> everything is 'all')
    variables : AnalysisConfig.VARIABLES
    meta      : extra metadata (lumi, energy, analysis_type, config, ...)

    Returns (meta, hists) with hists[(region, var_name)] = {'bkg': {(proc, era): TH1D},
                                                             'data': {era: TH1D}}.
    """
    file_eras = file_eras or {}
    file_group = OrderedDict()
    for g in bg_groups:
        for f in g['files']:
            file_group.setdefault(f, g)

    processes, eras = [], []
    hists = OrderedDict()
    region_meta, var_meta = [], OrderedDict()

    for reg in regions:
        var_names = []
        for var_key in reg['vars']:
            conf = variables.get(var_key)
            if not conf:
                continue
            name, nbins = conf['name'], conf['bins']
            xmin, xmax = conf['range']
            var_meta[name] = {'branch': var_key, 'label': conf['label'], 'bins': nbins, 'range': [xmin, xmax]}

            bkg, data = OrderedDict(), OrderedDict()
            for f, fdata in reg['bg_flat'].items():
                g = file_group.get(f)
                if g is None:
                    continue
                values, weights = _var_arrays(fdata, var_key)
                if values is None:
                    continue
                proc, era = process_label(g, f), file_eras.get(f, 'all')
                weights = np.asarray(weights, dtype=np.float64) * g.get('scale', 1.0)
                h = _fill(values, weights, nbins, xmin, xmax,
                          f"bkg__{safe_name(proc)}__{era}", proc)
                if (proc, era) in bkg:
                    bkg[(proc, era)].Add(h)
                else:
                    bkg[(proc, era)] = h
                if proc not in processes:
                    processes.append(proc)
                if era not in eras:
                    eras.append(era)

            if not reg['blind']:
                for f, fdata in reg['data_flat'].items():
                    values, weights = _var_arrays(fdata, var_key)
                    if values is None:
                        continue
                    era = data_era(f)
                    h = _fill(values, weights, nbins, xmin, xmax, f"data__{era}", "data")
                    if era in data:
                        data[era].Add(h)
                    else:
                        data[era] = h

            if bkg:
                hists[(reg['name'], name)] = {'bkg': bkg, 'data': data}
                var_names.append(name)

        if var_names:
            region_meta.append({k: reg.get(k) for k in ('name', 'label', 'region_type', 'blind', 'cut')}
                               | {'vars': var_names})

    meta = dict(meta, version=YIELDS_VERSION, processes=processes, eras=eras,
                regions=region_meta, variables=var_meta)
    return meta, hists


def write_yields(path, meta, hists):
    f = ROOT.TFile(str(path), "RECREATE")
    ROOT.TNamed("meta", json.dumps(meta)).Write()
    for (region, var), entry in hists.items():
        d = f.GetDirectory(region) or f.mkdir(region)
        vd = d.mkdir(var)
        vd.cd()
        for h in list(entry['bkg'].values()) + list(entry['data'].values()):
            h.Write()
        f.cd()
    f.Close()


def read_yields(path):
    """Inverse of write_yields: (meta, hists) with the same structure as build_yields."""
    f = ROOT.TFile.Open(str(path))
    if not f or f.IsZombie():
        raise IOError(f"cannot open yields file {path}")
    meta_obj = f.Get("meta")
    if not meta_obj:
        raise IOError(f"{path} is not a yields file (no 'meta' object)")
    meta = json.loads(meta_obj.GetTitle())
    hists = OrderedDict()
    for reg in meta['regions']:
        for var in reg['vars']:
            d = f.GetDirectory(f"{reg['name']}/{var}")
            entry = {'bkg': OrderedDict(), 'data': OrderedDict()}
            for key in d.GetListOfKeys():
                h = key.ReadObj()
                h.SetDirectory(0)
                parts = key.GetName().split('__')
                if parts[0] == 'bkg':
                    entry['bkg'][(h.GetTitle(), parts[2])] = h
                elif parts[0] == 'data':
                    entry['data'][parts[1]] = h
            hists[(reg['name'], var)] = entry
    f.Close()
    return meta, hists


class ScaleFactors:
    """
    Per-process MC scale factors, optionally per era and per region.

    YAML:
        scale_factors:
          QCD multijets: 1.10                    # every era
          W+jets: {run2: 1.05, run3: 0.95}       # per era (missing eras -> 'all' -> 1)
        regions:                                 # optional per-region overrides
          CustomRegion2:
            QCD multijets: 0.8

    Process names match the stack labels, ignoring case and spaces.
    """

    def __init__(self, global_sfs=None, region_sfs=None):
        self.global_sfs = global_sfs or {}
        self.region_sfs = region_sfs or {}

    @classmethod
    def from_yaml(cls, path):
        import yaml
        with open(path) as fh:
            doc = yaml.safe_load(fh) or {}
        return cls(doc.get('scale_factors', {}), doc.get('regions', {}))

    def validate(self, processes):
        known = {normalize_label(p) for p in processes}
        names = list(self.global_sfs) + [p for r in self.region_sfs.values() for p in r]
        unknown = sorted({n for n in names if normalize_label(n) not in known})
        if unknown:
            raise ValueError(f"unknown process(es) in scale factors: {unknown}; "
                             f"known: {processes}")

    @staticmethod
    def _lookup(table, proc, era):
        for name, val in table.items():
            if normalize_label(name) != normalize_label(proc):
                continue
            if isinstance(val, dict):
                return float(val.get(era, val.get('all', 1.0)))
            return float(val)
        return None

    def get(self, region, proc, era):
        for table in (self.region_sfs.get(region, {}), self.global_sfs):
            sf = self._lookup(table, proc, era)
            if sf is not None:
                return sf
        return 1.0


def mc_by_process(entry, region, sfs=None, eras=None):
    """{process: TH1D} summed over the selected eras with the scale factors applied."""
    out = OrderedDict()
    for (proc, era), h in entry['bkg'].items():
        if eras and era not in eras:
            continue
        hs = h.Clone(f"h_{safe_name(proc)}_{region}")
        hs.SetDirectory(0)
        hs.Scale(sfs.get(region, proc, era) if sfs else 1.0)
        if proc in out:
            out[proc].Add(hs)
        else:
            out[proc] = hs
    return out


def data_total(entry, eras=None):
    h = None
    for era, hd in entry['data'].items():
        if eras and era not in eras:
            continue
        if h is None:
            h = hd.Clone("h_data")
            h.SetDirectory(0)
        else:
            h.Add(hd)
    return h


def default_yields_path(output):
    stem = output[:-5] if output.endswith('.root') else output
    return f"{stem}_yields.root"


def load_scale_factors(path):
    return ScaleFactors.from_yaml(path) if path else ScaleFactors()


def is_qcd(label):
    return parse_background_name(label) == "QCD multijets"


def is_gjets(label):
    return "gamma" in parse_background_name(label)

