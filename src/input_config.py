"""
YAML input configuration support.

Allows specifying signal/background/data files and run parameters in a
reusable config file instead of (or alongside) CLI arguments.

Schema example
--------------
lumi: 138.0
energy: 13.0
flags: [passNHad1SelectionSRTight]
plots: [1d, 2d, ratio]
format: pdf
output: run2_nhad1

signal:
  - /eos/.../SMS_*_ct0p1_rjrskim.root

background:
  - name: QCD multijets
    files:
      - /eos/.../QCD_R18_*
  - name: EWK combined
    combine: true
    files:
      - /eos/.../WJets_R18_*
      - /eos/.../ZJets_R18_*

data:
  - name: Run 2
    combine: true
    files:
      - /eos/.../MET_R16_*
      - /eos/.../MET_R17_*
      - /eos/.../MET_R18_*
  - name: Run 3
    combine: true
    files:
      - /eos/.../JetMET_R22_*
      - /eos/.../JetMET_R23_*
"""

import glob as _glob
import os as _os

try:
    import yaml
except ImportError:
    raise ImportError(
        "PyYAML is required for --input-config support. Install it with: pip install pyyaml"
    )


# Parser defaults — used to detect whether a CLI arg is still at its default
# so YAML values can be applied without overriding explicit CLI choices.
_PARSER_DEFAULTS = {
    'lumi': 400.0,
    'energy': 13.6,
    'format': 'root',
    'output': 'standard_plots.root',
    'tree': 'kuSkimTree',
    'flags': ['passNHad1SelectionSRTight', 'passNLep1SelectionSRTight'],
    'plots': ['all'],
}


def _expand_globs(patterns, base_dir=None):
    """Expand a list of path strings / glob patterns to sorted concrete paths."""
    files = []
    for pattern in patterns:
        if base_dir and not _os.path.isabs(pattern):
            pattern = _os.path.join(base_dir, pattern)
        matches = sorted(_glob.glob(str(pattern)))
        if not matches:
            print(f"Warning: no files matched pattern '{pattern}'")
        files.extend(matches)
    return files


def _parse_groups(entries, base_dir=None):
    """
    Parse a YAML list of file entries into group dicts:
        {name: str|None, files: [str], combine: bool}

    Each entry may be:
      - a plain string  → treated as a single path/glob, no name, no combine
      - a dict          → may have 'name', 'files' (str or list), 'combine'
    """
    groups = []
    for entry in entries:
        if isinstance(entry, str):
            groups.append({
                'name': None,
                'files': _expand_globs([entry], base_dir),
                'combine': False,
            })
        elif isinstance(entry, dict):
            raw = entry.get('files', [])
            if isinstance(raw, str):
                raw = [raw]
            groups.append({
                'name': entry.get('name'),
                'files': _expand_globs(raw, base_dir),
                'combine': entry.get('combine', False),
            })
    return groups


def load_input_config(yaml_path):
    """
    Parse a YAML input config file.

    Returns a dict with:
      signal_files  -- flat list of expanded signal file paths
      bg_groups     -- list of {name, files, combine} dicts
      data_groups   -- list of {name, files, combine} dicts
      overrides     -- dict of optional CLI-level overrides (lumi, energy, …)
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    base_dir = cfg.get('base_dir', None)

    # Signal: flat list of paths/globs (no grouping for signals)
    raw_signal = cfg.get('signal', [])
    if isinstance(raw_signal, str):
        raw_signal = [raw_signal]
    signal_files = _expand_globs(raw_signal, base_dir)

    bg_groups = _parse_groups(cfg.get('background', []), base_dir)
    data_groups = _parse_groups(cfg.get('data', []), base_dir)

    override_keys = ('lumi', 'energy', 'flags', 'plots', 'format', 'output', 'tree')
    overrides = {k: cfg[k] for k in override_keys if k in cfg}

    return {
        'signal_files': signal_files,
        'bg_groups': bg_groups,
        'data_groups': data_groups,
        'overrides': overrides,
    }


def apply_config_to_args(args, overrides):
    """
    Apply YAML overrides to an argparse Namespace.
    A YAML value is only applied when the corresponding CLI arg is still
    at its parser default (i.e. the user did not explicitly set it).
    """
    for key, yaml_val in overrides.items():
        attr = key.replace('-', '_')
        current = getattr(args, attr, None)
        if current == _PARSER_DEFAULTS.get(key) or current is None:
            setattr(args, attr, yaml_val)


def unique_files_from_groups(groups):
    """Return a deduplicated, order-preserving flat file list from a group list."""
    seen = set()
    files = []
    for g in groups:
        for f in g['files']:
            if f not in seen:
                seen.add(f)
                files.append(f)
    return files


def assemble_grouped_map(flat_map, groups, combine_fn):
    """
    Build a grouped data map from a flat {region: {filepath: data}} map.

    Non-combined groups: entries are kept as-is, keyed by filepath.
    Combined groups:     all files in the group are merged via combine_fn
                         and stored under the group's 'name'.

    Parameters
    ----------
    flat_map   : {region_key: {filepath: data_dict}}
    groups     : list of {name, files, combine} dicts
    combine_fn : callable(dict) -> data_dict  (e.g. loader.combine_data)
    """
    result = {}
    for region, file_data in flat_map.items():
        result[region] = {}
        for group in groups:
            if group['combine']:
                subset = {f: file_data[f] for f in group['files'] if f in file_data}
                if subset:
                    merged = combine_fn(subset)
                    if merged:
                        result[region][group['name']] = merged
            else:
                for f in group['files']:
                    if f in file_data:
                        result[region][f] = file_data[f]
    return result
