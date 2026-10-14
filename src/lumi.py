"""
Per-run luminosity normalization.

Each MC file is scaled by the luminosity of the run (Run 2 / Run 3) it was
generated for, so a stack entry made of Run 2 + Run 3 samples of one process
is  sum_run( evtFillWgt * L_run ).  The luminosities follow from the selected
data years, as in LLPCombineJustin/macro/llpcombine.py (--years).
"""

import re
from collections import OrderedDict
from pathlib import Path

# Integrated luminosity (fb^-1) per data year.
# Keep in sync with LUMI_FB in LLPCombineJustin/macro/llpcombine.py.
LUMI_FB = OrderedDict([("16", 36.32), ("17", 42.01), ("18", 59.50),
                       ("22", 34.12), ("23", 28.27), ("24", 109.82), ("25", 113.67)])
ERA_GROUPS = OrderedDict([("run2", ["16", "17", "18"]),
                          ("run3", ["22", "23", "24", "25"]),
                          ("all", list(LUMI_FB))])
RUN_ENERGY = OrderedDict([("run2", "13 TeV"), ("run3", "13.6 TeV")])

# Year tokens in file names: hadd'd names (QCD_R18, JetMET_R25_...) and data eras (Run2024C)
_YEAR_PATTERNS = (re.compile(r'(?<![A-Za-z0-9])R(\d\d)(?!\d)'), re.compile(r'Run20(\d\d)'))
# MC campaign tokens: RunIISummer20UL18, Run3Summer23, Rn3Sm23
_CAMPAIGN_PATTERNS = (('run2', re.compile(r'RunII(?!I)')), ('run3', re.compile(r'Run3|Rn3')))


class LumiError(ValueError):
    pass


def run_of_year(year):
    for run in RUN_ENERGY:
        if year in ERA_GROUPS[run]:
            return run
    return None


def parse_years(spec):
    """'run2', 'run3', 'all', '16', '2016', 'MET16', a comma string or a list of them -> years in LUMI_FB order."""
    tokens = spec if isinstance(spec, (list, tuple)) else str(spec).split(',')
    picked = set()
    for tok in (str(t).strip() for t in tokens):
        if not tok:
            continue
        if tok.lower() in ERA_GROUPS:
            picked.update(ERA_GROUPS[tok.lower()])
            continue
        m = re.fullmatch(r'(?:MET|20)?(\d\d)', tok, flags=re.IGNORECASE)
        if not m or m.group(1) not in LUMI_FB:
            raise LumiError(f"years: unknown year '{tok}' "
                            f"(known: {', '.join(LUMI_FB)}, or {', '.join(ERA_GROUPS)})")
        picked.add(m.group(1))
    if not picked:
        raise LumiError("years: no year given")
    return [y for y in LUMI_FB if y in picked]


def detect_years(file_path):
    """Data years named in a file name (basename only), in LUMI_FB order."""
    name = Path(file_path).name
    found = {y for pat in _YEAR_PATTERNS for y in pat.findall(name)}
    return [y for y in LUMI_FB if y in found]


def detect_era(file_path):
    """'run2' / 'run3' from a file name, None if it names neither. Raises if it names both."""
    name = Path(file_path).name
    runs = {run_of_year(y) for y in detect_years(file_path)}
    runs.update(run for run, pat in _CAMPAIGN_PATTERNS if pat.search(name))
    if len(runs) > 1:
        raise LumiError(f"{name}: file name matches both Run 2 and Run 3; set 'era:' for it in the config")
    return runs.pop() if runs else None


def _file_era(file_path, groups):
    """Era of an MC file: the config's 'era:' if any group sets one, else from the file name."""
    explicit = {g['eras'][file_path] for g in groups if file_path in g.get('eras', {})}
    if len(explicit) > 1:
        raise LumiError(f"{file_path}: conflicting 'era:' values {sorted(explicit)}")
    if explicit:
        era = explicit.pop().lower()
        if era not in ERA_GROUPS:
            raise LumiError(f"{file_path}: unknown era '{era}' (known: {', '.join(ERA_GROUPS)})")
        return era
    return detect_era(file_path)


def resolve_run_lumis(years_spec, signal_groups, bg_groups, data_groups):
    """
    Work out the per-file MC luminosities.

    years_spec : the config's `years` (None -> the years of the data files, else
                 all years if the background files span both runs).
    Returns None when no years can be established (single global luminosity), else a dict:
      years, run_lumis {run: fb^-1}, total, energy_label, file_lumis {path: fb^-1},
      file_eras {path: era}.
    """
    bg_files = [f for g in bg_groups for f in g['files']]
    sig_files = [f for g in signal_groups for f in g['files']]
    data_files = [f for g in data_groups for f in g['files']]

    data_years = set()
    undetected = []
    for f in data_files:
        ys = detect_years(f)
        data_years.update(ys)
        if not ys:
            undetected.append(f)

    bg_eras = {f: _file_era(f, bg_groups) for f in bg_files}

    if years_spec is not None:
        years = parse_years(years_spec)
        extra = sorted(data_years - set(years))
        missing = [y for y in years if data_files and not undetected and y not in data_years]
        if extra:
            print(f"Warning: data files of 20{', 20'.join(extra)} are loaded but not in years: "
                  f"the MC is not scaled to them")
        if missing:
            print(f"Warning: years includes 20{', 20'.join(missing)} but no data file of "
                  f"{'that year is' if len(missing) == 1 else 'those years are'} loaded")
    elif data_files:
        if undetected:
            raise LumiError("cannot tell the year of data file(s):\n  " + "\n  ".join(undetected) +
                            "\nSet 'years:' (per-run luminosities) or 'lumi:' (one global luminosity).")
        years = [y for y in LUMI_FB if y in data_years]
    elif {'run2', 'run3'} <= set(bg_eras.values()):
        years = list(LUMI_FB)
    else:
        return None

    run_lumis = OrderedDict((run, sum(LUMI_FB[y] for y in years if y in ERA_GROUPS[run]))
                            for run in RUN_ENERGY)
    total = sum(run_lumis.values())
    era_lumi = dict(run_lumis, all=total)

    no_era = [f for f, era in bg_eras.items() if era is None]
    if no_era:
        raise LumiError("cannot tell whether these background files are Run 2 or Run 3:\n  " +
                        "\n  ".join(no_era) +
                        "\nGive them an era in the config:  - {path: <file>, era: run2|run3|all}")

    # Signal is a single campaign standing in for the whole dataset unless told otherwise
    file_eras = {f: _file_era(f, signal_groups) or 'all' for f in sig_files}
    file_eras.update(bg_eras)
    file_lumis = {f: era_lumi[era] for f, era in file_eras.items()}

    return {
        'years': years,
        'run_lumis': run_lumis,
        'total': total,
        'energy_label': " + ".join(RUN_ENERGY[r] for r in RUN_ENERGY if run_lumis[r] > 0),
        'file_lumis': file_lumis,
        'file_eras': file_eras,
    }


def print_lumi_table(info):
    print("\nPer-run luminosity normalization")
    print(f"  Years: {', '.join('20' + y for y in info['years'])}")
    for run, lumi in info['run_lumis'].items():
        print(f"  {run}: {lumi:.2f} fb^-1")
    print(f"  total: {info['total']:.2f} fb^-1 ({info['energy_label']})")
    width = max((len(Path(f).name) for f in info['file_lumis']), default=0)
    for f, lumi in info['file_lumis'].items():
        note = "   <-- contributes nothing" if lumi == 0 else ""
        print(f"    {Path(f).name:<{width}}  {info['file_eras'][f]:<5} {lumi:8.2f}{note}")
    print()
