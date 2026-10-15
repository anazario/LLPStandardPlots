"""
Diagnostics of the MC scale-factor likelihood of fit_sfs.py (driver: diagnose_sf_fit.py).

  inputs    config resolution, input audit / cutflow scan, DiagnosticLoader (DataLoader
            subclass adding event-level records), histogram sets (native + common selection)
  stats     composition, MC statistics, shape similarity, Asimov information, fits through
            fit_sfs.build_model / fit_sfs.fit, era tests, stability, toys, validation
  canvases  every diagnostic plot as a TCanvas in one ROOT file
  report    JSON / CSV / HTML summary and the per-case assessments
"""
