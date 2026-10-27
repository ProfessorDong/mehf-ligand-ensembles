"""Like-for-like fitting-fold -> test comparison (post hoc, LD 2026-10-26): per fold, the spread of the four channel
medians at s* on the fitting folds (from the frozen s* JSONs) against the same statistic on the 1,406 test examples."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(sys.argv[1]); TR = R / "expected/test_read"
CH = ["bond_length_ratio", "bond_angle_ratio", "dihedral_ratio", "planar_improper_ratio"]
ok = None
for a in ("ISO", "HARM"):
    for op in ("primary", "secondary"):
        for f in (1, 2, 3):
            d = pd.read_csv(TR / f"test_{a}_{op}_f{f}.per_example.csv").set_index("key"); m = np.isfinite(d[CH]).all(axis=1); ok = m if ok is None else ok & m
keys = ok[ok].index
fz = {"ISO": json.loads((R / "frozen/s_star_frozen_v2.json").read_text()), "HARM": json.loads((R / "frozen/s_star_frozen_v2.json").read_text()),
      "LAP": json.loads((R / "frozen/s_star_frozen_lap.json").read_text())}
out = {}
for a in ("ISO", "LAP", "HARM"):
    blk = fz[a]["PRIMARY_fitting_folds"][a] if a in fz[a]["PRIMARY_fitting_folds"] else fz[a]["PRIMARY_fitting_folds"]
    fit = [blk[f"fold{f}"]["spread_at_s_star"] for f in (1, 2, 3)]
    test = []
    for f in (1, 2, 3):
        med = pd.read_csv(TR / f"test_{a}_primary_f{f}.per_example.csv").set_index("key").loc[keys, CH].median()
        test.append(float(med.max() - med.min()))
    out[a] = {"fit": fit, "test": test, "ratio": [t / s for t, s in zip(test, fit)]}
    print(a, "fit", np.round(fit, 3), "test", np.round(test, 3), "ratio", np.round(out[a]["ratio"], 2))
Path("DEGRADATION.json").write_text(json.dumps(out, indent=1))
