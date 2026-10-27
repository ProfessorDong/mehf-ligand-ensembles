"""Re-estimate the four physical widths tau from TRAINING-split systems only.

Same estimator as the originals (leakage_audit.py `tau_contamination`): median over systems
of the per-system mean per-coordinate standard deviation, heavy atoms only; out-of-plane only
for centres whose reference-trajectory mean deviation is < 10 deg (as recorded in
theory_modules_v1).  The per-system values are the ones already measured on the original
random MISATO systems; this script only changes WHICH systems enter the median:
  keep a system iff its PDB entry belongs to a UniProt target in the project's `train`
  split (target_disjoint_split primary_split == 'train').  Systems in no split ("not in the
  corpus") are excluded, as are val/test.
Writes outputs/clean_protocol_v1/tau_train_only.json.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
from mdwm.data.mmp_transfer import DEFAULT_PAIRS, load_usable_pairs, target_disjoint_split  # noqa: E402
from mdwm.models import harmonic_prior as hp  # noqa: E402

pairs = target_disjoint_split(load_usable_pairs(DEFAULT_PAIRS))
pdb_split: dict[str, str] = {}
for split, grp in pairs.groupby("primary_split"):
    for p in set(grp.pdb_a.str.upper()) | set(grp.pdb_b.str.upper()):
        assert pdb_split.setdefault(p, split) == split
label = lambda p: pdb_split.get(p, "not_in_corpus")  # noqa: E731

ps = pd.read_csv("reference_outputs/isotropic_noise_cost_v1/per_system.csv")
ps["split"] = ps.system.astype(str).str.upper().str.replace(".NPZ", "", regex=False).map(label)
wb = pd.read_csv("reference_outputs/wilson_b_theory_v1/wilson_b.csv")
wb["split"] = wb.system.astype(str).str.split("_").str[0].str.upper().map(label)
tm = pd.DataFrame(json.loads(Path("reference_outputs/theory_modules_v1/results.json").read_text())["rows"])
tm["split"] = tm.system.astype(str).str.split("_").str[0].str.upper().map(label)
tm = tm.dropna(subset=["tau_oop_deg"])

current = {"bond_angstrom": hp.TAU_BOND_ANGSTROM, "angle_deg": math.degrees(hp.TAU_ANGLE_RADIAN),
           "torsion_deg": math.degrees(hp.TAU_TORSION_RADIAN), "oop_deg": math.degrees(hp.TAU_OOP_RADIAN)}
out = {"criterion": "systems whose PDB entry has UniProt target in split 'train' of "
                    "target_disjoint_split(load_usable_pairs(DEFAULT_PAIRS)); val, test and "
                    "not-in-corpus systems excluded; median over systems of per-system mean "
                    "per-coordinate std, heavy atoms only; oop restricted to centres with "
                    "reference mean deviation < 10 deg (as recorded in theory_modules_v1)",
       "sources": {"bond": "wilson_b_theory_v1/wilson_b.csv", "angle_torsion": "isotropic_noise_cost_v1/per_system.csv",
                   "oop": "theory_modules_v1/results.json"}, "widths": {}}
for name, frame, col in (("bond_angstrom", wb, "bond_length_spread_physical"),
                         ("angle_deg", ps, "bond_angle_spread_physical"),
                         ("torsion_deg", ps, "dihedral_spread_physical"),
                         ("oop_deg", tm, "tau_oop_deg")):
    tr = frame[frame.split == "train"]
    new = float(tr[col].median())
    out["widths"][name] = {
        "n_systems_total": int(len(frame)), "n_train": int(len(tr)),
        "n_val": int((frame.split == "val").sum()), "n_test": int((frame.split == "test").sum()),
        "n_not_in_corpus": int((frame.split == "not_in_corpus").sum()),
        "current_constant": current[name], "median_all_systems": float(frame[col].median()),
        "train_only": new, "relative_change_vs_current": new / current[name] - 1,
        "bootstrap_se_of_median": float(pd.Series([tr[col].sample(len(tr), replace=True, random_state=i).median()
                                                   for i in range(2000)]).std()),
    }
(ROOT / "outputs/clean_protocol_v1/tau_train_only.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out["widths"], indent=2))
