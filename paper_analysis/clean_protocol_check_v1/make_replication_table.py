"""Builds Supplementary Table S9 (replication by fold) directly from the UConn per-example CSVs and
the per-run JSONs (LD, 2026-10-26). Usage: python -I make_replication_table.py <cleanrun dir> <out.tex>"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(sys.argv[1]); TR = R / "expected/test_read"
CH = ["bond_length_ratio", "bond_angle_ratio", "dihedral_ratio", "planar_improper_ratio"]
load = lambda st: pd.read_csv(TR / f"{st}.per_example.csv").set_index("key")
fr = {(a, op, f): load(f"test_{a}_{op}_f{f}") for a in ("ISO", "HARM", "LAP") for op in ("primary", "secondary") for f in (1, 2, 3)}
ok = None
for (a, op, f), d in fr.items():
    if a == "LAP": continue
    m = np.isfinite(d[CH]).all(axis=1); ok = m if ok is None else ok & m
keys = ok[ok].index; assert len(keys) == 1406
S = lambda d: d.loc[keys, CH].max(axis=1) - d.loc[keys, CH].min(axis=1)
lines = []
for op, lab in (("primary", "fit"), ("secondary", "dev")):
    for a in ("ISO", "LAP", "HARM"):
        cells = []
        for f in (1, 2, 3):
            d = fr[(a, op, f)]; s_used = json.loads((TR / f"test_{a}_{op}_f{f}.json").read_text())["internal_scale"]
            cells.append(f"{s_used:.2f} & {d.loc[keys, 'dihedral_ratio'].median():.2f} & {S(d).median():.3f}")
        diff = " & ".join(f"${(S(fr[(a, op, f)]) - S(fr[('ISO', op, f)])).mean():+.3f}$" if a != "ISO" else "---" for f in (1, 2, 3))
        lines.append(f"MEHF-{a}, {lab} & " + " & ".join(cells) + f" & {diff} \\\\")
    if op == "primary": lines.append("\\midrule")
head = r"""\begin{tabular}{@{}lccccccccccccc@{}}
\toprule
 & \multicolumn{3}{c}{Fold 1} & \multicolumn{3}{c}{Fold 2} & \multicolumn{3}{c}{Fold 3} & \multicolumn{3}{c}{Mean diff.\ vs ISO} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(l){11-13}
Arm & $s^{\ast}$ & Tors. & $S$ & $s^{\ast}$ & Tors. & $S$ & $s^{\ast}$ & Tors. & $S$ & 1 & 2 & 3 \\
\midrule
"""
Path(sys.argv[2]).write_text(head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
print("\n".join(lines))
