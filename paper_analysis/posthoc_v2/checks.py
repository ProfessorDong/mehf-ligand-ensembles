import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(sys.argv[1]); REF = ROOT/"reference_outputs"; out = {}
# 1. angle term
S = REF/"angle_term_v1/spectrum"
def norm(c):
    d = json.load(open(S/f"{c}.json")); return d["bond_angle"]["ratio_median"]/d["internal_rmsf"]["ratio_median"]
base = np.mean([norm(f"A_f{f}") for f in (1,2,3)])
loss = [100*np.mean([norm(f"B_f{f}")-norm(f"A_f{f}"), norm(f"D_f{f}")-norm(f"C_f{f}")])/base for f in (1,2,3)]
proj = [100*np.mean([norm(f"C_f{f}")-norm(f"A_f{f}"), norm(f"D_f{f}")-norm(f"B_f{f}")])/base for f in (1,2,3)]
out["angle_loss_pct_per_fold"] = loss; out["angle_loss_pct_mean"] = float(np.mean(loss))
out["angle_proj_pct_per_fold"] = proj; out["angle_proj_pct_mean"] = float(np.mean(proj))
# 2. equal-target
SRC = {"ISO":(REF/"sigma_star_test_v1","ISO"),"SIG":(REF/"sigma_star_test_v1","SIGMA_star")}
CH4 = ["bond_length","bond_angle","dihedral","planar_improper"]
def arm(a):
    d,n = SRC[a]; fr=[]
    for f in (1,2,3):
        x = pd.read_csv(d/f"test_{n}_f{f}.per_example.csv").set_index("key")
        r = x[[c+"_ratio" for c in CH4]]; ok = r.notna().all(axis=1); lr = np.log(x[[c+"_ratio" for c in CH4+["internal_rmsf"]]]).abs()
        fr.append(pd.DataFrame({"S": (r.max(axis=1)-r.min(axis=1)).where(ok), "E4": lr[[c+"_ratio" for c in CH4]].mean(axis=1, skipna=False),
             "E5": lr.mean(axis=1, skipna=False), "E3": lr[[c+"_ratio" for c in CH4[:3]]].mean(axis=1, skipna=False)}))
    return sum(fr)/3
ex = pd.read_csv(REF/"sigma_star_test_v1/test_examples.csv").set_index("example_id")["uniprot"]
I, H = arm("ISO"), arm("SIG"); rng_seed = 20260925
for m in ("S","E4","E5","E3"):
    d = (H[m]-I[m]).dropna(); t = ex.loc[d.index]
    per = d.groupby(t).mean(); vals = per.to_numpy()
    rng = np.random.default_rng(rng_seed); bs = [vals[rng.integers(0,len(vals),len(vals))].mean() for _ in range(10000)]
    out[f"equal_target_{m}"] = {"mean": float(vals.mean()), "ci95": [float(np.percentile(bs,2.5)), float(np.percentile(bs,97.5))],
                                "n_targets": int(len(vals)), "targets_lower": int((vals<0).sum())}
k4 = (H["E4"]-I["E4"]).dropna().index; sizes = ex.loc[k4].value_counts()
out["top3_targets"] = {k:int(v) for k,v in sizes.head(3).items()}; out["top3_share"] = float(sizes.head(3).sum()/len(k4))
(Path(sys.argv[2])/"RESULT.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
