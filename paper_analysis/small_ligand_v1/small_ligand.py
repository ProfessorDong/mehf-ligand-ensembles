"""Post-hoc small-ligand E4 analysis. See PLAN.md. Run: python -I small_ligand.py <MEHF root> <out dir>"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

REF = Path(sys.argv[1]) / "reference_outputs"; OUT = Path(sys.argv[2])
CH = ["bond_length", "bond_angle", "dihedral", "planar_improper"]
F = {"ISO": "sigma_star_test_v1/test_ISO_f{}.per_example.csv", "GS": "sigma_star_test_v1/test_ANISO_v3_f{}.per_example.csv",
     "LAP": "laplacian_prior_v1/test_LAPLACIAN_f{}.per_example.csv", "HARM": "sigma_star_test_v1/test_SIGMA_star_f{}.per_example.csv"}
D = {a: [pd.read_csv(REF / p.format(f), index_col="key") for f in (1, 2, 3)] for a, p in F.items()}
keys = sorted(set.intersection(*[set(d.index[d[[c + "_ratio" for c in CH]].notna().all(axis=1)]) for a in D for d in D[a]]))
assert len(keys) == 1410
L = {a: sum(np.abs(np.log(d.loc[keys, [c + "_ratio" for c in CH]]))for d in D[a]) / 3 for a in ("ISO", "HARM")}
dch = (L["HARM"] - L["ISO"])            # per-channel |log| differences
dE4 = dch.mean(axis=1)
ex = pd.read_csv(REF / "sigma_star_test_v1/test_examples.csv", index_col="example_id").loc[keys]
heavy = pd.Series(np.where(ex["product"] == ex["pdb_a"], ex["heavy_a"], ex["heavy_b"]), index=keys)
t = ex["uniprot"]; small = heavy < 20
out = {}
# 1 concentration
sb = pd.DataFrame({"t": t[small], "d": dE4[small]}).groupby("t")["d"].agg(["size", "mean"]).sort_values("size", ascending=False)
out["small_bin_targets"] = sb.reset_index().to_dict(orient="records")
out["small_bin_top2_share"] = float(sb["size"].head(2).sum() / small.sum())
# 2 target level
tg = pd.DataFrame({"t": t, "d": dE4, "h": heavy}).groupby("t").agg(d=("d", "mean"), h=("h", "median"), n=("d", "size"))
rho = spearmanr(tg.h, tg.d).correlation
rng = np.random.default_rng(20260925); bs = []
for _ in range(10000):
    s = tg.iloc[rng.integers(0, len(tg), len(tg))]
    bs.append(spearmanr(s.h, s.d).correlation)
out["target_spearman_rho"] = float(rho); out["target_spearman_ci95"] = [float(x) for x in np.nanpercentile(bs, [2.5, 97.5])]
for name, m in (("small_targets", tg.h < 20), ("large_targets", tg.h >= 20)):
    out[name] = {"n_targets": int(m.sum()), "mean_of_target_mean_dE4": float(tg.d[m].mean()),
                 "targets_dE4_positive": int((tg.d[m] > 0).sum()), "examples": int(tg.n[m].sum())}
# 3 within target
el = [u for u in tg.index if (small & (t == u)).sum() >= 5 and (~small & (t == u)).sum() >= 5]
out["within_target_eligible"] = len(el)
out["within_target"] = {u: float(dE4[small & (t == u)].mean() - dE4[~small & (t == u)].mean()) for u in el}
# 4 channels
out["channel_median_dlog"] = {b: {c: float(dch.loc[m.to_numpy(), c + "_ratio"].median()) for c in CH}
                              for b, m in (("lt20", small), ("ge20", ~small))}
out["channel_median_log_ratio"] = {a: {b: {c: float((sum(np.log(d.loc[keys, c + "_ratio"]) for d in D[a]) / 3)[m.to_numpy()].median()) for c in CH}
                                       for b, m in (("lt20", small), ("ge20", ~small))} for a in ("ISO", "HARM")}
out["planar_centre_count"] = "not stored in the per-example records"
# 5 sensitivity
def tboot(v, g):
    per = pd.Series(v).groupby(g.to_numpy())
    vals, grp = v.to_numpy(), g.to_numpy(); us = np.unique(grp); idx = {u: np.where(grp == u)[0] for u in us}
    r = np.random.default_rng(20260925); bs = []
    for _ in range(10000):
        pick = np.concatenate([idx[u] for u in us[r.integers(0, len(us), len(us))]]); bs.append(vals[pick].mean())
    return float(vals.mean()), [float(x) for x in np.percentile(bs, [2.5, 97.5])]
out["all_example_mean_dE4"] = tboot(dE4, t)
out["excl_lt20_example_mean_dE4"] = tboot(dE4[~small], t[~small])
(OUT / "RESULT.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
