"""Aggregate the B2 clash comparison (leaky vs clean) from outputs/clean_protocol_v1/b2_clash/*.json."""
import itertools, json, os
from pathlib import Path
import numpy as np, pandas as pd
os.chdir(Path(__file__).resolve().parents[1])
D = Path("outputs/clean_protocol_v1/b2_clash")
rows = []
for p in sorted(D.glob("*_f?_*.json")):
    if p.name == "operating_points.json": continue
    name, f, sample = p.stem.rsplit("_", 2)
    d = json.loads(p.read_text()); pe = pd.read_csv(p.with_suffix(".per_example.csv"))
    for tag in ("full", "crop", "outside"):
        rows.append(dict(arm=name, fold=int(f[1:]), sample=sample, region=tag, s=d["internal_scale"], n=d["n_examples_sampled"],
                         frame_rate=d[f"{tag}_pocket_clash"]["frame_rate_mean"],
                         any_clash=d[f"{tag}_pocket_clash"]["examples_with_any_clash"],
                         torsion=d["dihedral"]["ratio_median"],
                         pooled_frames_hit=float((pe[f"{tag}_clash_frame_rate"]).mean())))
df = pd.DataFrame(rows); df.to_csv(D / "b2_clash_long.csv", index=False)
out = {}
# per arm, mean over folds (equal weight) of frame_rate / any_clash, per sample and region
agg = df.groupby(["arm", "sample", "region"])[["frame_rate", "any_clash", "torsion"]].mean().reset_index()
out["per_arm_mean_over_folds"] = agg.to_dict("records")
# B2 tolerance: max over ordered pairs of LEAKY arms of |rate_a - rate_b|, same example set (fold, sample), full region
leaky = [a for a in df.arm.unique() if a.startswith("leaky_")]
def maxdiff(arms, region):
    best = None
    for (fold, sample), g in df[(df.region == region) & df.arm.isin(arms)].groupby(["fold", "sample"]):
        r = g.set_index("arm").frame_rate
        for a, b in itertools.permutations(arms, 2):
            v = abs(r[a] - r[b])
            if best is None or v > best["abs_diff"]: best = dict(abs_diff=float(v), arms=[a, b], fold=int(fold), sample=sample, rates=[float(r[a]), float(r[b])])
    return best
three = ["leaky_ISO", "leaky_ANISO", "leaky_SIGMA"]
out["B2_tolerance_max_pairwise_leaky_full"] = {"three_prereg_arms_ISO_GS_Sigma": maxdiff(three, "full"), "all_four_incl_LAP": maxdiff(leaky, "full")}
out["B2_tolerance_outside_region"] = {"three": maxdiff(three, "outside"), "four": maxdiff(leaky, "outside")}
# clean minus leaky per arm pair (same folds & samples), full and outside; clean at primary and secondary s*
pairs = {"ISO": ("clean_ISO", "leaky_ISO"), "HARM_vs_SIGMA": ("clean_HARM", "leaky_SIGMA")}
cmp = []
for key, (c, l) in pairs.items():
    for which in ("primary", "secondary"):
        for region in ("full", "crop", "outside"):
            for sample in ("fit", "dev"):
                a = df[(df.arm == f"{c}_{which}") & (df.region == region) & (df["sample"] == sample)].set_index("fold").frame_rate
                b = df[(df.arm == l) & (df.region == region) & (df["sample"] == sample)].set_index("fold").frame_rate
                cmp.append(dict(pair=key, clean_s_star=which, region=region, sample=sample, clean_mean=float(a.mean()), leaky_mean=float(b.mean()),
                                diff_clean_minus_leaky=float((a - b).mean()), per_fold_diff=[float(x) for x in (a - b).sort_index()]))
out["clean_minus_leaky"] = cmp
(D / "b2_clash_summary.json").write_text(json.dumps(out, indent=1))
pd.set_option("display.width", 250)
print(agg[agg.region != "crop"].pivot_table(index=["arm"], columns=["sample", "region"], values="frame_rate").round(5).to_string())
print(agg.pivot_table(index=["arm"], columns=["sample", "region"], values="any_clash").round(3).to_string())
print(json.dumps({k: out[k] for k in ("B2_tolerance_max_pairwise_leaky_full", "B2_tolerance_outside_region")}, indent=1))
print(pd.DataFrame(cmp)[lambda x: x.region != "crop"][["pair", "clean_s_star", "region", "sample", "clean_mean", "leaky_mean", "diff_clean_minus_leaky"]].round(5).to_string())
