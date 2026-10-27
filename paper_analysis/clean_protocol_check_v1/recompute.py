"""Independent recomputation (LD, 2026-10-26) of the clean-protocol test numbers from the per-example CSVs.
Written from the definitions in PREREGISTRATION.md (sha256 3863992a...), not by importing UConn's scripts.
Usage: python -I recompute.py <cleanrun dir>  -> prints a comparison against expected/TEST_RESULT*.json."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

R = Path(sys.argv[1]); TR = R / "expected/test_read"
CH = ["bond_length_ratio", "bond_angle_ratio", "dihedral_ratio", "planar_improper_ratio"]
ex = pd.read_csv(R / "inputs/test_examples_clean.csv")
uni = dict(zip(ex.example_id, ex.uniprot))
load = lambda stem: pd.read_csv(TR / f"{stem}.per_example.csv").set_index("key")
fr = {(a, op): [load(f"test_{a}_{op}_f{f}") for f in (1, 2, 3)] for a in ("ISO", "HARM", "LAP") for op in ("primary", "secondary")}
ctl = {s: [load(f"test_CONTROL_{s}")] for s in ("A_source_s4.0", "A_source_s4.0_proj", "A_source_s5.0", "A_source_s5.0_proj", "B_parent_copy", "B_parent_copy_proj")}

# usable: all four ratios finite in all three folds of both clean ISO/HARM arms (Addendum A), at both operating points
ok = None
for (a, op), dfs in fr.items():
    if a == "LAP": continue
    for d in dfs:
        m = np.isfinite(d[CH]).all(axis=1)
        ok = m if ok is None else (ok & m.reindex(ok.index, fill_value=False))
keys = ok[ok].index

def per_ex(dfs):
    sp = sum((d.loc[keys, CH].max(axis=1) - d.loc[keys, CH].min(axis=1)) for d in dfs) / len(dfs)
    ps = sum((d.loc[keys, CH].max(axis=1) - d.loc[keys, CH].min(axis=1)) / d.loc[keys, CH].mean(axis=1) for d in dfs) / len(dfs)
    rm = sum(d.loc[keys, "internal_rmsf_ratio"] for d in dfs) / len(dfs)
    anyclash = np.mean([(d.loc[keys, "full_clash_frame_rate"] > 0).mean() for d in dfs])
    return sp, ps, rm, anyclash

def boot(d, seed=20260925, n=10000):
    cl = np.array([uni[k] for k in d.index]); v = d.to_numpy()
    groups = [np.flatnonzero(cl == u) for u in sorted(set(cl))]
    rng = np.random.default_rng(seed); out = np.empty(n)
    for b in range(n):
        out[b] = v[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])].mean()
    return np.percentile(out, [2.5, 97.5])

T = json.loads((R / "expected/TEST_RESULT.json").read_text()); TL = json.loads((R / "expected/TEST_RESULT_LAP.json").read_text())
rows = []
def chk(name, got, want, tol=1e-6):
    rows.append(f"{'OK' if abs(got - want) <= tol else 'MISMATCH':8s} {name:45s} mine={got:.6f} theirs={want:.6f}")
print(f"usable examples {len(keys)}, targets {len({uni[k] for k in keys})}")
res = {}
for op in ("primary", "secondary"):
    for a in ("ISO", "HARM", "LAP"):
        res[(a, op)] = per_ex(fr[(a, op)])
    o = T["ops"][op]
    for a, blk in (("ISO", o["clean_ISO"]), ("HARM", o["clean_HARM"])):
        sp, ps, rm, ac = res[(a, op)]
        chk(f"{op} {a} median S", sp.median(), blk["median_spread"]); chk(f"{op} {a} median psi", ps.median(), blk["median_psi"])
        chk(f"{op} {a} RMSF", rm.median(), blk["internal_rmsf_ratio_median"]); chk(f"{op} {a} any-clash share", ac, blk["clash"]["full"]["examples_with_any_clash"])
    d = res[("HARM", op)][0] - res[("ISO", op)][0]; lo, hi = boot(d)
    c = o["primary_contrast_HARM_minus_ISO"]
    chk(f"{op} HARM-ISO mean", d.mean(), c["mean_difference"]); chk(f"{op} CI lo", lo, c["ci95"][0]); chk(f"{op} CI hi", hi, c["ci95"][1])
    chk(f"{op} share HARM lower", (d < 0).mean(), c["fraction_first_lower"])
    L = TL["ops"][op]; sp, ps, rm, ac = res[("LAP", op)]
    chk(f"{op} LAP median S", sp.median(), L["clean_LAP"]["median_spread"]); chk(f"{op} LAP psi", ps.median(), L["clean_LAP"]["median_psi"])
    d2 = res[("HARM", op)][0] - sp; lo, hi = boot(d2)
    chk(f"{op} HARM-LAP mean", d2.mean(), L["SigmaStar_minus_LAP"]["mean_difference"]); chk(f"{op} HARM-LAP CI lo", lo, L["SigmaStar_minus_LAP"]["ci95"][0]); chk(f"{op} HARM-LAP CI hi", hi, L["SigmaStar_minus_LAP"]["ci95"][1])
names = {"A_source_s4.0": "A_s4.0_off", "A_source_s4.0_proj": "A_s4.0_on", "A_source_s5.0": "A_s5.0_off", "A_source_s5.0_proj": "A_s5.0_on", "B_parent_copy": "B_off", "B_parent_copy_proj": "B_on"}
for s, dfs in ctl.items():
    sp, ps, rm, ac = per_ex(dfs); blk = T["controls"][names[s]]
    chk(f"control {names[s]} S", sp.median(), blk["median_spread"]); chk(f"control {names[s]} psi", ps.median(), blk["median_psi"])
    chk(f"control {names[s]} torsion", dfs[0].loc[keys, "dihedral_ratio"].median(), blk["realised_torsion_ratio_pooled_median"])
    chk(f"control {names[s]} any-clash share", ac, blk["clash"]["full"]["examples_with_any_clash"])
print("\n".join(rows)); print("MISMATCH count:", sum(r.startswith("MISMATCH") for r in rows))

# ---- extra values cited in the manuscript (added 2026-10-26) ----
M = Path(sys.argv[2]) if len(sys.argv) > 2 else None   # MEHF/reference_outputs, for the frozen original-protocol CSVs
extra = []
def chk2(name, got, want, tol=1e-6):
    extra.append(f"{'OK' if abs(got - want) <= tol else 'MISMATCH':8s} {name:45s} mine={got:.6f} theirs={want:.6f}")
for op in ("primary", "secondary"):
    d = res[("LAP", op)][0] - res[("ISO", op)][0]; lo, hi = boot(d); L = TL["ops"][op]["LAP_minus_ISO"]
    chk2(f"{op} LAP-ISO mean", d.mean(), L["mean_difference"]); chk2(f"{op} LAP-ISO CI lo", lo, L["ci95"][0]); chk2(f"{op} LAP-ISO CI hi", hi, L["ci95"][1])
    for f in range(3):
        sp = lambda a: fr[(a, op)][f].loc[keys, CH].max(axis=1) - fr[(a, op)][f].loc[keys, CH].min(axis=1)
        chk2(f"{op} fold{f+1} HARM-ISO mean", (sp("HARM") - sp("ISO")).mean(), T["ops"][op]["per_fold_mean_difference"][f])
        chk2(f"{op} fold{f+1} HARM torsion median", fr[("HARM", op)][f].loc[keys, "dihedral_ratio"].median(), T["ops"][op]["clean_HARM"]["realised_torsion_ratio_per_fold"][f])
    h = res[("HARM", op)]
    chk2(f"{op} HARM channel bond", (sum(x.loc[keys, CH[0]] for x in fr[("HARM", op)]) / 3).median(), T["ops"][op]["clean_HARM"]["channel_ratio_median"][CH[0]])
if M is not None:
    old = [pd.read_csv(M / f"sigma_star_test_v1/test_{a}_f{f}.per_example.csv").set_index("key") for a in ("SIGMA_star", "ISO") for f in (1, 2, 3)]
    spo = lambda dfs: sum((x.loc[keys, CH].max(axis=1) - x.loc[keys, CH].min(axis=1)) for x in dfs) / 3
    dl = spo(old[:3]) - spo(old[3:])
    chk2("original-protocol HARM-ISO on the 1,406", dl.mean(), T["ops"]["primary"]["DiD_vs_leaky"]["delta_leaky_on_common"])
    chk2("original-protocol HARM S on the 1,406", spo(old[:3]).median(), T["ops"]["primary"]["DiD_vs_leaky"]["delta_leaky_on_common"] * 0 + T["ops"]["primary"]["B2"]["leaky_SIGMA_frozen_median_spread_same_examples"])
    for op in ("primary", "secondary"):
        did = (res[("HARM", op)][0] - res[("ISO", op)][0]) - dl; lo, hi = boot(did); D = T["ops"][op]["DiD_vs_leaky"]
        chk2(f"{op} DiD", did.mean(), D["DiD"]); chk2(f"{op} DiD CI lo", lo, D["DiD_ci95"][0]); chk2(f"{op} DiD CI hi", hi, D["DiD_ci95"][1])
cr = pd.read_csv(R / "expected/test_read/crop_recall_per_system.csv"); CJ = json.loads((R / "expected/test_read/crop_recall.json").read_text())
for sub, df in (("all_corpus_post_exclusion", cr), ("test_list_systems", cr[cr.in_test])):
    for k in ("overlap", "recall_pair"):
        chk2(f"crop {sub} {k} median", df[k].median(), CJ[sub][k]["median"]); chk2(f"crop {sub} {k} min", df[k].min(), CJ[sub][k]["min"])
print("\n".join(extra)); print("EXTRA MISMATCH count:", sum(r.startswith("MISMATCH") for r in extra))

# ---- every cell of the clean-protocol table (added 2026-10-26) ----
cells = []
def row(name, dfs, blk):
    n = len(dfs)
    for c in CH:
        cells.append((f"{name} {c}", (sum(x.loc[keys, c] for x in dfs) / n).median(), blk["channel_ratio_median"][c]))
    cells.append((f"{name} RMSF", (sum(x.loc[keys, "internal_rmsf_ratio"] for x in dfs) / n).median(), blk["internal_rmsf_ratio_median"]))
    cells.append((f"{name} any-clash", np.mean([(x.loc[keys, "full_clash_frame_rate"] > 0).mean() for x in dfs]), blk["clash"]["full"]["examples_with_any_clash"]))
for op in ("primary", "secondary"):
    row(f"{op} ISO", fr[("ISO", op)], T["ops"][op]["clean_ISO"]); row(f"{op} HARM", fr[("HARM", op)], T["ops"][op]["clean_HARM"])
    row(f"{op} LAP", fr[("LAP", op)], TL["ops"][op]["clean_LAP"])
for s, dfs in ctl.items():
    row(f"control {names[s]}", dfs, T["controls"][names[s]])
bad = [c for c in cells if abs(c[1] - c[2]) > 1e-9]
print(f"TABLE CELLS checked {len(cells)}, mismatches {len(bad)}")
for c in cells: print(f"  {c[0]:45s} {c[1]:.4f}")
# operating points actually used, from each run's own json
for a, op in [(a, op) for a in ("ISO", "HARM", "LAP") for op in ("primary", "secondary")]:
    print("s* used", a, op, [round(json.loads((TR / f"test_{a}_{op}_f{f}.json").read_text())["internal_scale"], 3) for f in (1, 2, 3)])
