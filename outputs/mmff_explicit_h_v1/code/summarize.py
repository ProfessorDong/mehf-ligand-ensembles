import json, sys, glob
from pathlib import Path
import numpy as np
M = Path(sys.argv[1]); O = Path(sys.argv[2])
SEEDS = ["20260924", "1234", "7", "31337", "99"]
out = {}
# --- reproduction of original gate arms
refdirs = {"diffsbdd": M/"reference_outputs/corrector_gate_v1/scaleup_with_r5",
           "targetdiff": M/"reference_outputs/published_models_v1/targetdiff"}
repro = {}
for ds, rd in refdirs.items():
    worst = 0.0; nf = 0; mism = []
    for s in SEEDS:
        ref = json.loads((rd/f"holdout2000_{s}.json").read_text())
        new = json.loads((O/f"gates/{ds}/holdout2000_{s}.json").read_text())
        assert ref["n_molecules"] == new["n_molecules"]
        for arm, fields in ref["arms"].items():
            for k, v in fields.items():
                w = new["arms"][arm][k]; nf += 1
                if v is None or w is None:
                    if v != w: mism.append((s, arm, k, v, w))
                    continue
                rel = abs(v - w) / max(abs(v), 1e-12)
                worst = max(worst, rel if abs(v) > 1e-9 else abs(v - w))
                if rel > 1e-9 and abs(v - w) > 1e-12: mism.append((s, arm, k, v, w))
    repro[ds] = {"fields_compared": nf, "max_rel_diff": worst, "mismatches": mism,
                 "n_molecules": ref["n_molecules"]}
out["gate_reproduction"] = repro
# --- gate tables (5-split mean of per-split values)
def table(dirpath):
    R = [json.loads((dirpath/f"holdout2000_{s}.json").read_text()) for s in SEEDS]
    t = {}
    for arm in R[0]["arms"]:
        e = {}
        for k in ["angle_rms", "wasserstein_angle", "planarity", "rmsd_from_raw", "bond_rms",
                  "wasserstein_bond", "clashes"]:
            vals = [r["arms"][arm][k] for r in R]
            e[k] = float(np.mean(vals)); e[k + "_split_sd"] = float(np.std(vals, ddof=1))
        for k in ["frac_with_clash", "clash_mean", "n_evaluated"]:
            e[k] = float(np.mean([r["extra_fields"][arm][k] for r in R]))
        t[arm] = e
    status = R[0]["minimize_status"]
    return t, status, R[0]["n_molecules"]
for ds in ["diffsbdd", "targetdiff", "diffsbdd_common_subset"]:
    t, st, n = table(O/"gates"/ds)
    out[f"gate_{ds}"] = {"n_molecules": n, "minimize_status_seed20260924": st, "arms": t}
# --- sweep
sw = json.loads((O/"sweep/sweep_explicit_h.json").read_text())
ref = json.loads((M/"reference_outputs/paper_evidence_v1/e_ff_restraint_sweep.json").read_text())
mism = []; worst = 0.0
for arm, f in ref["arms"].items():
    for k, v in f.items():
        w = sw["arms"][arm][k]
        rel = abs(v - w) / max(abs(v), 1e-12) if abs(v) > 1e-9 else abs(v - w)
        worst = max(worst, rel)
        if rel > 1e-9: mism.append((arm, k, v, w))
out["sweep_reproduction_within_corrected_run"] = {"max_rel_diff": worst, "mismatches": mism}
out["sweep"] = sw
(O/"SUMMARY.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out["gate_reproduction"], indent=1)); print(out["sweep_reproduction_within_corrected_run"])
order = ["A_raw", "B_mmff_free", "HB_mmffH_free", "C_mmff_restrained", "HC_mmffH_restraint100",
         "G_mmff_restraint5", "HG_mmffH_restraint5", "D_projection", "E_projection_plus_improper"]
for ds in ["diffsbdd", "targetdiff", "diffsbdd_common_subset"]:
    print(ds)
    for a in order:
        e = out[f"gate_{ds}"]["arms"][a]
        print(f"  {a:28s} {e['angle_rms']:6.2f} {e['wasserstein_angle']:5.2f} {e['planarity']:6.2f} {e['rmsd_from_raw']:5.2f} | sd {e['angle_rms_split_sd']:.3f} {e['wasserstein_angle_split_sd']:.3f} {e['planarity_split_sd']:.3f} {e['rmsd_from_raw_split_sd']:.4f} | Wb {e['wasserstein_bond']:.4f} clash% {100*e['frac_with_clash']:.1f} n {e['n_evaluated']:.0f}")
for a, e in sw["arms"].items():
    print(f"{a:12s} {e['rmsd_from_raw']:.3f} {e['wasserstein_bond']:.4f} {e['wasserstein_angle']:.2f} {e['planarity']:.2f} {100*e['frac_with_clash']:.1f} | int {e['rmsd_internal']:.3f} clashmean {e['clash_mean']:.2f}")
