"""Analysis of the clean-LAP test read (read 4). Written and hashed BEFORE any clean-LAP test output exists.
Imports the helper functions of clean_protocol_test_decision.py (sha256 405cc82b..., unchanged). Raises on anything missing.
Contrast of LAP_DECISION.txt: MEHF(Sigma*=HARM) minus MEHF-LAP on per-example channel spread, paired, 10,000-resample UniProt
cluster bootstrap, seed 20260925, on the usable examples of the completed ISO/HARM read intersected with LAP-finite examples."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import clean_protocol_test_decision as D  # noqa: E402

root = ROOT / "outputs/clean_protocol_v1/test_read"
ex = pd.read_csv(ROOT / "outputs/clean_protocol_v1/test_examples_clean.csv")
uniprot = dict(zip(ex["example_id"], ex["uniprot"]))
from mdwm.data.mmp_transfer import DEFAULT_STORE, load_system
n_heavy = {k: int(np.asarray(load_system(k.split("->")[1], DEFAULT_STORE, "pocket")["coords"]).shape[1]) for k in ex["example_id"]}
frozen = json.loads((ROOT / "outputs/clean_protocol_v1/s_star_frozen_lap.json").read_text())
report = {"read": "FOURTH read of the test split; arm chosen AFTER the ISO/HARM numbers were known (LAP_DECISION.txt)",
          "frozen_operating_points": "s_star_frozen_lap.json", "ops": {}}
for op, fk in (("primary", "PRIMARY_fitting_folds"), ("secondary", "SECONDARY_dev_folds")):
    lap = D.read_arm(root, f"test_LAP_{op}_f{{f}}")
    harm = D.read_arm(root, f"test_HARM_{op}_f{{f}}")
    iso = D.read_arm(root, f"test_ISO_{op}_f{{f}}")
    cc = {}
    for f in (1, 2, 3):
        c = D.config_check(root, f"test_LAP_{op}_f{f}", "frame0")
        want = frozen[fk]["LAP"][f"fold{f}"]["s_star"]
        if abs(c["internal_scale"] - want) > 1e-12:
            raise RuntimeError(f"operating point mismatch LAP {op} f{f}")
        cc[f"f{f}"] = c
    base = D.finite_keys(harm).intersection(D.finite_keys(iso))          # the 1,406 of the completed read
    keys = base.intersection(D.finite_keys(lap))
    lap_blk, lap_pe = D.arm_block(lap, keys, n_heavy, uniprot)
    harm_blk, harm_pe = D.arm_block(harm, keys, n_heavy, uniprot)
    iso_blk, iso_pe = D.arm_block(iso, keys, n_heavy, uniprot)
    if lap_blk["clash"] is None:
        raise RuntimeError("A4.2 gate: full-pocket clash rate missing for LAP")
    d_h_l = D.paired(harm_pe["spread"], lap_pe["spread"], uniprot)      # Sigma* minus LAP
    d_l_i = D.paired(lap_pe["spread"], iso_pe["spread"], uniprot)       # LAP minus ISO (context)
    d_h_i = D.paired(harm_pe["spread"], iso_pe["spread"], uniprot)      # reproduces the completed read
    per_fold = [float(((harm[f].loc[keys, list(D.CH)].max(axis=1) - harm[f].loc[keys, list(D.CH)].min(axis=1)) -
                       (lap[f].loc[keys, list(D.CH)].max(axis=1) - lap[f].loc[keys, list(D.CH)].min(axis=1))).mean()) for f in (1, 2, 3)]
    tors = lap_blk["realised_torsion_ratio_per_fold"]; pooled = lap_blk["realised_torsion_ratio_pooled_median"]
    b1 = {"per_fold": tors, "pooled_median": pooled,
          "per_fold_outside_window": [bool(not (D.B1_WINDOW[0] <= v <= D.B1_WINDOW[1])) for v in tors],
          "pooled_outside_window": bool(not (D.B1_WINDOW[0] <= pooled <= D.B1_WINDOW[1]))}
    b1["fires"] = bool(any(b1["per_fold_outside_window"]) or b1["pooled_outside_window"])
    b4 = {"clean_LAP_median_spread": lap_blk["median_spread"], "clean_HARM_median_spread": harm_blk["median_spread"],
          "fires": bool(lap_blk["median_spread"] <= harm_blk["median_spread"])}
    pred = {"between_ISO_and_Sigma_star": bool(harm_blk["median_spread"] < lap_blk["median_spread"] < iso_blk["median_spread"]),
            "median_in_1.1_1.4": bool(1.1 <= lap_blk["median_spread"] <= 1.4),
            "Sigma_minus_LAP_negative": bool(d_h_l["mean_difference"] < 0), "interval_excludes_zero": bool(d_h_l["excludes_zero"])}
    pred["all_components_hold"] = bool(all(pred.values()))
    report["ops"][op] = {"n_base_usable_ISO_HARM": int(len(base)), "n_usable_with_LAP": int(len(keys)),
                         "n_targets": int(len({uniprot[k] for k in keys})), "dropped_vs_base": int(len(base) - len(keys)),
                         "rows_produced_per_fold": {f: int(len(lap[f])) for f in (1, 2, 3)},
                         "config_check": cc, "clean_LAP": lap_blk, "clean_HARM_same_examples": harm_blk, "clean_ISO_same_examples": iso_blk,
                         "SigmaStar_minus_LAP": d_h_l, "SigmaStar_minus_LAP_per_fold_mean": per_fold,
                         "LAP_minus_ISO": d_l_i, "SigmaStar_minus_ISO_reproduced": d_h_i,
                         "B1_LAP": b1, "B4": b4, "prediction_LAP_DECISION": pred,
                         "median_psi": {"LAP": lap_blk["median_psi"], "HARM": harm_blk["median_psi"], "ISO": iso_blk["median_psi"]}}
out = ROOT / "reference_outputs/clean_protocol_v1/TEST_RESULT_LAP.json"
out.write_text(json.dumps(report, indent=2))
print(json.dumps({op: {k: report["ops"][op][k] for k in ("n_usable_with_LAP", "n_targets", "SigmaStar_minus_LAP", "B1_LAP", "B4", "prediction_LAP_DECISION")} for op in report["ops"]}, indent=1))
