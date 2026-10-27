"""Analysis of the single clean-protocol test read (read 3).  Written and hashed BEFORE any test_* output exists.

Applies, mechanically, PREREGISTRATION.md (sha256 3863992a...): section 5.1 primary, section 6 Tiers, Amendment 3 reading
rule, blocking conditions B1-B4 (B3 as amended by Amendment 5, B2 as pinned by Amendment 4), and the quantities needed to
score P1-P10.  Reads only files named below.  Raises on anything missing; no default is ever substituted.

  --root   directory holding the read's outputs (test_{ISO,HARM}_{primary,secondary}_f{k}.*, test_LEAKY_*_f{k}.*,
           test_CONTROL_*.*)
  --frozen-leaky-dir  where the frozen read-1/read-2 per-example CSVs live (reference_outputs by default)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CH = ("bond_length_ratio", "bond_angle_ratio", "dihedral_ratio", "planar_improper_ratio")
N_BOOT, BOOT_SEED = 10000, 20260925
READ1_CUT, ATTEN_CUT = -1.021, -0.30
B1_WINDOW = (0.85, 1.25)
TIER_ORDER = {"R": 3, "A": 2, "W": 1, "N": 0}


# ----------------------------------------------------------------------------------------------- helpers
def read_arm(root: Path, stem_fmt: str, folds=(1, 2, 3)) -> dict[int, pd.DataFrame]:
    out = {}
    for f in folds:
        p = root / (stem_fmt.format(f=f) + ".per_example.csv")
        if not p.exists():
            raise FileNotFoundError(p)
        out[f] = pd.read_csv(p).set_index("key")
    return out


def finite_keys(frames: dict[int, pd.DataFrame]) -> pd.Index:
    keys = None
    for df in frames.values():
        ok = df.index[np.isfinite(df[list(CH)].to_numpy()).all(axis=1)]
        keys = ok if keys is None else keys.intersection(ok)
    return keys


def per_example(frames: dict[int, pd.DataFrame], keys: pd.Index) -> pd.DataFrame:
    """Per-example spread and psi (each computed per fold checkpoint, then averaged over folds), channel ratios and RMSF ratio
    (averaged over folds)."""
    sp, ps, ch, rm = [], [], [], []
    for df in frames.values():
        r = df.loc[keys, list(CH)]
        spread = r.max(axis=1) - r.min(axis=1)
        sp.append(spread); ps.append(spread / r.mean(axis=1))
        ch.append(r); rm.append(df.loc[keys, "internal_rmsf_ratio"])
    n = len(frames)
    out = pd.DataFrame({"spread": sum(sp) / n, "psi": sum(ps) / n, "rmsf": sum(rm) / n})
    mean_r = sum(ch) / n
    for c in CH:
        out[c] = mean_r[c]
    return out


def cluster_bootstrap(diff: np.ndarray, cluster: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOT_SEED)
    groups = pd.Series(range(len(diff))).groupby(cluster).apply(lambda s: np.asarray(s)).tolist()
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        picked = rng.integers(0, len(groups), len(groups))
        draws[b] = diff[np.concatenate([groups[p] for p in picked])].mean()
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def paired(a: pd.Series, b: pd.Series, uniprot: dict) -> dict:
    keys = a.index.intersection(b.index)
    d = (a.loc[keys] - b.loc[keys]).to_numpy()
    lo, hi = cluster_bootstrap(d, np.array([uniprot[k] for k in keys]))
    return {"n_examples": int(len(keys)), "n_targets": int(len({uniprot[k] for k in keys})),
            "mean_difference": float(d.mean()), "median_difference": float(np.median(d)), "ci95": [lo, hi],
            "excludes_zero": bool(lo > 0 or hi < 0), "fraction_first_lower": float((d < 0).mean())}


def tier(delta: dict, med_harm: float, med_iso: float) -> dict:
    """Section 6, applied verbatim.  N is checked first (CI contains 0, sign reverses, median not below the comparator)."""
    m = delta["mean_difference"]
    if (not delta["excludes_zero"]) or m >= 0 or not (med_harm < med_iso):
        t = "N"
    elif m <= READ1_CUT:
        t = "R"
    elif m <= ATTEN_CUT:
        t = "A"
    else:
        t = "W"
    return {"tier": t, "mean_difference": m, "ci95": delta["ci95"], "median_below_iso": bool(med_harm < med_iso)}


def pooled_clash(frames: dict[int, pd.DataFrame], keys: pd.Index, n_heavy: dict) -> dict:
    """Quantity (i) of Amendment 4: clashing (ligand-heavy, pocket-heavy) pairs over all pairs, pooled over examples, per fold,
    then averaged over folds.  Also frame rate, median pairs/frame, fraction of examples with >=1 clash."""
    res = {}
    for region, size_col in (("full", "n_pocket_available"), ("outside", "n_outside_frame0_crop")):
        per_fold = []
        for df in frames.values():
            d = df.loc[keys]
            npairs = np.array([n_heavy[k] for k in keys]) * d[size_col].to_numpy()
            frac = d[f"{region}_clash_pair_fraction"].to_numpy()
            per_fold.append(dict(
                pooled_pair_fraction=float((frac * npairs).sum() / npairs.sum()),
                frame_rate_mean=float(d[f"{region}_clash_frame_rate"].mean()),
                pairs_per_frame_median=float(d[f"{region}_clash_pairs_per_frame"].median()),
                examples_with_any_clash=float((d[f"{region}_clash_frame_rate"] > 0).mean())))
        res[region] = {k: float(np.mean([p[k] for p in per_fold])) for k in per_fold[0]}
        res[region]["per_fold_pooled_pair_fraction"] = [p["pooled_pair_fraction"] for p in per_fold]
    return res


def torsion_by_fold(frames, keys) -> list[float]:
    return [float(df.loc[keys, "dihedral_ratio"].median()) for df in frames.values()]


def arm_block(frames, keys, n_heavy, uniprot) -> dict:
    pe = per_example(frames, keys)
    return {
        "n_examples": int(len(keys)),
        "median_spread": float(pe["spread"].median()), "q25": float(pe["spread"].quantile(.25)), "q75": float(pe["spread"].quantile(.75)),
        "median_psi": float(pe["psi"].median()),
        "channel_ratio_median": {c: float(pe[c].median()) for c in CH},
        "internal_rmsf_ratio_median": float(pe["rmsf"].median()),
        "realised_torsion_ratio_per_fold": torsion_by_fold(frames, keys),
        "realised_torsion_ratio_pooled_median": float(pe["dihedral_ratio"].median()),
        "per_fold_median_spread": [float((df.loc[keys, list(CH)].max(axis=1) - df.loc[keys, list(CH)].min(axis=1)).median()) for df in frames.values()],
        "per_fold_spread_of_medians": [float(df.loc[keys, list(CH)].median().max() - df.loc[keys, list(CH)].median().min()) for df in frames.values()],
        "clash": pooled_clash(frames, keys, n_heavy) if "full_clash_pair_fraction" in next(iter(frames.values())).columns else None,
    }, pe


def single_block(df: pd.DataFrame, keys, n_heavy) -> tuple[dict, pd.DataFrame]:
    return arm_block({1: df}, keys, n_heavy, None)


def config_check(root: Path, stem: str, want_pocket: str) -> dict:
    j = json.loads((root / (stem + ".json")).read_text())
    got = j["config_check"]["pocket_selection"]
    if got != want_pocket:
        raise RuntimeError(f"T7: {stem} resolved pocket_selection={got!r}, expected {want_pocket!r}")
    return {"pocket_selection": got, "harmonic_tau_set": j["config_check"]["harmonic_tau_set"], "internal_scale": j["internal_scale"],
            "n_examples_sampled": j["n_examples_sampled"]}


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "outputs/clean_protocol_v1/test_read")
    ap.add_argument("--frozen-leaky-dir", type=Path, default=ROOT / "reference_outputs")
    ap.add_argument("--examples", type=Path, default=ROOT / "outputs/clean_protocol_v1/test_examples_clean.csv")
    ap.add_argument("--frozen-op", type=Path, default=ROOT / "outputs/clean_protocol_v1/s_star_frozen_v2.json")
    ap.add_argument("--out-json", type=Path, default=ROOT / "reference_outputs/clean_protocol_v1/TEST_RESULT.json")
    ap.add_argument("--control-a-scales", type=float, nargs="+", default=[4.0, 5.0])
    args = ap.parse_args()
    root = args.root

    ex = pd.read_csv(args.examples)
    uniprot = dict(zip(ex["example_id"], ex["uniprot"]))
    from mdwm.data.mmp_transfer import DEFAULT_STORE, load_system   # heavy-atom counts only
    n_heavy = {}
    for k in ex["example_id"]:
        n_heavy[k] = int(np.asarray(load_system(k.split("->")[1], DEFAULT_STORE, "pocket")["coords"]).shape[1])

    frozen = json.loads(args.frozen_op.read_text())
    report: dict = {"n_examples_in_clean_list": int(len(ex)), "n_targets_in_clean_list": int(ex["uniprot"].nunique()),
                    "n_pairs_in_clean_list": int(ex["pair_key"].nunique()),
                    "frozen_operating_points": args.frozen_op.name, "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED}, "ops": {}}

    # ---- clean arms at both operating points
    arm_frames, arm_keys = {}, {}
    for op in ("primary", "secondary"):
        for arm in ("ISO", "HARM"):
            arm_frames[(op, arm)] = read_arm(root, f"test_{arm}_{op}_f{{f}}")
        keys = finite_keys(arm_frames[(op, "ISO")]).intersection(finite_keys(arm_frames[(op, "HARM")]))
        arm_keys[op] = keys
    # leaky arms (clash on the post-exclusion list, own frozen operating point) and frozen leaky per-example CSVs
    leaky_names = {"ISO": "ISO", "GS": "GS", "LAP": "LAP", "SIGMA": "SIGMA"}
    leaky_frames = {n: read_arm(root, f"test_LEAKY_{n}_f{{f}}") for n in leaky_names}
    frozen_csv = {
        "ISO": read_arm(args.frozen_leaky_dir / "sigma_star_test_v1", "test_ISO_f{f}"),
        "GS": read_arm(args.frozen_leaky_dir / "sigma_star_test_v1", "test_ANISO_v3_f{f}"),
        "SIGMA": read_arm(args.frozen_leaky_dir / "sigma_star_test_v1", "test_SIGMA_star_f{f}"),
        "LAP": read_arm(args.frozen_leaky_dir / "laplacian_prior_v1", "test_LAPLACIAN_f{f}"),
    }
    # config checks (T7)
    cc = {}
    for op in ("primary", "secondary"):
        for arm in ("ISO", "HARM"):
            for f in (1, 2, 3):
                cc[f"{arm}_{op}_f{f}"] = config_check(root, f"test_{arm}_{op}_f{f}", "frame0")
                want = frozen[("PRIMARY_fitting_folds" if op == "primary" else "SECONDARY_dev_folds")][arm][f"fold{f}"]["s_star"]
                if abs(cc[f"{arm}_{op}_f{f}"]["internal_scale"] - want) > 1e-12:
                    raise RuntimeError(f"operating point mismatch {arm} {op} f{f}")
    for n in leaky_names:
        for f in (1, 2, 3):
            cc[f"LEAKY_{n}_f{f}"] = config_check(root, f"test_LEAKY_{n}_f{f}", "min_over_frames")
    report["config_check"] = cc

    common_all = arm_keys["primary"].intersection(arm_keys["secondary"])
    report["example_counts"] = {
        "listed": int(len(ex)),
        "rows_produced_per_fold": {f"{arm}_{op}_f{f}": int(len(arm_frames[(op, arm)][f])) for op in ("primary", "secondary") for arm in ("ISO", "HARM") for f in (1, 2, 3)},
        "usable_primary": int(len(arm_keys["primary"])), "usable_secondary": int(len(arm_keys["secondary"])),
        "usable_both": int(len(common_all)),
        "dropped_no_finite_four_channels_primary": int(len(ex) - len(arm_keys["primary"])),
        "targets_usable_primary": int(len({uniprot[k] for k in arm_keys["primary"]})),
        "targets_usable_secondary": int(len({uniprot[k] for k in arm_keys["secondary"]})),
    }

    # frozen leaky per-example spreads on the same examples (Delta_leaky for P3, B2's spread comparison)
    frozen_pe = {}
    for n, fr in frozen_csv.items():
        keys = finite_keys(fr)
        frozen_pe[n] = per_example(fr, keys)

    primary_tiers = {}
    for op in ("primary", "secondary"):
        keys = arm_keys[op]
        iso_blk, iso_pe = arm_block(arm_frames[(op, "ISO")], keys, n_heavy, uniprot)
        harm_blk, harm_pe = arm_block(arm_frames[(op, "HARM")], keys, n_heavy, uniprot)
        delta = paired(harm_pe["spread"], iso_pe["spread"], uniprot)
        per_fold_delta = []
        for f in (1, 2, 3):
            h = arm_frames[(op, "HARM")][f].loc[keys, list(CH)]; i = arm_frames[(op, "ISO")][f].loc[keys, list(CH)]
            per_fold_delta.append(float(((h.max(axis=1) - h.min(axis=1)) - (i.max(axis=1) - i.min(axis=1))).mean()))
        t = tier(delta, harm_blk["median_spread"], iso_blk["median_spread"])
        primary_tiers[op] = t["tier"]
        # B1
        b1 = {}
        for arm, blk in (("ISO", iso_blk), ("HARM", harm_blk)):
            per_fold = blk["realised_torsion_ratio_per_fold"]
            b1[arm] = {"per_fold": per_fold, "pooled_median": blk["realised_torsion_ratio_pooled_median"],
                       "per_fold_outside_window": [bool(not (B1_WINDOW[0] <= v <= B1_WINDOW[1])) for v in per_fold],
                       "pooled_outside_window": bool(not (B1_WINDOW[0] <= blk["realised_torsion_ratio_pooled_median"] <= B1_WINDOW[1])),
                       "json_reported_per_fold": [float(json.loads((root / f"test_{arm}_{op}_f{f}.json").read_text())["dihedral"]["ratio_median"]) for f in (1, 2, 3)]}
        b1_fires = any(any(v["per_fold_outside_window"]) or v["pooled_outside_window"] for v in b1.values())
        # fitting-fold -> test degradation (P4)
        fit_key = "PRIMARY_fitting_folds" if op == "primary" else "SECONDARY_dev_folds"
        deg = {}
        for arm, blk in (("ISO", iso_blk), ("HARM", harm_blk)):
            fit_spread = [frozen[fit_key][arm][f"fold{f}"]["spread_at_s_star"] for f in (1, 2, 3)]
            deg[arm] = {"fit_spread_at_s_star_per_fold": fit_spread,
                        "test_headline_median_over_fit_per_fold": [blk["median_spread"] / s for s in fit_spread],
                        "test_headline_median_over_mean_fit": blk["median_spread"] / float(np.mean(fit_spread)),
                        "test_per_fold_spread_of_medians_over_fit": [a / b for a, b in zip(blk["per_fold_spread_of_medians"], fit_spread)]}
        # DiD vs leaky (P3): common set of clean usable keys and frozen leaky usable keys
        fk = keys.intersection(frozen_pe["ISO"].index).intersection(frozen_pe["SIGMA"].index)
        clean_d = (harm_pe["spread"] - iso_pe["spread"]).loc[fk]
        leaky_d = (frozen_pe["SIGMA"]["spread"] - frozen_pe["ISO"]["spread"]).loc[fk]
        did_vec = (clean_d - leaky_d).to_numpy()
        lo, hi = cluster_bootstrap(did_vec, np.array([uniprot[k] for k in fk]))
        lo_c, hi_c = cluster_bootstrap(clean_d.to_numpy(), np.array([uniprot[k] for k in fk]))
        lo_l, hi_l = cluster_bootstrap(leaky_d.to_numpy(), np.array([uniprot[k] for k in fk]))
        did = {"n_common": int(len(fk)), "n_frozen_usable": int(len(frozen_pe["ISO"].index.intersection(frozen_pe["SIGMA"].index))),
               "n_clean_usable": int(len(keys)),
               "delta_clean_on_common": float(clean_d.mean()), "delta_clean_ci95": [lo_c, hi_c],
               "delta_leaky_on_common": float(leaky_d.mean()), "delta_leaky_ci95": [lo_l, hi_l],
               "DiD": float(did_vec.mean()), "DiD_ci95": [lo, hi]}
        report["ops"][op] = {"clean_ISO": iso_blk, "clean_HARM": harm_blk, "primary_contrast_HARM_minus_ISO": delta,
                             "per_fold_mean_difference": per_fold_delta, "tier": t, "B1": {"fires": b1_fires, "detail": b1},
                             "degradation_fit_to_test": deg, "DiD_vs_leaky": did}

    # ---- leaky arms: clash on the post-exclusion list at each arm's own frozen op, plus reproduction check
    leaky_blocks, leaky_rate = {}, {}
    ckeys = arm_keys["primary"]
    for n in leaky_names:
        fr = leaky_frames[n]
        keys = finite_keys(fr).intersection(ckeys)
        blk, pe = arm_block(fr, keys, n_heavy, uniprot)
        leaky_blocks[n] = blk
        leaky_rate[n] = {"pair_fraction": blk["clash"]["full"]["pooled_pair_fraction"], "frame_rate": blk["clash"]["full"]["frame_rate_mean"]}
        # reproduction of the frozen per-example spread by the fresh inference run
        both = pe.index.intersection(frozen_pe[n].index)
        blk["reproduction_vs_frozen_csv"] = {"n": int(len(both)),
                                             "max_abs_spread_diff": float((pe.loc[both, "spread"] - frozen_pe[n].loc[both, "spread"]).abs().max()),
                                             "median_spread_fresh": float(pe.loc[both, "spread"].median()),
                                             "median_spread_frozen": float(frozen_pe[n].loc[both, "spread"].median())}
        # frozen median on the clean common set (for DiD / B2 spread comparison)
        blk["frozen_median_spread_on_clean_common"] = float(frozen_pe[n].loc[frozen_pe[n].index.intersection(ckeys), "spread"].median())
    report["leaky_arms"] = leaky_blocks
    import itertools
    def tol(names, field):
        return max(abs(leaky_rate[a][field] - leaky_rate[b][field]) for a, b in itertools.permutations(names, 2))
    b2_tol = {f: {"four_arms_ISO_GS_LAP_SIGMA(pinned, A4.3)": tol(["ISO", "GS", "LAP", "SIGMA"], f),
                  "three_arms_ISO_GS_SIGMA": tol(["ISO", "GS", "SIGMA"], f)} for f in ("pair_fraction", "frame_rate")}
    report["B2_tolerance"] = b2_tol
    for op in ("primary", "secondary"):
        hb = report["ops"][op]["clean_HARM"]
        leaky_sig_med = leaky_blocks["SIGMA"]["frozen_median_spread_on_clean_common"]
        improves = bool(hb["median_spread"] < leaky_sig_med)
        res = {"clean_HARM_median_spread": hb["median_spread"], "leaky_SIGMA_frozen_median_spread_same_examples": leaky_sig_med,
               "spread_improves_vs_leaky": improves}
        for f, key in (("pair_fraction", "pooled_pair_fraction"), ("frame_rate", "frame_rate_mean")):
            worsen = hb["clash"]["full"][key] - leaky_rate["SIGMA"][f]
            res[f"clash_{f}"] = {"clean_HARM": hb["clash"]["full"][key], "leaky_SIGMA": leaky_rate["SIGMA"][f], "worsening": worsen,
                                 "exceeds_tolerance_pinned_4arm": bool(worsen > b2_tol[f]["four_arms_ISO_GS_LAP_SIGMA(pinned, A4.3)"]),
                                 "exceeds_tolerance_3arm": bool(worsen > b2_tol[f]["three_arms_ISO_GS_SIGMA"])}
        res["fires"] = bool(improves and res["clash_pair_fraction"]["exceeds_tolerance_pinned_4arm"])
        res["fires_under_any_variant_of_the_definition"] = bool(improves and any(
            res[f"clash_{f}"][k] for f in ("pair_fraction", "frame_rate") for k in ("exceeds_tolerance_pinned_4arm", "exceeds_tolerance_3arm")))
        report["ops"][op]["B2"] = res

    # ---- controls
    controls = {}
    def ctrl(stem):
        df = pd.read_csv(root / (stem + ".per_example.csv")).set_index("key")
        return df, json.loads((root / (stem + ".json")).read_text())
    variants = {"B_off": "test_CONTROL_B_parent_copy", "B_on": "test_CONTROL_B_parent_copy_proj"}
    for s in args.control_a_scales:
        variants[f"A_s{s}_off"] = f"test_CONTROL_A_source_s{s}"
        variants[f"A_s{s}_on"] = f"test_CONTROL_A_source_s{s}_proj"
    for name, stem in variants.items():
        df, js = ctrl(stem)
        keys = finite_keys({1: df}).intersection(ckeys)
        blk, pe = single_block(df, keys, n_heavy)
        blk["projection"] = js["projection"]; blk["n_rows_produced"] = int(len(df)); blk["device"] = js.get("device")
        t_tor = blk["realised_torsion_ratio_pooled_median"]
        blk["matched"] = bool(B1_WINDOW[0] <= t_tor <= B1_WINDOW[1])
        controls[name] = blk
    report["controls"] = controls
    for op in ("primary", "secondary"):
        hb = report["ops"][op]["clean_HARM"]
        b3 = {}
        for name, blk in controls.items():
            if blk["matched"]:
                fires = bool(blk["median_spread"] < hb["median_spread"])
                rule = "(a) matched: fires iff spread below clean HARM"
            else:
                fires = bool(blk["median_psi"] < hb["median_psi"])
                rule = "(b) unmatched: fires iff median psi below clean HARM's (spread cannot fire it)"
            b3[name] = {"matched": blk["matched"], "rule": rule, "fires": fires, "median_spread": blk["median_spread"],
                        "median_psi": blk["median_psi"], "torsion_ratio": blk["realised_torsion_ratio_pooled_median"],
                        "clean_HARM_median_spread": hb["median_spread"], "clean_HARM_median_psi": hb["median_psi"]}
        report["ops"][op]["B3"] = {
            "per_variant": b3,
            "fires_projection_ON_variants_only_(section6_wording)": bool(any(v["fires"] for k, v in b3.items() if k.endswith("_on"))),
            "fires_any_variant_(amendment5_wording)": bool(any(v["fires"] for v in b3.values()))}
        report["ops"][op]["B4"] = {"applicable": False, "reason": "MEHF-LAP was not trained under the clean protocol (no clean_lap output, no LAP_DECISION.txt); B4 is conditional on LAP being trained",
                                   "cross_protocol_leaky_LAP_frozen_median_spread": leaky_blocks["LAP"]["frozen_median_spread_on_clean_common"]}

    # ---- headline per Amendment 3 A3.3
    tp, ts = primary_tiers["primary"], primary_tiers["secondary"]
    if tp == "N" or ts == "N":
        headline = "N"
    else:
        headline = tp if TIER_ORDER[ts] >= TIER_ORDER[tp] else ts
    blocks = {op: {"B1": report["ops"][op]["B1"]["fires"], "B2": report["ops"][op]["B2"]["fires"],
                   "B3_projection_ON": report["ops"][op]["B3"]["fires_projection_ON_variants_only_(section6_wording)"],
                   "B3_any_variant": report["ops"][op]["B3"]["fires_any_variant_(amendment5_wording)"], "B4": None} for op in primary_tiers}
    max_ds = max(abs(frozen["SECONDARY_dev_folds"][a][f"fold{f}"]["s_star"] - frozen["PRIMARY_fitting_folds"][a][f"fold{f}"]["s_star"])
                 for a in ("ISO", "HARM") for f in (1, 2, 3))
    report["decision"] = {"tier_primary": tp, "tier_secondary": ts, "headline_tier_after_A3.3": headline,
                          "operating_point_sensitive": bool(tp != ts), "max_abs_sstar_dev_minus_fit": float(max_ds),
                          "blocking_conditions": blocks,
                          "note": "blocking conditions downgrade/block the headline but never create a tier; section 6 forbids inventing tiers"}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["decision"], indent=2))
    print(json.dumps({op: report["ops"][op]["primary_contrast_HARM_minus_ISO"] for op in report["ops"]}, indent=2))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    main()
