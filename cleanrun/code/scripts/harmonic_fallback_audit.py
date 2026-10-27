"""Measure (do not change) how often `MappedEditHierarchicalFlow._harmonic_local_noise` silently falls
back to ISOTROPIC noise during the clean HARM runs.

Branches of that function that leave a molecule isotropic:
  (a) condition.bond_index is None                   -> whole batch (cannot happen: the collate always sets it)
  (b) live heavy atoms < 4                           -> `continue`
  (c) sigma_star_factor(...) returns None            -> `continue` (no bonds/angles, no internal coordinates,
                                                         or NO eigenvalue above 1e-6 * max survives)
Also recorded: rigid-mode count n_rigid of every successful factor (6 expected for a non-linear molecule) --
a molecule whose F is rank-deficient beyond that would pass (c) while still being badly shaped.
The reference passed is exactly what training passes: the dataset item's float32 `product_init`, cast to float64,
with the item's own bond_index. Molecules = distinct (bond_index, product_init rounded to 3 decimals), the
model's own cache key. Sets = the post-exclusion fit and dev lists of each clean fold (the HARM runs' train/dev data),
evaluated under the tau set the runs used (train_only_large) and, for comparison, legacy.
"""
import json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); os.chdir(ROOT)
from mdwm.data.mmp_transfer import MMPTransferDataset  # noqa: E402
from mdwm.models.harmonic_prior import sigma_star_factor, topology  # noqa: E402

OUT = ROOT / "outputs/clean_protocol_v1"

def item_for(row):
    ds = MMPTransferDataset(pd.DataFrame([row]), product_init_mode="product_frame0", n_frames=50,
                            max_pocket_atoms=96, pocket_selection="frame0", frame_alignment="pocket")
    return ds[0]

def probe(args):
    product, row = args
    torch.set_num_threads(1)
    it = item_for(row)
    ref = torch.from_numpy(it["product_init"]); bonds = torch.from_numpy(it["bond_index"])
    live = int(ref.shape[0]); out = {"product": product, "live": live}
    if live < 4:
        out["branch"] = "b_live_lt_4"; return out
    key = (bonds.numpy().tobytes(), ref.to(torch.float32).numpy().round(3).tobytes())
    out["key"] = hash(key)
    for tau in ("train_only_large", "legacy"):
        with torch.enable_grad():
            factor, n_rigid = sigma_star_factor(ref.double(), bonds, tau_set=tau)
        if factor is None:
            pairs, angles, _, _ = topology(bonds, live)
            out[tau] = {"branch": "c_factor_None", "reason": "no_pairs_or_angles" if (not pairs or not angles) else "no_eigenvalue_above_cutoff_or_no_rows"}
        else:
            out[tau] = {"branch": "ok", "n_rigid": int(n_rigid), "n_modes": int(factor.shape[1])}
    return out

def main():
    sets = {}
    for k in (1, 2, 3):
        for sp in ("fit", "dev"):
            sets[(k, sp)] = pd.read_csv(OUT / f"folds/f{k}/{sp}_examples.csv")
    allrows = pd.concat(sets.values()).drop_duplicates("product")
    jobs = [(r.product, r._asdict()) for r in allrows.itertuples(index=False)]
    jobs = [(p, {k: v for k, v in d.items()}) for p, d in jobs]
    with Pool(16) as pool:
        res = pool.map(probe, jobs, chunksize=4)
    by = {r["product"]: r for r in res}
    report = {"n_distinct_products_probed": len(by), "folds": {}}
    for tau in ("train_only_large", "legacy"):
        tau_rep = {}
        for (k, sp), df in sets.items():
            prods = df["product"].astype(str)
            def status(p):
                r = by[p]
                return "b_live_lt_4" if r.get("branch") == "b_live_lt_4" else r[tau]["branch"]
            st = prods.map(status)
            distinct = pd.Series(prods.unique()).map(status)
            nr = [by[p][tau]["n_rigid"] for p in prods.unique() if by[p].get("branch") != "b_live_lt_4" and by[p][tau]["branch"] == "ok"]
            tau_rep[f"fold{k}_{sp}"] = {
                "n_examples": int(len(df)), "n_distinct_molecules": int(len(distinct)),
                "distinct_by_branch": distinct.value_counts().to_dict(),
                "examples_by_branch": st.value_counts().to_dict(),
                "fraction_examples_isotropic_fallback": float((st != "ok").mean()),
                "n_rigid_counts": pd.Series(nr).value_counts().sort_index().to_dict()}
        report["folds"][tau] = tau_rep
    report["note"] = "measurement only; the fallback code is unchanged and the clean HARM runs were trained as-is"
    (OUT / "harmonic_fallback_audit.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))

if __name__ == "__main__":
    main()
