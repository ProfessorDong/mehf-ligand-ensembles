"""Task A of the clean protocol: derive the systems whose frame-0 pocket crop cannot be
represented by the stored candidate pool, and write the post-exclusion split counts.

The store keeps protein heavy atoms within R=8 A of the ligand at ANY stored frame, capped at
600 atoms (closest by all-frame minimum distance). For a system let d0_j be the frame-0 distance
of pool atom j to the nearest ligand HEAVY atom (exactly `pocket_ranking_distance(.., 'frame0')`)
and D96 the 96th smallest d0 over the pool. Any protein atom outside the pool has
d0 >= m_j > B, where B = 8 if the pool is untruncated, else B = min(8, max_pool all-frame min
distance) (outside atoms had all-frame min distance >= the largest kept one). Hence if D96 <= B the
frame-0 top-96 of the pool IS the frame-0 top-96 of the whole protein. A system is excluded iff
D96 > B (or the pool has fewer than 96 atoms).
Reads store npz + pairs.csv only. Does not read any coordinates of a test-split system for any
purpose other than this geometric check (the check is split-blind, by system), and builds no model input.
"""
import json, sys, os
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); os.chdir(ROOT)
from mdwm.data.mmp_transfer import (DEFAULT_STORE, load_usable_pairs, target_disjoint_split,
                                    directional_examples, load_mappings, DEFAULT_MAPPINGS)

OUT = ROOT / "outputs/clean_protocol_v1"
RADIUS, CROP = 8.0, 96

def measure(pdb):
    with np.load(Path(DEFAULT_STORE) / f"{pdb}.npz") as z:
        lig0 = z["ligand_coords"][0].astype(np.float64)
        heavy = z["ligand_atomic_numbers"] > 1
        pk0 = z["pocket_coords"][0].astype(np.float64)
        m = z["pocket_min_distance_over_frames"].astype(np.float64)
    d0 = np.sqrt(((pk0[:, None] - lig0[heavy][None]) ** 2).sum(-1)).min(1)
    n = len(d0)
    srt = np.sort(d0)
    d96 = float(srt[CROP - 1]) if n >= CROP else float("inf")
    truncated = n >= 600  # cap
    bound = min(RADIUS, float(m.max())) if truncated else RADIUS
    return dict(pdb=pdb, n_pool=n, d96=d96, truncated_at_cap=bool(truncated), bound=bound,
                n_d0_le_8=int((d0 <= RADIUS).sum()), min_over_frames_max=float(m.max()))

def main():
    raw = pd.read_csv(ROOT / "outputs/mmp_transfer_corpus_v1/pairs.csv")
    for c in ("pdb_a", "pdb_b"): raw[c] = raw[c].astype(str).str.upper()
    ok = raw[raw.status == "ok"]
    systems = sorted(set(ok.pdb_a) | set(ok.pdb_b))
    rows = [measure(s) for s in systems]
    df = pd.DataFrame(rows)
    df["excluded"] = df.d96 > df.bound
    exc = sorted(df[df.excluded].pdb)
    prev = "1O0F 1RGK 1Z6D 1ZZ1 2A3B 2EA2 2Q93 2YBS 4IKS 4JKW 4JX9 4U54 4YXO 5EFC 5FLS 5U2F 5W7U 5X4Q 6HSZ 6MO9".split()
    # simple criterion (user's): 96th smallest frame-0 distance > 8, ignoring cap
    simple = sorted(df[df.d96 > RADIUS].pdb)
    print(f"systems measured {len(df)}; excluded (full criterion) {len(exc)}; simple d96>8 {len(simple)}")
    print("matches user's list:", exc == sorted(prev), "| simple matches:", simple == sorted(prev))
    print("only mine:", sorted(set(exc) - set(prev)), "only theirs:", sorted(set(prev) - set(exc)))
    print("truncated at cap:", int(df.truncated_at_cap.sum()), " pool<96:", int((df.n_pool < CROP).sum()))
    print("d96 quantiles: median %.3f p99 %.3f max %.3f" % (df.d96.median(), df.d96.quantile(.99), df.d96.replace(np.inf, np.nan).max()))
    near = df[(df.d96 - df.bound).abs() < 0.05]
    print("within 0.05 A of bound:", near[["pdb", "d96", "bound"]].values.tolist())
    excset = set(exc)

    # ---- split counts -------------------------------------------------------
    usable = target_disjoint_split(load_usable_pairs())
    maps = load_mappings(DEFAULT_MAPPINGS)
    def table(pairs):
        out = {}
        for sp in ("train", "val", "test"):
            p = pairs[pairs.primary_split == sp]
            ex = directional_examples(p)
            ex = ex[ex.pair_key.isin(maps.keys())]
            out[sp] = dict(targets=int(p.uniprot.nunique()), pairs=int(len(p)),
                           pairs_with_mapping=int(ex.pair_key.nunique()), directed_examples=int(len(ex)),
                           directed_targets=int(ex.uniprot.nunique()))
        out["total"] = {k: sum(out[s][k] for s in ("train", "val", "test")) for k in out["train"]}
        return out
    before = table(usable)
    hit = usable.pdb_a.isin(excset) | usable.pdb_b.isin(excset)
    after = table(usable[~hit])
    # alternative: drop only the directed example whose PRODUCT is excluded
    ex_all = directional_examples(usable); ex_all = ex_all[ex_all.pair_key.isin(maps.keys())]
    n_prod = int(ex_all[ex_all.primary_split.isin(["train","val","test"])]["product"].isin(excset).sum())
    # tau-targets: do any targets disappear entirely?
    lost_targets = sorted(set(usable.uniprot) - set(usable[~hit].uniprot))
    per_split_removed = {sp: int((hit & (usable.primary_split == sp)).sum()) for sp in ("train", "val", "test")}
    ok_hit = ok.pdb_a.isin(excset) | ok.pdb_b.isin(excset)
    arithmetic = dict(
        ok_rows=int(len(ok)), ok_rows_dropped=int(ok_hit.sum()),
        usable_pairs_before=int(len(usable)), usable_pairs_dropped=int(hit.sum()),
        usable_pairs_dropped_by_split=per_split_removed, usable_pairs_after=int((~hit).sum()),
        directed_examples_dropped_if_only_excluded_product_removed=n_prod,
        targets_lost_entirely=lost_targets)
    alt = set(prev)
    alt_hit = usable.pdb_a.isin(alt) | usable.pdb_b.isin(alt)
    alternative = dict(
        note="same arithmetic under the 20-system list supplied by the requester (all-ligand-atom distances incl. H); NOT used for training",
        ok_rows_dropped=int((ok.pdb_a.isin(alt) | ok.pdb_b.isin(alt)).sum()),
        usable_pairs_dropped=int(alt_hit.sum()), after_exclusion=table(usable[~alt_hit]))
    (OUT / "excluded_systems.json").write_text(json.dumps(dict(
        excluded=exc, n_excluded=len(exc), matches_user_list=exc == sorted(prev),
        extra_vs_requester_list=sorted(set(exc) - set(prev)), missing_vs_requester_list=sorted(set(prev) - set(exc)),
        why_different="requester's 20 correspond to 96th-nearest frame-0 distance measured to ALL ligand atoms incl. H; the loader's frame0 ranking (pocket_ranking_distance) uses ligand HEAVY atoms only (>= all-atom distance), and the store pool is built from all-atom distances, so the provable-identity argument must use the heavy-atom distance",
        derivation=__doc__, criterion=f"D96 > B; crop={CROP}, radius={RADIUS}, store cap=600",
        n_systems_measured=len(df), population="all pdb_a/pdb_b of status=='ok' rows of outputs/mmp_transfer_corpus_v1/pairs.csv",
        d96_median=float(df.d96.median()), d96_p99=float(df.d96.quantile(.99)),
        per_excluded_system=df[df.excluded].drop(columns="excluded").to_dict("records"),
        n_truncated_at_cap_in_population=int(df.truncated_at_cap.sum()),
        rule="a pair is dropped entirely (both directions) if either pdb is excluded; applied AFTER the split and fold assignment (target->split and target->fold maps are those of the pre-exclusion corpus)",
    ), indent=1))
    (OUT / "split_counts_clean.json").write_text(json.dumps(dict(
        before_exclusion=before, after_exclusion=after, arithmetic=arithmetic, alternative_requester_list_of_20=alternative,
        note="usable pairs = status ok & transformation_agrees & split_a==split_b (4,913); directed = 2 x pairs with atom mapping; split = target_disjoint_split(seed 20260918) on the PRE-exclusion corpus, then rows dropped"), indent=1))
    print(json.dumps(dict(before=before, after=after, arithmetic=arithmetic), indent=1))

if __name__ == "__main__":
    main()
