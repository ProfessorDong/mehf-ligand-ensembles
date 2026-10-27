"""P7 / section 5.2.6 crop diagnostics, no model.  For every post-exclusion system of the corpus (status == ok rows of
pairs.csv, minus the 28 excluded) with more stored pocket atoms than the 96-atom cap:
  overlap      = |crop0 & crop_min| / 96   (crop0 = frame-0 heavy-atom ranking, crop_min = the store's all-frame minimum
                 distance ranking, i.e. the legacy crop; both exactly as mmp_transfer.pocket_ranking_distance)
  recall_pair  = fraction of (pocket heavy atom, ligand heavy atom) pairs that come below 4.0 A at ANY of the stored frames
                 whose pocket atom lies inside crop0
  recall_atom  = same at the level of pocket atoms (a pocket atom is a contact atom if any ligand heavy atom is < 4 A at any frame)
Reported on the whole corpus and on the systems that are products of the clean test list.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mdwm.data.mmp_transfer import DEFAULT_STORE, load_system, pocket_ranking_distance  # noqa: E402

CAP, CONTACT = 96, 4.0
ex = set(json.load(open(ROOT / "outputs/clean_protocol_v1/excluded_systems.json"))["excluded"])
pairs = pd.read_csv(ROOT / "outputs/mmp_transfer_corpus_v1/pairs.csv")
pairs = pairs[pairs["status"] == "ok"]
ids = sorted((set(pairs.pdb_a) | set(pairs.pdb_b)) - ex)
test = pd.read_csv(ROOT / "outputs/clean_protocol_v1/test_examples_clean.csv")
test_products = set(test["product"]) | set(test["parent"])
rows = []
for i, pdb in enumerate(ids):
    s = load_system(pdb, DEFAULT_STORE, "pocket")
    pocket = np.asarray(s["pocket_coords"], dtype=np.float64)          # (T, P, 3)
    lig = np.asarray(s["coords"], dtype=np.float64)                    # (T, N, 3) heavy
    P = pocket.shape[1]
    if P <= CAP:
        rows.append(dict(pdb=pdb, n_pool=P, capped=False, in_test=pdb in test_products)); continue
    d = np.sqrt(((pocket[:, :, None, :] - lig[:, None, :, :]) ** 2).sum(-1))   # (T, P, N)
    crop0 = np.argsort(pocket_ranking_distance(s, "frame0"), kind="stable")[:CAP]
    cropm = np.argsort(pocket_ranking_distance(s, "min_over_frames"), kind="stable")[:CAP]
    in0 = np.zeros(P, bool); in0[crop0] = True
    contact_pair = (d < CONTACT).any(axis=0)                           # (P, N)
    contact_atom = contact_pair.any(axis=1)
    rows.append(dict(pdb=pdb, n_pool=P, capped=True, in_test=pdb in test_products,
                     overlap=len(set(crop0) & set(cropm)) / CAP,
                     n_contact_pairs=int(contact_pair.sum()), n_contact_atoms=int(contact_atom.sum()),
                     recall_pair=float(contact_pair[in0].sum() / contact_pair.sum()) if contact_pair.any() else np.nan,
                     recall_atom=float(contact_atom[in0].sum() / contact_atom.sum()) if contact_atom.any() else np.nan))
df = pd.DataFrame(rows)
out_dir = ROOT / "outputs/clean_protocol_v1/test_read"; out_dir.mkdir(exist_ok=True, parents=True)
df.to_csv(out_dir / "crop_recall_per_system.csv", index=False)
def summ(g):
    c = g[g.capped]
    r = {"n_systems": int(len(g)), "n_capped": int(len(c))}
    for k in ("overlap", "recall_pair", "recall_atom"):
        v = c[k].dropna()
        r[k] = {"n": int(len(v)), "mean": float(v.mean()), "median": float(v.median()), "min": float(v.min()),
                "frac_below_0.90": float((v < 0.90).mean())}
    return r
res = {"definition": __doc__, "all_corpus_post_exclusion": summ(df), "test_list_systems": summ(df[df.in_test]),
       "P7_recall_pair_corpus_mean": float(df[df.capped].recall_pair.mean())}
(out_dir / "crop_recall.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1)[:2500])
