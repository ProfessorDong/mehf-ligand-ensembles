"""Regenerate the fit/dev example lists of the clean protocol from the post-exclusion corpus.
Calls the training code's own build_folded_examples (so the lists are exactly what training sees)
and writes outputs/clean_protocol_v1/folds/f{k}/{fit,dev}_examples.csv. Train split only."""
import json, os, sys
from pathlib import Path
import pandas as pd, yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); os.chdir(ROOT)
from mdwm.training.mehf_transfer import build_folded_examples

cols = ["parent", "product", "uniprot", "direction", "pair_key", "fold"]
summary = {}
for k in (1, 2, 3):
    cfg = yaml.safe_load(open(f"configs/clean_f{k}_harm.yaml"))
    fit, dev, meta = build_folded_examples(cfg)
    d = ROOT / f"outputs/clean_protocol_v1/folds/f{k}"; d.mkdir(parents=True, exist_ok=True)
    fit[cols].to_csv(d / "fit_examples.csv", index=False); dev[cols].to_csv(d / "dev_examples.csv", index=False)
    old = Path(f"outputs/laplacian_prior_v1/full_f{k}")
    of, od = pd.read_csv(old / "fit_examples.csv"), pd.read_csv(old / "dev_examples.csv")
    # target->fold map must be unchanged; rows removed must be exactly the excluded pairs
    ex = set(json.load(open("outputs/clean_protocol_v1/excluded_systems.json"))["excluded"])
    old_all = pd.concat([of, od])
    assert (old_all.drop_duplicates("uniprot").set_index("uniprot").fold.reindex(pd.concat([fit, dev]).uniprot.unique()) ==
            pd.concat([fit, dev]).drop_duplicates("uniprot").set_index("uniprot").fold).all()
    kept = set(pd.concat([fit, dev])["parent"] + "->" + pd.concat([fit, dev])["product"])
    removed = old_all[~(old_all["parent"] + "->" + old_all["product"]).isin(kept)]
    assert all((a in ex) or (b in ex) for a, b in zip(removed['parent'], removed['product'])), "removed a non-excluded pair"
    summary[k] = {**meta, "old_fit": len(of), "old_dev": len(od), "removed": len(removed),
                  "fit_targets": int(fit.uniprot.nunique()), "dev_targets": int(dev.uniprot.nunique())}
    print(k, summary[k], flush=True)
(ROOT / "outputs/clean_protocol_v1/folds/summary.json").write_text(json.dumps(summary, indent=1))
