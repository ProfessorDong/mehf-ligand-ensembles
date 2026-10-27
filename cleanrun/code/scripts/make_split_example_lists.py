"""Write the frozen val and test example lists used by the confirmatory reads.

Both lists are a pure function of the corpus (`outputs/mmp_transfer_corpus_v1`)
and the target-disjoint split in `mdwm.data.mmp_transfer.target_disjoint_split`
(seed 20260918): train 3,489 pairs / 345 UniProt targets, val 696 / 106,
test 728 / 66, each pair expanded into both directions.

    outputs/pareto_v1/val_examples.csv          1,392 examples (read once, 2026-09-23)
    outputs/sigma_star_test_v1/test_examples.csv 1,456 examples (read once, 2026-09-25)
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdwm.data.mmp_transfer import usable_examples  # noqa: E402

TARGETS = {
    "val": ROOT / "outputs/pareto_v1/val_examples.csv",
    "test": ROOT / "outputs/sigma_star_test_v1/test_examples.csv",
}

if __name__ == "__main__":
    import os
    os.chdir(ROOT)
    for split, path in TARGETS.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = usable_examples(split=split)
        frame.to_csv(path, index=False)
        print(f"{split}: {len(frame)} examples, {frame['uniprot'].nunique()} targets -> {path.relative_to(ROOT)}")
