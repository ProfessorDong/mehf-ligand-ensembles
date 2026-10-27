"""Collect TargetDiff per-pocket samples into one multi-molecule SDF per pocket,
named so that `geometry_corrector_gate.py` and `published_model_isotropy_test.py`
parse the pocket code the same way they do for the DiffSBDD release
(`path.name.split('-')[0]`).

Nothing is filtered here: every molecule TargetDiff's own reconstruction wrote is
copied through, so the downstream scripts see the RAW output.
"""
import json
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/external/targetdiff/samples"
DEST = ROOT / "data/external/targetdiff/samples_sdf"
DEST.mkdir(parents=True, exist_ok=True)

counts = Counter()
per_pocket = {}
for directory in sorted(SRC.iterdir()):
    if not (directory / "sample.pt").exists():
        counts["incomplete"] += 1
        continue
    sdf_dir = directory / "sdf"
    if not sdf_dir.is_dir():
        counts["no_sdf_dir"] += 1
        continue
    files = sorted(sdf_dir.glob("*.sdf"))
    if not files:
        counts["empty"] += 1
        continue
    pocket = directory.name.split("_")[0]
    out = DEST / f"{pocket}-{directory.name}-targetdiff.sdf"
    writer = Chem.SDWriter(str(out))
    written = 0
    for f in files:
        for mol in Chem.SDMolSupplier(str(f), removeHs=True, sanitize=False):
            if mol is None:
                counts["unreadable"] += 1
                continue
            writer.write(mol)
            written += 1
    writer.close()
    per_pocket[directory.name] = written
    counts["molecules"] += written
    counts["pockets"] += 1

json.dump({"counts": counts, "per_pocket": per_pocket},
          open(DEST / "collect_index.json", "w"), indent=2)
print(dict(counts))
vals = sorted(per_pocket.values())
if vals:
    print("per pocket min/median/max:", vals[0], vals[len(vals) // 2], vals[-1])
