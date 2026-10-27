"""Apply the existing matched-torsion rule (scripts/matched_torsion_fit.py, imported unchanged)
to the clean-protocol grids and write the frozen operating points.

PRIMARY  : s* from the FITTING-fold grids (fit_examples.csv sample), the existing rule.
SECONDARY: s* from the same rule on the target-disjoint DEV-fold grids (dev_examples.csv sample).
Never touches val or test. The rule module hardcodes its read root and its output path, so it is run
against a temporary root in which outputs/clean_protocol_v1 is symlinked and outputs/sigma_star_test_v1
is empty (the technique of laplacian_matched_torsion.py); nothing under the real outputs/ is written
except the frozen JSON below.
"""
import datetime, hashlib, json, os, shutil, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import matched_torsion_fit as rule  # noqa: E402

OUT = ROOT / "outputs/clean_protocol_v1"
ARMS = {"LAP_fit": ("clean_protocol_v1", "grid/lap_fit"), "LAP_dev": ("clean_protocol_v1", "grid/lap_dev")}

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp / "outputs" / "sigma_star_test_v1").mkdir(parents=True)
    os.symlink(OUT, tmp / "outputs" / "clean_protocol_v1")
    rule.ROOT = tmp
    rule.ARMS = ARMS
    rule.GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]   # measured points only; rule code unchanged
    rule.main()
    report = json.loads((tmp / "outputs" / "sigma_star_test_v1" / "matched_torsion_fit.json").read_text())

def pick(arm):
    res = {}
    for fold in (1, 2, 3):
        e = report["arms"][arm][f"fold{fold}"]
        res[f"fold{fold}"] = {"matched": e["matched"], "s_star": e["matched_torsion_scale"],
                              "crossings": e["torsion_crossings_of_1.0"],
                              "spread_at_s_star": e.get("channel_spread_at_s_star"),
                              "torsion_ratio_curve": [r["dihedral"] for r in e["grid"]],
                              "n_examples_per_point": sorted({r["n_examples"] for r in e["grid"]})}
    return res

frozen = {
    "frozen_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "rule": "scripts/matched_torsion_fit.py, imported unchanged; sha256 " +
            hashlib.sha256((ROOT / "scripts/matched_torsion_fit.py").read_bytes()).hexdigest(),
    "grid": rule.GRID,
    "PRIMARY_fitting_folds": {"LAP": pick("LAP_fit")},
    "SECONDARY_dev_folds": {"LAP": pick("LAP_dev")},
    "test_read": False,
    "note": "LAP arm only; decided per LAP_DECISION.txt (after ISO/HARM test read); grid 1.0-5.0, no extension needed",
}
(OUT / "s_star_frozen_lap.json").write_text(json.dumps(frozen, indent=2))
(OUT / "matched_torsion_fit_full_lap.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: frozen[k] for k in ("PRIMARY_fitting_folds", "SECONDARY_dev_folds")}, indent=1))
