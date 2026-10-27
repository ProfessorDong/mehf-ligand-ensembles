"""Apply the hashed matched-torsion rule of `scripts/matched_torsion_fit.py`, unchanged,
to the LAPLACIAN arm (outputs/laplacian_prior_v1/PREREGISTRATION.md).

The original script writes a fixed path that holds the confirmed arms' result, so it
is run against a temporary root in which outputs/laplacian_prior_v1 is linked in and
outputs/sigma_star_test_v1 is empty; its output is then moved to
outputs/laplacian_prior_v1/matched_torsion_fit.json. The rule code itself is imported,
not copied.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import matched_torsion_fit as rule  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp / "outputs" / "sigma_star_test_v1").mkdir(parents=True)
    os.symlink(ROOT / "outputs" / "laplacian_prior_v1", tmp / "outputs" / "laplacian_prior_v1")
    rule.ROOT = tmp
    rule.ARMS = {"LAPLACIAN": ("laplacian_prior_v1", "spectrum")}
    rule.main()
    shutil.move(str(tmp / "outputs" / "sigma_star_test_v1" / "matched_torsion_fit.json"),
                str(ROOT / "outputs" / "laplacian_prior_v1" / "matched_torsion_fit.json"))
print("written outputs/laplacian_prior_v1/matched_torsion_fit.json")
