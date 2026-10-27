"""Compare regenerated results against the frozen reference record.

    python reproduce/verify.py                      # outputs/ vs reference_outputs/
    python reproduce/verify.py --reproduced X --reference Y [--rtol 1e-5]

Every .json and .csv under the reference tree is looked up at the same relative
path under the reproduced tree and classified:

  IDENTICAL        byte-for-byte equal
  WITHIN_TOL       every number equal within --atol + --rtol * |value| and every string
                   equal, except strings that are absolute filesystem paths (these
                   record where a run happened, not a result) and fields that the
                   reference, written by an older script version, does not have
  DIFFERENT        anything else; the first differences are printed
  NOT_REGENERATED  no counterpart was produced

Files listed in HISTORICAL are superseded records kept for transparency (voided
runs, retracted first measurements, and outputs of earlier script versions);
they are reported separately and do not count as failures. Exit status is 1 if
any non-historical file is DIFFERENT.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ATOL = 1e-6

HISTORICAL = {
    # Voided: Gauss-Seidel prior without under-relaxation diverged (see FINDING).
    "aniso_prior_v1/*": "voided run (diverging sampler), kept as the record of the bug",
    # Retracted first measurement (SHAKE-frozen X-H bonds inflated the bond ratio to 25.99).
    "isotropic_noise_cost_v1/summary.json": "retracted first measurement; summary_heavy.json supersedes it",
    "*smoke*.json": "smoke test",
    "*/smoke.per_example.csv": "smoke test",
    # 200-molecule corrector gates: produced by an earlier version of
    # geometry_corrector_gate.py (27 vs 26 fitted angle classes). Superseded by
    # corrector_gate_v1/scaleup/ (2,000 molecules x 5 seeds), which reproduces exactly.
    "corrector_gate_v1/gate*.json": "earlier script version; superseded by scaleup/",
    "corrector_gate_v1/holdout_*.json": "earlier script version; superseded by scaleup/",
    "corrector_gate_v1/with_r5.json": "earlier script version; superseded by scaleup/",
    "corrector_gate_v1/debug.json": "debug output",
    # Soft-target cell G, folds 1-2: the two earliest soft-target runs, consistent
    # with predating the switch of the soft-target draw to the seeded sampler
    # generator (see mehf.py, _project_geometry). Fold 3 and every later run
    # reproduce exactly. Regenerated values move the cell-G mean angle ratio
    # 1.699 -> 1.703; the section-2 effects use cells A-D only.
    "angle_term_v1/spectrum/G_f1.*": "earlier code version (unseeded soft-target draw)",
    "angle_term_v1/spectrum/G_f2.*": "earlier code version (unseeded soft-target draw)",
    # Not cited by any finding.
    "wilson_b_theory_v1/projector_*.json": "not part of the paper",
    # Stage-1 hyper-parameter search (source of rigid 0.35 / internal 3.5); provenance only.
    "subspace_prior_stage1/*": "hyper-parameter provenance, not regenerated",
    # Stage-1 scale-calibration probes of internal_calibration_v1 (fitting folds only;
    # labelled EXPLORATORY in the files; the example-sampling mode is not recorded,
    # and probe.json is from the retracted head-of-file sampling). They are the
    # provenance of rigid = 0.32, not reported results.
    "internal_calibration_v1/bal_*": "stage-1 exploratory calibration probe",
    "internal_calibration_v1/probe*": "stage-1 exploratory calibration probe (retracted sampling)",
    "internal_calibration_v1/spread_f*": "stage-1 exploratory calibration probe",
    "internal_calibration_v1/js_dev_*/*": "earlier Jensen-Shannon evaluation; metric abandoned",
    # Superseded prior version; ANISO v3 is the evaluated anisotropic arm.
    "aniso_prior_v2/spectrum_f*": "superseded prior version (v3 is evaluated)",
    "aniso_prior_v2/full_f[12]/*": "superseded prior version (v3 is evaluated)",
    "denovo_v1/*": "theory note for future work, not regenerated",
}


def historical(rel: str) -> str | None:
    for pattern, why in HISTORICAL.items():
        if fnmatch.fnmatch(rel, pattern):
            return why
    return None


def is_path(value) -> bool:
    return isinstance(value, str) and value.startswith("/")


def compare_json(a, b, rtol, where="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in b:
                out.append(f"{where}/{k}: missing from the regenerated file")
            elif k not in a:
                # Newer script versions record extra configuration fields; a field
                # the older reference never had is a schema addition, not a result.
                continue
            else:
                compare_json(a[k], b[k], rtol, f"{where}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{where}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                compare_json(x, y, rtol, f"{where}[{i}]", out)
    elif isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            out.append(f"{where}: {a} vs {b}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not (a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))
                or abs(a - b) <= ATOL + rtol * max(abs(a), abs(b))):
            out.append(f"{where}: {a!r} vs {b!r}")
    elif is_path(a) and is_path(b):
        pass
    elif a != b:
        out.append(f"{where}: {str(a)[:60]!r} vs {str(b)[:60]!r}")
    return out


def compare_csv(pa: Path, pb: Path, rtol):
    a, b = pd.read_csv(pa, low_memory=False), pd.read_csv(pb, low_memory=False)
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return [f"shape/columns {a.shape} vs {b.shape}"]
    problems = []
    for c in a.columns:
        x, y = a[c], b[c]
        if x.dtype.kind in "fi" and y.dtype.kind in "fi":
            both_nan = x.isna() & y.isna()
            scale = pd.concat([x.abs(), y.abs()], axis=1).max(axis=1)
            bad = ~both_nan & ~((x - y).abs() <= ATOL + rtol * scale)
            if bad.any():
                i = bad.idxmax()
                problems.append(f"column {c}: {int(bad.sum())} rows differ, e.g. row {i}: {x[i]!r} vs {y[i]!r}")
        elif not (x.astype(str) == y.astype(str)).all():
            if not all(is_path(v) for v in x.dropna()):
                problems.append(f"column {c}: text differs")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reproduced", type=Path, default=ROOT / "outputs")
    parser.add_argument("--reference", type=Path, default=ROOT / "reference_outputs")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6,
                        help="absolute tolerance, for values near zero (as numpy.isclose)")
    parser.add_argument("--only", default="*", help="glob on the relative path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    global ATOL
    ATOL = args.atol

    tally = {"IDENTICAL": [], "WITHIN_TOL": [], "DIFFERENT": [], "NOT_REGENERATED": []}
    hist = {k: [] for k in tally}
    for ref in sorted(args.reference.rglob("*")):
        if ref.suffix not in (".json", ".csv") or not ref.is_file():
            continue
        rel = ref.relative_to(args.reference).as_posix()
        if not fnmatch.fnmatch(rel, args.only):
            continue
        rep = args.reproduced / rel
        if not rep.exists():
            status, detail = "NOT_REGENERATED", []
        elif ref.read_bytes() == rep.read_bytes():
            status, detail = "IDENTICAL", []
        else:
            try:
                detail = (compare_json(json.loads(ref.read_text()), json.loads(rep.read_text()), args.rtol)
                          if ref.suffix == ".json" else compare_csv(ref, rep, args.rtol))
            except Exception as exc:  # noqa: BLE001
                detail = [f"could not compare: {exc}"]
            status = "WITHIN_TOL" if not detail else "DIFFERENT"
        (hist if historical(rel) else tally)[status].append((rel, detail))

    print(f"reference:  {args.reference}\nreproduced: {args.reproduced}\nrtol: {args.rtol}  atol: {args.atol}\n")
    for status, rows in tally.items():
        print(f"{status:<16} {len(rows)}")
        if status == "DIFFERENT" or (status != "NOT_REGENERATED" and not args.quiet and status == "WITHIN_TOL" and False):
            for rel, detail in rows:
                print(f"   {rel}")
                for d in detail[:5]:
                    print(f"      {d}")
    if tally["NOT_REGENERATED"] and not args.quiet:
        print("\nnot regenerated:")
        for rel, _ in tally["NOT_REGENERATED"]:
            print(f"   {rel}")
    print("\nhistorical / superseded (not counted):",
          ", ".join(f"{k} {len(v)}" for k, v in hist.items() if v))
    return 1 if tally["DIFFERENT"] else 0


if __name__ == "__main__":
    sys.exit(main())
