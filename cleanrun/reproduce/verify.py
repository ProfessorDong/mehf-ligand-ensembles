#!/usr/bin/env python
"""Compare a rerun in cleanrun/rerun/ against the recorded run in cleanrun/expected/.

Compares SUMMARY statistics, not individual rows. Per-example values are
draw-specific: the sampler seeds one generator per batch of four from the first
example's key in that batch, so any change to the list, its order or the batch
size changes almost every row. See README section 3.
"""
import json, os, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP, GOT = os.path.join(ROOT, "expected"), os.path.join(ROOT, "rerun")

TOL = {"spread": 0.02, "psi": 0.02, "ratio": 0.02, "contrast": 0.05}

def load(p):
    try:
        with open(p) as f: return json.load(f)
    except Exception: return None

def cmp_num(label, a, b, tol):
    if a is None or b is None:
        print(f"  SKIP {label}: missing"); return None
    ok = abs(a - b) <= tol
    print(f"  {'OK  ' if ok else 'DIFF'} {label}: expected {a:.4f}, got {b:.4f}, |d|={abs(a-b):.4f} (tol {tol})")
    return ok

def main():
    if not os.path.isdir(GOT):
        print(f"no rerun found at {GOT} — run the reproduce steps first"); return 1
    results = []

    # exclusions
    e, g = load(f"{EXP}/../inputs/excluded_systems.json"), load(f"{GOT}/excluded_systems.json")
    if e and g:
        same = sorted(e["excluded"]) == sorted(g["excluded"])
        print(f"\nexclusions: {'OK   identical' if same else 'DIFF'} ({len(g['excluded'])} systems)")
        results.append(same)

    # tau
    e, g = load(f"{EXP}/../inputs/tau_train_only_large.json"), load(f"{GOT}/tau_train_only_large.json")
    if e and g:
        print("\nwidths (tau):")
        for k in ("bond_angstrom", "angle_deg", "torsion_deg", "oop_deg"):
            ev, gv = e.get(k, {}).get("train_only"), g.get(k, {}).get("train_only")
            if isinstance(ev, (int, float)) and isinstance(gv, (int, float)):
                results.append(cmp_num(k, ev, gv, abs(ev) * 0.01))

    # test decision
    e, g = load(f"{EXP}/TEST_RESULT.json"), load(f"{GOT}/TEST_RESULT.json")
    if e and g:
        print("\ntest decision:")
        for op in ("primary", "secondary"):
            try:
                ea, ga = e["ops"][op], g["ops"][op]
            except Exception:
                continue
            results.append(cmp_num(f"{op} Sigma*-ISO mean diff",
                                   ea["tier"]["mean_difference"], ga["tier"]["mean_difference"], TOL["contrast"]))
            for arm in ("clean_ISO", "clean_HARM"):
                if arm in ea and arm in ga:
                    results.append(cmp_num(f"{op} {arm} median spread",
                                           ea[arm]["median_spread"], ga[arm]["median_spread"], TOL["spread"]))
            et, gt = ea["tier"]["tier"], ga["tier"]["tier"]
            ok = et == gt
            print(f"  {'OK  ' if ok else 'DIFF'} {op} tier: expected {et}, got {gt}")
            results.append(ok)

    done = [r for r in results if r is not None]
    print(f"\n{sum(done)}/{len(done)} checks passed")
    if not done:
        print("nothing compared — did the rerun produce output?"); return 1
    return 0 if all(done) else 2

if __name__ == "__main__":
    sys.exit(main())
