"""Pre-registered secondary: full-uncropped-pocket clash rate of every clean arm AT the frozen
primary s* (and the secondary dev s*), measured on the fitting sample and the dev sample.
Reads s_star_frozen.json; runs mehf_internal_geometry_spectrum.py with --full-pocket-clash at exactly s*.
Sequential, GPU 1 only. Skips (and records) an arm/fold with no matched s*."""
import json, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT)
OUT = ROOT / "outputs/clean_protocol_v1"
frozen = json.loads((OUT / "s_star_frozen.json").read_text())
env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", PYTHONPATH="src:scripts")
done = {}
for which, key in (("primary", "PRIMARY_fitting_folds"), ("secondary", "SECONDARY_dev_folds")):
    for arm in ("ISO", "HARM"):
        for fold in (1, 2, 3):
            e = frozen[key][arm][f"fold{fold}"]
            if not e["matched"]:
                done[f"{which}/{arm}/f{fold}"] = "UNMATCHED, not run"; continue
            s = e["s_star"]
            for sample in ("fit", "dev"):
                out = OUT / "clash_at_sstar" / f"{arm.lower()}_f{fold}_{which}s_{sample}.json"
                out.parent.mkdir(exist_ok=True)
                extra = ["--n-examples", "96"] if sample == "fit" else ["--n-examples", "160", "--use-dev"]
                cmd = [sys.executable, "scripts/mehf_internal_geometry_spectrum.py", "--checkpoint",
                       f"outputs/clean_protocol_v1/{arm.lower()}_f{fold}/checkpoint.pt", "--rigid", "0.32",
                       "--impropers", "--restore", "off", "--angle-target-width", "0.0", "--projection-iterations", "8",
                       "--projection-angle-relaxation", "0.5", "--angle-projection", "on", "--full-pocket-clash",
                       "--internal", repr(float(s)), "--output", str(out)] + extra
                subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)
                d = json.loads(out.read_text())
                done[f"{which}/{arm}/f{fold}/{sample}"] = {"s_star": s, "n": d["n_examples_sampled"],
                    "full": d["full_pocket_clash"], "crop": d["crop_pocket_clash"],
                    "torsion_ratio": d["dihedral"]["ratio_median"]}
(OUT / "clash_at_sstar_summary.json").write_text(json.dumps(done, indent=1))
print(json.dumps(done, indent=1))
