"""Supplementary tables and figures from stored records (post-hoc, descriptive). See PLAN.md.

Run: python -I make_supplement.py <MEHF root> <MEHF_TCBB_Liang dir>
Writes supplement_tikz/*.tex and analysis/supplement_v1/VALUES.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]).resolve()
PAPER = Path(sys.argv[2]).resolve()
REF = ROOT / "reference_outputs"
OUT = PAPER / "supplement_tikz"
OUT.mkdir(exist_ok=True)
AN = PAPER / "analysis"
V = {}

CH4 = ["bond_length", "bond_angle", "dihedral", "planar_improper"]
CH5 = CH4 + ["internal_rmsf"]
CHNAME = {"bond_length": "Bond length", "bond_angle": "Bond angle", "dihedral": "Torsion",
          "planar_improper": "Out-of-plane", "internal_rmsf": "Internal RMSF"}
ARMS = ["ISO", "GS", "LAP", "HARM"]
FILES = {"ISO": REF / "sigma_star_test_v1/test_ISO_f{f}.per_example.csv",
         "GS": REF / "sigma_star_test_v1/test_ANISO_v3_f{f}.per_example.csv",
         "LAP": REF / "laplacian_prior_v1/test_LAPLACIAN_f{f}.per_example.csv",
         "HARM": REF / "sigma_star_test_v1/test_SIGMA_star_f{f}.per_example.csv"}
COLOR = {"ISO": "cISO", "GS": "cGS", "LAP": "cLAP", "HARM": "cHARMd"}
MARK = {"ISO": "*", "GS": "square*", "LAP": "triangle*", "HARM": "diamond*"}


def fmt(x, d=3):
    return f"{x:.{d}f}".replace("-", "$-$") if x < 0 else f"{x:.{d}f}"


# ------------------------------------------------------------------ per-example records
fold = {a: {f: pd.read_csv(str(FILES[a]).format(f=f)).set_index("key") for f in (1, 2, 3)} for a in ARMS}
ok = None
for a in ARMS:
    for f in (1, 2, 3):
        good = fold[a][f][[c + "_ratio" for c in CH4]].notna().all(axis=1)
        idx = set(good[good].index)
        ok = idx if ok is None else ok & idx
ok = sorted(ok)
assert len(ok) == 1410, len(ok)


def per_fold_S(a, f):
    r = fold[a][f].loc[ok, [c + "_ratio" for c in CH4]]
    return r.max(axis=1) - r.min(axis=1)


def avg(a):
    """Fold-averaged per-example S and E4 (the paper's aggregation)."""
    S = sum(per_fold_S(a, f) for f in (1, 2, 3)) / 3
    E4 = sum(np.log(fold[a][f].loc[ok, [c + "_ratio" for c in CH4]]).abs().mean(axis=1) for f in (1, 2, 3)) / 3
    return S, E4


# ------------------------------------------------------------------ Table S2: per-fold test medians
rows, chk = [], {}
for a in ARMS:
    for f in (1, 2, 3):
        x = fold[a][f].loc[ok]
        med = {c: float(x[c + "_ratio"].median()) for c in CH5}
        med["S"] = float(per_fold_S(a, f).median())
        rows.append((a, f, med))
    chk[a] = float(avg(a)[0].median())
V["table_S2_per_fold"] = [{"arm": a, "fold": f, **m} for a, f, m in rows]
V["check_fold_averaged_median_S"] = chk
for a, exp in (("HARM", [0.740, 0.758, 0.804]), ("ISO", [2.008, 1.851, 2.032])):
    got = [round(m["S"], 3) for b, f, m in rows if b == a]
    assert got == exp, (a, got)
assert round(chk["HARM"], 3) == 0.770 and round(chk["ISO"], 3) == 1.974, chk
LAB = {"ISO": "MEHF-ISO", "GS": "MEHF-GS", "LAP": "MEHF-LAP", "HARM": "MEHF-HARM"}
t = [r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
     r"Prior & Fold & Bond & Angle & Torsion & Out-of-plane & Int.\ RMSF & Spread \\", r"\midrule"]
for i, (a, f, m) in enumerate(rows):
    lab = LAB[a] if f == 1 else ""
    t.append(f"{lab} & {f} & " + " & ".join(f"{m[c]:.2f}" for c in CH5) + f" & {m['S']:.3f} \\\\")
    if f == 3 and i < len(rows) - 1:
        t.append(r"\addlinespace[2pt]")
t += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_perfold.tex").write_text("\n".join(t) + "\n")

# ------------------------------------------------------------------ Fig. S1: operating-point calibration
fits = json.load(open(REF / "sigma_star_test_v1/matched_torsion_fit.json"))["arms"]
fits.update(json.load(open(REF / "laplacian_prior_v1/matched_torsion_fit.json"))["arms"])
KEY = {"ISO": "ISO", "GS": "ANISO_v3", "LAP": "LAPLACIAN", "HARM": "SIGMA_star"}
curves, sstar = {}, {}
for a in ARMS:
    arm = fits[KEY[a]]
    grids = [{g["scale"]: g for g in arm[f"fold{f}"]["grid"]} for f in (1, 2, 3)]
    scales = sorted(set.intersection(*[set(g) for g in grids]))
    curves[a] = {c: [(s, float(np.mean([g[s][c] for g in grids]))) for s in scales] for c in CH5}
    sstar[a] = [arm[f"fold{f}"].get("matched_torsion_scale") for f in (1, 2, 3)]
V["fig_S1_curves"] = curves
V["fig_S1_sstar"] = sstar
g = [r"\begin{tikzpicture}",
     r"\begin{groupplot}[group style={group size=5 by 1, horizontal sep=0.75cm}, mehfaxis,"
     r" width=0.235\textwidth, height=3.6cm, xmin=0.8, xmax=4.2, xtick={1,2,3,4}, ymin=0,"
     r" xlabel={$s_{\mathrm{i}}$}]"]
for k, c in enumerate(CH5):
    opts = f"title={{{CHNAME[c]}}}" + (", ylabel={Fitting-fold width ratio}" if k == 0 else "")
    if k == 4:
        opts += ", legend to name=legcal, legend columns=4, legend style={/tikz/every even column/.append style={column sep=8pt}}"
    g.append(rf"\nextgroupplot[{opts}]")
    for a in ARMS:
        co = " ".join(f"({s:.2f},{v:.6f})" for s, v in curves[a][c])
        g.append(rf"\addplot[{COLOR[a]}, line width=0.7pt, mark={MARK[a]}, mark size=1.1pt] coordinates {{{co}}};")
        if k == 4:
            g.append(rf"\addlegendentry{{{LAB[a]}}}")
    g.append(r"\draw[refline] (axis cs:0.8,1) -- (axis cs:4.2,1);")
    if c == "dihedral":
        for a in ARMS:
            for s in sstar[a]:
                if s is not None:
                    g.append(rf"\draw[{COLOR[a]}, line width=0.5pt] (axis cs:{s:.3f},0) -- (axis cs:{s:.3f},0.35);")
g += [r"\end{groupplot}", r"\end{tikzpicture}", r"\\[2pt]\centerline{\pgfplotslegendfromname{legcal}}"]
(OUT / "fig_calibration.tex").write_text("\n".join(g) + "\n")

# ------------------------------------------------------------------ Fig. S2 + Table S4: target and example level
ex = pd.read_csv(REF / "sigma_star_test_v1/test_examples.csv").set_index("example_id")
ex = ex.loc[ok]
prod_heavy = np.where(ex["product"] == ex["pdb_a"], ex["heavy_a"], ex["heavy_b"])
ex = ex.assign(prod_heavy=prod_heavy)
SI, EI = avg("ISO")
SH, EH = avg("HARM")
tgt = pd.DataFrame({"t": ex["uniprot"], "SI": SI, "SH": SH}).groupby("t").agg(
    SI=("SI", "mean"), SH=("SH", "mean"), n=("SI", "size"))
V["fig_S2_targets"] = tgt.reset_index().to_dict(orient="records")
V["fig_S2_targets_harm_lower"] = int((tgt.SH < tgt.SI).sum())
classes = [("n<10", tgt.n < 10, 1.2), ("10<=n<100", (tgt.n >= 10) & (tgt.n < 100), 2.2), ("n>=100", tgt.n >= 100, 3.4)]
lim = float(np.ceil(max(tgt.SI.max(), tgt.SH.max()) * 2) / 2)
p = [r"\begin{tikzpicture}",
     rf"\begin{{axis}}[mehfaxis, width=0.9\columnwidth, height=0.9\columnwidth, xmin=0, xmax={lim}, ymin=0, ymax={lim},"
     r" xmajorgrids, xlabel={MEHF-ISO, per-target mean spread $S$}, ylabel={MEHF-HARM, per-target mean spread $S$},"
     r" legend pos=north west, legend style={fill=white, fill opacity=0.85, text opacity=1}]"]
for name, m, size in classes:
    sub = tgt[m]
    co = " ".join(f"({r.SI:.4f},{r.SH:.4f})" for r in sub.itertuples())
    lab = {"n<10": "$<10$ examples", "10<=n<100": "10--99 examples", "n>=100": "$\\geq 100$ examples"}[name]
    p.append(rf"\addplot[only marks, mark=*, mark size={size}pt, draw=cHARMd, fill=cHARM, fill opacity=0.65] coordinates {{{co}}};")
    p.append(rf"\addlegendentry{{{lab} ({len(sub)})}}")
p.append(rf"\draw[refline] (axis cs:0,0) -- (axis cs:{lim},{lim});")
p += [r"\end{axis}", r"\end{tikzpicture}"]
(OUT / "fig_targets.tex").write_text("\n".join(p) + "\n")

dS, dE = SH - SI, EH - EI
het = []
bins_h = [("$<20$", ex.prod_heavy < 20), ("20--29", (ex.prod_heavy >= 20) & (ex.prod_heavy < 30)),
          ("30--39", (ex.prod_heavy >= 30) & (ex.prod_heavy < 40)), ("$\\geq 40$", ex.prod_heavy >= 40)]
q = ex.mcs_fraction.quantile([0.25, 0.5, 0.75]).to_numpy()
bins_m = [(f"$\\leq {q[0]:.2f}$", ex.mcs_fraction <= q[0]),
          (f"{q[0]:.2f}--{q[1]:.2f}", (ex.mcs_fraction > q[0]) & (ex.mcs_fraction <= q[1])),
          (f"{q[1]:.2f}--{q[2]:.2f}", (ex.mcs_fraction > q[1]) & (ex.mcs_fraction <= q[2])),
          (f"$>{q[2]:.2f}$", ex.mcs_fraction > q[2])]
t = [r"\begin{tabular}{@{}lrrcccc@{}}", r"\toprule",
     r"Bin & $n$ & Targets & Median $\Delta S$ & $\Delta S<0$ & Median $\Delta E_4$ & $\Delta E_4<0$ \\", r"\midrule"]
for head, bins in (("Product heavy atoms", bins_h), ("MCS fraction (quartiles)", bins_m)):
    t.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{head}}}}} \\")
    for lab, m in bins:
        m = m.to_numpy()
        r = {"bin": lab, "n": int(m.sum()), "targets": int(ex.uniprot[m].nunique()),
             "dS_median": float(dS[m].median()), "dS_lower": float((dS[m] < 0).mean()),
             "dE4_median": float(dE[m].median()), "dE4_lower": float((dE[m] < 0).mean())}
        het.append({"group": head, **r})
        t.append(f"\\quad {lab} & {r['n']} & {r['targets']} & {fmt(r['dS_median'])} & {100 * r['dS_lower']:.1f}\\% & "
                 f"{fmt(r['dE4_median'])} & {100 * r['dE4_lower']:.1f}\\% \\\\")
t += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_heterogeneity.tex").write_text("\n".join(t) + "\n")
V["table_S4_heterogeneity"] = het
V["mcs_quartiles"] = q.tolist()

# ------------------------------------------------------------------ Table S3: weighting
p1 = json.load(open(AN / "posthoc_v1/RESULT.json"))["paired"]
p2 = json.load(open(AN / "posthoc_v2/RESULT.json"))
t = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
     r"& \multicolumn{2}{c}{Example-weighted (reported)} & \multicolumn{2}{c}{Target-weighted (post hoc)} \\",
     r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
     r"Endpoint & Mean [95\% CI] & HARM lower & Mean [95\% CI] & HARM lower \\", r"\midrule"]
for m in ("S", "E4", "E5", "E3"):
    a, b = p1[m]["SIG_minus_ISO"], p2[f"equal_target_{m}"]
    nm = "$S$" if m == "S" else f"$E_{m[1]}$"
    t.append(f"{nm} & {fmt(a['mean'])} [{fmt(a['ci95_mean'][0])}, {fmt(a['ci95_mean'][1])}] & "
             f"{100 * a['frac_sig_lower']:.1f}\\% of examples & {fmt(b['mean'])} [{fmt(b['ci95'][0])}, {fmt(b['ci95'][1])}] & "
             f"{b['targets_lower']}/{b['n_targets']} targets \\\\")
t += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_weighting.tex").write_text("\n".join(t) + "\n")

# ------------------------------------------------------------------ Table S5: width provenance
tau = json.load(open(REF / "leakage_audit_v1/audit.json"))["tau_contamination"]
NM = {"bond_angstrom": ("Bond length", "\\AA", 4), "angle_deg": ("Bond angle", "$^{\\circ}$", 3),
      "torsion_deg": ("Torsion", "$^{\\circ}$", 2), "oop_deg": ("Out-of-plane", "$^{\\circ}$", 3)}
t = [r"\begin{tabular}{@{}lrrrccr@{}}", r"\toprule",
     r"Channel & Systems & Val. & Test & Width (all) & Width (w/o val./test) & Change \\", r"\midrule"]
for k, (nm, u, d) in NM.items():
    r = tau[k]
    t.append(f"{nm} ({u}) & {r['n_systems']:,} & {r['n_val']} & {r['n_test']} & {r['median_all']:.{d + 1}f} & "
             f"{r['median_without_val_test']:.{d + 1}f} & {fmt(100 * r['relative_change'], 2)}\\% \\\\")
t += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_widths.tex").write_text("\n".join(t) + "\n")
V["table_S5_widths"] = tau

# ------------------------------------------------------------------ Fig. S3: pocket overlap
po = json.load(open(AN / "pocket_v1/RESULT.json"))
ov = np.array([s["overlap"] * 96 for s in po["per_system"]])
edges = np.arange(np.floor(ov.min()) - 0.5, 97.5, 2.0)
h, _ = np.histogram(ov, bins=edges)
V["fig_S3_overlap_atoms"] = {"median": float(np.median(ov)), "q25": float(np.percentile(ov, 25)),
                             "q75": float(np.percentile(ov, 75)), "min": float(ov.min()), "n": int(len(ov))}
co = " ".join(f"({0.5 * (edges[i] + edges[i + 1]):.1f},{h[i]})" for i in range(len(h)))
p = [r"\begin{tikzpicture}",
     r"\begin{axis}[mehfaxis, ybar interval=0.9, width=0.9\columnwidth, height=4.2cm, ymin=0,"
     r" xtick={55,60,...,95}, xticklabel style={/pgf/number format/.cd, fixed, precision=0},"
     r" xlabel={Pocket atoms shared with the first-frame 96 (of 96)}, ylabel={Test systems}]"]
p.append(r"\addplot[fill=cISO!70, draw=white] coordinates {" +
         " ".join(f"({edges[i]:.1f},{h[i]})" for i in range(len(h))) + f" ({edges[-1]:.1f},0)" + "};")
med = float(np.median(ov))
p.append(rf"\draw[refline] (axis cs:{med:.2f},0) -- (axis cs:{med:.2f},{h.max() * 1.05:.1f}) node[above, font=\scriptsize] {{median {med:.0f}}};")
p += [r"\end{axis}", r"\end{tikzpicture}"]
(OUT / "fig_pocket.tex").write_text("\n".join(p) + "\n")

# ------------------------------------------------------------------ Table S6: angle-term effects
t = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule", r"Effect & Fold 1 & Fold 2 & Fold 3 & Mean \\", r"\midrule"]
for lab, k in (("Bond-angle loss", "angle_loss_pct"), ("Angle step of projection", "angle_proj_pct")):
    vals = p2[k + "_per_fold"] + [p2[k + "_mean"]]
    t.append(lab + " & " + " & ".join(fmt(v, 1) + "\\%" for v in vals) + r" \\")
t += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_angleterm.tex").write_text("\n".join(t) + "\n")

(AN / "supplement_v1/VALUES.json").write_text(json.dumps(V, indent=1, default=float))
print("checks passed; wrote", sorted(x.name for x in OUT.glob("*.tex")))
print("targets HARM lower:", V["fig_S2_targets_harm_lower"], "of", len(tgt))
for r in het:
    print(r)
print(V["fig_S3_overlap_atoms"])
