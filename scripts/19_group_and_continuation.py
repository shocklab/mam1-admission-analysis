"""Differential prediction by population group, false exclusion, and continuation.

Emits tab_groups.tex, tab_falseexcl.tex, tab_continuation.tex and values_groups.tex.

Two results that are different claims, kept separate in the paper:

  1. VALIDITY. A Cleary-style test of whether school-leaving Mathematics predicts the outcome in
     the same way across groups, decomposed into intercept and slope components, with the same
     test for NBT Mathematics.
  2. COMPOSITION. The false-exclusion table: how many students in each group who passed would be
     excluded by a cut at 80. This depends on where each group's marks sit relative to the cut as
     well as on pass rates below it, so the script also reports below-80 pass rates by group.

The group labels are the institution's own, taken verbatim from the extract. Groups under 40
students are pooled rather than dropped. Run from the project root.
"""
import warnings, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
from scipy import stats
warnings.simplefilter("ignore")
if not Path("anonymised-data").is_dir():
    raise SystemExit("run from the project root")
T = Path("paper/tables")
d = pd.read_csv("anonymised-data/paper1_analysis_sample_3streams.csv")
pgc = [c for c in d.columns if "opulation" in c][0]
d = d.rename(columns={"Math Grd12 Fin Rslt": "nsc", "NBT Math Score": "nbt", pgc: "grp"})
d["cohort"] = d.cohort.astype(str)
m = d[d.stream.isin(["mainstream", "augmented"])].copy()
big = [g for g, n in m.grp.value_counts().items() if n >= 40]
m["grp2"] = np.where(m.grp.isin(big), m.grp, "Other or not recorded")


def w(name, body):
    (T / name).write_text(body); print("  wrote", name)


# ---- differential validity -------------------------------------------------
mk = m.dropna(subset=["nsc", "nbt", "s1_mark", "grp2"])
rows = []
for g, s in mk.groupby("grp2"):
    r1, b1 = s.nsc.corr(s.s1_mark), np.polyfit(s.nsc, s.s1_mark, 1)[0]
    r2, b2 = s.nbt.corr(s.s1_mark), np.polyfit(s.nbt, s.s1_mark, 1)[0]
    rows.append((g, len(s), r1, b1, r2, b2))
rows.sort(key=lambda r: -r[1])
w("tab_groups.tex", """\\begin{tabular}{lrrrrr}
\\toprule
 & & \\multicolumn{2}{c}{School-leaving Mathematics} & \\multicolumn{2}{c}{NBT Mathematics} \\\\
\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}
Group & $n$ & $r$ & slope & $r$ & slope \\\\
\\midrule
""" + "\n".join(f"{g} & {n} & {r1:.2f} & {b1:.2f} & {r2:.2f} & {b2:.2f} \\\\"
                for g, n, r1, b1, r2, b2 in rows) + """
\\bottomrule
\\end{tabular}""")

# ---- Cleary test -----------------------------------------------------------
vals = {}
mk = mk.copy(); mk["afr"] = (mk.grp2 == "African").astype(int)
for var, key in (("nsc", "NSC"), ("nbt", "NBT")):
    a = smf.ols(f"s1_mark ~ {var} + stream + cohort", mk).fit()
    b = smf.ols(f"s1_mark ~ {var} * afr + stream + cohort", mk).fit()
    lr = 2 * (b.llf - a.llf); df = b.df_model - a.df_model
    vals[f"Cleary{key}"] = f"{lr:.1f}"
    vals[f"ClearyP{key}"] = ("<10^{-6}" if stats.chi2.sf(lr, df) < 1e-6
                             else f"{stats.chi2.sf(lr, df):.3f}")

# ---- false exclusion at a cut of 80 ---------------------------------------
c = m.dropna(subset=["nsc", "s1_pass", "grp2"])
frows = []
for g, s in c.groupby("grp2"):
    ex = int(((s.nsc < 80) & (s.s1_pass == 1)).sum())
    frows.append((g, len(s), 100 * (s.nsc < 80).mean(), ex, 100 * ex / len(s)))
frows.sort(key=lambda r: -r[1])
w("tab_falseexcl.tex", """\\begin{tabular}{lrrrr}
\\toprule
Group & $n$ & \\% below 80 & Excluded who passed & \\% of group \\\\
\\midrule
""" + "\n".join(f"{g} & {n} & {p:.1f} & {ex} & {q:.1f} \\\\" for g, n, p, ex, q in frows) + """
\\bottomrule
\\end{tabular}""")
# quote the range across NAMED groups; the pooled residual category is too small to name
named = [r for r in frows if not r[0].startswith("Other")]
vals |= {"FEmin": f"{min(r[4] for r in named):.1f}",
         "FEmax": f"{max(r[4] for r in named):.1f}",
         "FEminG": min(named, key=lambda r: r[4])[0],
         "FEmaxG": max(named, key=lambda r: r[4])[0]}

# ---- semester 2 continuation ----------------------------------------------
# 2024 and 2025 only: a 2026 semester-two record is a registration not yet graded or cleaned for
# the prerequisite, so it says nothing yet about who continued.
cont = m[m.cohort.isin(["2024", "2025"])].dropna(subset=["nsc"]).copy()
cont["grpNSC"] = np.where(cont.nsc >= 80, "80 or above", "below 80")
crows = []
for (g, st), s in cont.groupby(["grpNSC", "stream"]):
    if len(s) < 10: continue
    crows.append((g, st, len(s), 100 * s.observed_s2.mean()))
w("tab_continuation.tex", """\\begin{tabular}{llrr}
\\toprule
School mark & Stream & $n$ & Observed in semester 2 (\\%) \\\\
\\midrule
""" + "\n".join(f"{g} & {st} & {n} & {p:.1f} \\\\" for g, st, n, p in crows) + """
\\bottomrule
\\end{tabular}""")
b80 = cont[cont.nsc < 80]
vals |= {"ContBelowAug": f"{100*b80[b80.stream=='augmented'].observed_s2.mean():.0f}",
         "ContBelowMain": f"{100*b80[b80.stream=='mainstream'].observed_s2.mean():.0f}"}
w("values_groups.tex", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in vals.items()))
print("\nvalues:", vals)

# ---- decomposition and the common-window check ----------------------------
mk2 = mk.copy()
for var, key in (("nsc", "NSC"), ("nbt", "NBT")):
    f0 = f"s1_mark ~ {var} + stream + cohort"
    fi = f"s1_mark ~ {var} + afr + stream + cohort"
    fs = f"s1_mark ~ {var} * afr + stream + cohort"
    a0, ai, asl = (smf.ols(f, mk2).fit() for f in (f0, fi, fs))
    vals[f"Int{key}"] = f"{2*(ai.llf-a0.llf):.1f}"
    vals[f"Slope{key}"] = f"{2*(asl.llf-ai.llf):.1f}"
# common-window Cleary: equalises dispersion across groups, so it cannot be a
# differential-truncation artefact
wnd = mk2[(mk2.nsc >= 80) & (mk2.nsc <= 95)]
a0 = smf.ols("s1_mark ~ nsc + stream + cohort", wnd).fit()
a1 = smf.ols("s1_mark ~ nsc * afr + stream + cohort", wnd).fit()
vals["WinCleary"] = f"{2*(a1.llf-a0.llf):.1f}"
for g, key in (("African", "Afr"), ("White", "Whi")):
    s = wnd[wnd.grp2 == g]
    vals[f"WinR{key}"] = f"{s.nsc.corr(s.s1_mark):.2f}"
    vals[f"WinSD{key}"] = f"{s.nsc.std():.1f}"
# below-80 pass rates by group: flat, so the false-exclusion spread is composition
b80 = c[c.nsc < 80]
rr = [100*s.s1_pass.mean() for g, s in b80.groupby("grp2") if len(s) >= 15]
vals |= {"BelowEightyMin": f"{min(rr):.0f}", "BelowEightyMax": f"{max(rr):.0f}"}
w("values_groups.tex", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in vals.items()))
print("decomposition added:", {k: vals[k] for k in ("IntNSC","SlopeNSC","IntNBT","SlopeNBT","WinCleary","BelowEightyMin","BelowEightyMax")})
