"""Numbers in the paper that no other script produces.

These are the screening counts and the comparison of excluded with retained students
(Section 3.1), the sensitivity analyses without the 2026 cohort and without the 25 passes
recorded as a code rather than a mark (Section 3.2), the population-group models with separate
intercepts and slopes (Section 3.5), and the mainstream entry-course denominator (Table 12).
This script recomputes each from the analysis sample, with the specifications the paper
describes, and prints it beside the value reported in the paper. It prints aggregate figures
only. Run from the project root.
"""
import warnings, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
warnings.simplefilter("ignore")
if not Path("anonymised-data").is_dir():
    raise SystemExit("run from the project root")
d = pd.read_csv("anonymised-data/paper1_analysis_sample_3streams.csv").rename(
    columns={"Math Grd12 Fin Rslt": "nsc", "NBT Math Score": "nbt", "Population Grp": "grp"})
d["cohort"] = d.cohort.astype(str)
BIN = sm.families.Binomial()


def show(label, ours, paper):
    print(f"  {label:<58} ours {ours:<22} manuscript {paper}")


def fit(f, data, binary):
    return smf.glm(f, data, family=BIN).fit() if binary else smf.ols(f, data).fit()


def lr(f0, f1, data, binary):
    m0, m1 = fit(f0, data, binary), fit(f1, data, binary)
    x = 2 * (m1.llf - m0.llf); k = int(round(m1.df_model - m0.df_model))
    return x, k, stats.chi2.sf(x, k)


# ---- Section 3.1: screening and the excluded students -------------------------------------
# The analysis file holds complete cases only, so the screened population is rebuilt by running
# scripts/10 with its output sent to a temporary folder and reading its table D before the
# complete-case filter. The files in anonymised-data are not touched.
import io, os, runpy, tempfile, contextlib
print("Section 3.1, screening")
with tempfile.TemporaryDirectory() as tmp:
    os.environ["OUTCSV"] = str(Path(tmp) / "x_3streams.csv")
    with contextlib.redirect_stdout(io.StringIO()):
        D = runpy.run_path("scripts/10_build_analysis_sample.py")["D"]
    del os.environ["OUTCSV"]
D = D.rename(columns={"Math Grd12 Fin Rslt": "nsc", "NBT Math Score": "nbt"})
D["cohort"] = D.cohort.astype(str)
s = D[(D.scope == "three") & D.first_time]
d = d.assign(complete=True)
assert len(s[s.complete]) == len(d), "scripts/10 no longer gives the analysis file's sample"
inc = s[~s.complete]; ret = d
show("screened, by cohort", f"{len(s)} = " + "/".join(str(n) for n in s.cohort.value_counts().sort_index()),
     "1,119 = 252/470/397")
show("excluded, by cohort", f"{len(inc)} = " + "/".join(str(n) for n in inc.cohort.value_counts().sort_index()),
     "137 = 35/49/53")
show("retained, by cohort", f"{len(ret)} = " + "/".join(str(n) for n in ret.cohort.value_counts().sort_index()),
     "982 = 217/421/344")
for lab, g in (("excluded", inc), ("retained", ret)):
    sh = g.stream.value_counts(normalize=True)
    show(f"{lab}: % mainstream, % extended",
         f"{100*sh.get('mainstream', 0):.1f}, {100*sh.get('extended', 0):.1f}",
         "74.5, 10.2" if lab == "excluded" else "61.1, 22.4")
show("excluded with an NBT score: n, mean", f"{inc.nbt.notna().sum()}, {inc.nbt.mean():.1f}", "85, 72.7")
show("retained: mean NBT", f"{ret.nbt.mean():.1f}", "69.7")

m = ret[ret.stream.isin(["mainstream", "augmented"])]
mk = m.dropna(subset=["nsc", "nbt", "s1_mark"]); pa = m.dropna(subset=["nsc", "nbt", "s1_pass"])


def nested(mkd, pad, tag, paper):
    out = []
    for y, data, binary in (("s1_mark", mkd, False), ("s1_pass", pad, True)):
        nsc = f"{y} ~ bs(nsc, df=4) + stream + cohort"
        nbt = f"{y} ~ bs(nbt, df=4) + stream + cohort"
        both = f"{y} ~ bs(nsc, df=4) + bs(nbt, df=4) + stream + cohort"
        a = lr(nsc, both, data, binary); b = lr(nbt, both, data, binary)
        out.append(f"{y} n {len(data)}: NBT adds {a[0]:.2f} (p {a[2]:.2g}), NSC adds {b[0]:.2f} (p {b[2]:.2g})")
    print(f"  {tag}"); [print("    ours      ", o) for o in out]; print("    manuscript", paper)


def jump80(data, y, binary):
    s = data.copy(); s["jump"] = (s.nsc >= 80).astype(float)
    f0 = f"{y} ~ bs(nsc, df=4) + stream + cohort"
    m1 = fit(f0 + " + jump", s, binary)
    x = max(2 * (m1.llf - fit(f0, s, binary).llf), 0)
    return m1.params["jump"], m1.bse["jump"], stats.chi2.sf(x, 1)


# ---- Section 3.2 / 3.3: sensitivity analyses ----------------------------------------------
print("\nSection 3.2, sensitivity analyses")
n26 = lambda x: x[x.cohort != "2026"]
nested(n26(mk), n26(pa), "without 2026",
       "mark n 440: 61.07, 52.04; pass n 462: 19.87 (p 0.00053), 14.28 (p 0.0064)")
for y, data, binary, paper in (("s1_mark", n26(mk.dropna(subset=["s1_mark"])), False, "0.754"),
                               ("s1_pass", n26(pa), True, "0.997")):
    c_, se, p = jump80(data, y, binary)
    show(f"without 2026, jump at 80, {y}: p", f"{p:.3f}", paper)
code_only = (pa.s1_pass == 1) & pa.s1_mark.isna()
show("code-only passes in the pass sample", int(code_only.sum()), "25")
nested(mk, pa[~code_only], "without code-only passes (pass rows only apply)",
       "pass n 685: 20.80 (p 0.00035), 21.15 (p 0.00030)")
c_, se, p = jump80(mk, "s1_mark", False)
show("jump at 80, mark: coefficient, SE", f"{c_:+.2f}, {se:.2f}", "-0.66, 3.21")

# ---- Section 3.5: population-group models -------------------------------------------------
print("\nSection 3.5, population-group models (mark sample)")
big = [g for g, n in m.grp.value_counts().items() if n >= 40]
mk = mk.assign(grp2=np.where(mk.grp.isin(big), mk.grp, "Other or not recorded"))
print(f"  groups: {mk.grp2.value_counts().to_dict()}")
ps = []
for sc, paper in (("nsc", "chi2(8) 58.60; chi2(4) 19.70"), ("nbt", "chi2(8) 23.73; chi2(4) 15.49")):
    base = f"s1_mark ~ {sc} + stream + cohort"
    main = f"s1_mark ~ {sc} + C(grp2) + stream + cohort"
    full = f"s1_mark ~ {sc} * C(grp2) + stream + cohort"
    a = lr(base, full, mk, False); b = lr(main, full, mk, False); ps.append(a[2])
    show(f"{sc}: intercepts+slopes; slopes given intercepts",
         f"chi2({a[1]}) {a[0]:.2f}, chi2({b[1]}) {b[0]:.2f}", paper)
    print(f"      p values: {a[2]:.2g}, {b[2]:.2g}")
order = np.argsort(ps); adj = np.empty(2); run = 0
for i, j in enumerate(order):
    run = max(run, min(1, (len(ps) - i) * ps[j])); adj[j] = run
show("Holm-adjusted omnibus p (nsc, nbt)", f"{adj[0]:.2g}, {adj[1]:.4f}", "<0.001, 0.0025")
mf = fit("s1_mark ~ nsc * C(grp2) + stream + cohort", mk, False)
for x, paper in ((80, "-1.92 [-6.49, 2.65]"), (85, "-5.94 [-8.98, -2.91]"), (90, "-9.96 [-12.91, -7.02]")):
    L = pd.Series(0.0, index=mf.params.index)
    L["C(grp2)[T.White]"] = -1; L["nsc:C(grp2)[T.White]"] = -x   # African is the reference level
    t = mf.t_test(L.values.reshape(1, -1)); lo, hi = t.conf_int()[0]
    show(f"African minus White at school mark {x}", f"{float(t.effect[0]):+.2f} [{lo:+.2f}, {hi:+.2f}]", paper)

# ---- Table 12: mainstream entry-course denominator ----------------------------------------
print("\nTable 12, mainstream 2024-25")
ms = ret[(ret.stream == "mainstream") & ret.cohort.isin(["2024", "2025"])]
show("mainstream students", len(ms), "390")
show("no semester-1 record", int(ms.s1_course.isna().sum()), "17")
show("semester-1 record, no classifiable result",
     int((ms.s1_course.notna() & ms.s1_pass.isna()).sum()), "4")
show("classified", int(ms.s1_pass.notna().sum()), "369")
