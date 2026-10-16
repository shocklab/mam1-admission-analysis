"""Placement: do the two support streams differ on the measures used to place students?

Emits tab_placement_cohort.tex, tab_entrycourse.tex and values_placement.tex.

The main result is a null (the school mark does not separate the two support streams), so the
script reports what is needed to make a null informative rather than merely non-significant:

  - the minimum detectable difference, so the reader can see whether the null is tight
  - omega-squared as well as eta-squared, because eta-squared is biased upward and is above zero
    in expectation even when there is no effect
  - the same comparison BY COHORT, since a pooled null can hide differences of opposite sign
  - a movers sensitivity, restricting to students with a single semester-1 stream record (a
    change of stream between semester 1 and semester 2 is not counted as a move)

The entry-course comparison is the one outcome available for all three streams, since MAM1005H
has no semester-1 result. 2026 is excluded from it because most extended grades were unresolved.
Run from the project root.
"""
import warnings, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
warnings.simplefilter("ignore")
if not Path("anonymised-data").is_dir():
    raise SystemExit("run from the project root")
rng = np.random.default_rng(20260819)
T = Path("paper/tables")
d = pd.read_csv("anonymised-data/paper1_analysis_sample_3streams.csv").rename(
    columns={"Math Grd12 Fin Rslt": "nsc", "NBT Math Score": "nbt"})
sup = d[d.stream.isin(["extended", "augmented"])].copy()
v = {}


def eta(x, var):
    s = x.dropna(subset=[var]); gm = s[var].mean()
    ssb = sum(len(g) * (g[var].mean() - gm) ** 2 for _, g in s.groupby("stream"))
    return 100 * ssb / ((s[var] - gm) ** 2).sum()


rows = []
for co, g in sup.groupby("cohort"):
    e, a = g[g.stream == "extended"], g[g.stream == "augmented"]
    rows.append(f"{co} & {len(e)} & {len(a)} & {e.nsc.mean()-a.nsc.mean():+.2f} & "
                f"{eta(g,'nsc'):.2f} & {e.nbt.mean()-a.nbt.mean():+.2f} & {eta(g,'nbt'):.2f} \\\\")
e, a = sup[sup.stream == "extended"], sup[sup.stream == "augmented"]
rows.append("\\midrule\nPooled & %d & %d & %+.2f & %.2f & %+.2f & %.2f \\\\" % (
    len(e), len(a), e.nsc.mean()-a.nsc.mean(), eta(sup, "nsc"),
    e.nbt.mean()-a.nbt.mean(), eta(sup, "nbt")))
(T/"tab_placement_cohort.tex").write_text("""\\begin{tabular}{lrrrrrr}
\\toprule
 & \\multicolumn{2}{c}{$n$} & \\multicolumn{2}{c}{School Mathematics} & \\multicolumn{2}{c}{NBT Mathematics} \\\\
\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}
Cohort & extended & augmented & difference & $\\eta^2$ (\\%) & difference & $\\eta^2$ (\\%) \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}""")
s2 = sup.copy(); s2["cohort"] = s2.cohort.astype(str)
m0 = smf.ols("nsc ~ stream + cohort", s2).fit(); m1 = smf.ols("nsc ~ stream * cohort", s2).fit()
v["InterP"] = f"{m1.compare_f_test(m0)[1]:.4f}"
v["ExtNsix"] = int((s2.cohort.eq("2026") & s2.stream.eq("extended")).sum())
v["ExtNfive"] = int((s2.cohort.eq("2025") & s2.stream.eq("extended")).sum())
v["ExtNSCsix"] = f"{s2[s2.cohort.eq('2026') & s2.stream.eq('extended')].nsc.mean():.1f}"
v["ExtNSCfive"] = f"{s2[s2.cohort.eq('2025') & s2.stream.eq('extended')].nsc.mean():.1f}"

# entry-course outcome, common to all three streams. 2026 excluded: its extended grades are mostly
# unresolved, being a full-year course.
ent = d[d.s1_pass.notna() & (d.cohort < 2026)]
erows = []
for st in ("mainstream", "augmented", "extended"):
    g = ent[ent.stream == st]
    erows.append(f"{st.capitalize()} & {len(g)} & {100*g.s1_pass.mean():.1f} \\\\")
(T/"tab_entrycourse.tex").write_text("""\\begin{tabular}{lrr}
\\toprule
Stream & $n$ classified & Passed the entry course (\\%) \\\\
\\midrule
""" + "\n".join(erows) + """
\\bottomrule
\\end{tabular}""")
ee, aa = ent[ent.stream == "extended"].s1_pass.values, ent[ent.stream == "augmented"].s1_pass.values
bs = [100*(rng.choice(ee, len(ee)).mean() - rng.choice(aa, len(aa)).mean()) for _ in range(20000)]
v |= {"EntryDiff": f"{100*(ee.mean()-aa.mean()):+.1f}",
      "EntryLo": f"{np.percentile(bs,2.5):+.1f}", "EntryHi": f"{np.percentile(bs,97.5):+.1f}",
      "EntryExt": f"{100*ee.mean():.1f}", "EntryAug": f"{100*aa.mean():.1f}"}

# unfloored omega-squared, and the smallest difference the design could detect
s = sup.dropna(subset=["nsc"]); gm = s.nsc.mean(); k = 2; n = len(s)
ssb = sum(len(g)*(g.nsc.mean()-gm)**2 for _, g in s.groupby("stream"))
sst = ((s.nsc-gm)**2).sum(); msw = (sst-ssb)/(n-k)
v["OmegaRaw"] = f"{100*(ssb-(k-1)*msw)/(sst+msw):+.2f}"
sd = np.sqrt(msw); n1, n2 = len(e.dropna(subset=["nsc"])), len(a.dropna(subset=["nsc"]))
v["MDEmarks"] = f"{2.8*sd*np.sqrt(1/n1+1/n2):.1f}"
Path(T/"values_placement.tex").write_text(
    "\n".join(f"\\newcommand{{\\{kk}}}{{{x}}}" for kk, x in v.items()))
print("values:", v)

# ---- movers sensitivity (students can move between streams during the year) --------
# Single semester-1 stream record only; a semester-2 change of stream is not counted.
settled = d[d.s1_streams_n <= 1]
sub = settled[settled.stream.isin(["extended", "augmented"])]
e2, a2 = sub[sub.stream == "extended"], sub[sub.stream == "augmented"]
v["SetExtN"] = len(e2); v["SetAugN"] = len(a2)
v["SetNSCdiff"] = f"{e2.nsc.mean()-a2.nsc.mean():+.2f}"
v["SetEtaNSC"] = f"{eta(sub,'nsc'):.2f}"; v["SetEtaNBT"] = f"{eta(sub,'nbt'):.2f}"
ent2 = settled[settled.s1_pass.notna() & (settled.cohort < 2026)]
ee2 = ent2[ent2.stream == "extended"].s1_pass; aa2 = ent2[ent2.stream == "augmented"].s1_pass
bs2 = [100*(rng.choice(ee2, len(ee2)).mean() - rng.choice(aa2, len(aa2)).mean()) for _ in range(20000)]
v |= {"SetEntryDiff": f"{100*(ee2.mean()-aa2.mean()):+.1f}",
      "SetEntryLo": f"{np.percentile(bs2,2.5):+.1f}", "SetEntryHi": f"{np.percentile(bs2,97.5):+.1f}",
      "NMovers": int((d.s1_streams_n > 1).sum()),
      "NMainstream": int((d.stream == "mainstream").sum()),
      "ExtN": int((d.stream == "extended").sum()),
      "AugN": int((d.stream == "augmented").sum())}
Path("paper/tables/values_placement.tex").write_text(
    "\n".join(f"\\newcommand{{\\{kk}}}{{{x}}}" for kk, x in v.items()))
print("movers sensitivity:", {k: v[k] for k in ("NMovers","SetExtN","SetEtaNSC","SetEntryDiff")})
