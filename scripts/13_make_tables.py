"""Regenerate most of the paper's tables from the analysis sample and emit LaTeX fragments.

Produces, in order: the sample table, the school-mark x benchmark-quartile crosstab, the nested
spline models, the fitted-surface predictions, the support-stream comparison, a batch of macros
and the discontinuity tests. The model comparison table is written by scripts/17 and the
admission comparison by scripts/22.

Reads anonymised-data/paper1_analysis_sample_3streams.csv, which scripts/10 writes. Read that
script first: the judgement calls that decide the sample all live there, not here.

Run from the project root.
"""
import warnings, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
warnings.simplefilter("ignore")
# Fixed seed: the bootstrap intervals below are otherwise not reproducible between runs.
rng = np.random.default_rng(20260817)
if not Path("anonymised-data").is_dir():
    raise SystemExit("run from the project root: the paths here are relative to it")
T = Path("paper/tables"); T.mkdir(parents=True, exist_ok=True)
d = pd.read_csv("anonymised-data/paper1_analysis_sample_3streams.csv").rename(
    columns={"Math Grd12 Fin Rslt": "nsc", "NBT Math Score": "nbt"})
d["cohort"] = d.cohort.astype(str)
# "two" is the separately built two-stream sample (scripts/10), used only for the counts that
# compare the two samples. Outcome analyses use "m" below, the mainstream and augmented students
# of the three-stream sample, because MAM1005H runs a full year and so has no semester-1 outcome
# to compare; "d" (all three streams) is used for intake and placement.
two = pd.read_csv("anonymised-data/paper1_analysis_sample_2streams.csv").rename(
    columns={"Math Grd12 Fin Rslt": "nsc", "NBT Math Score": "nbt"})


def w(name, body):
    (T/name).write_text(body); print(f"  wrote {name}")


# ---- Table 1: sample -------------------------------------------------------
p = d.pivot_table(index="cohort", columns="stream", values="anon_id", aggfunc="count").fillna(0).astype(int)
p = p[["mainstream", "augmented", "extended"]]
p["Total"] = p.sum(axis=1)
p.loc["Total"] = p.sum()
rows = "\n".join(f"{i} & {r.mainstream} & {r.augmented} & {r.extended} & {r.Total} \\\\"
                 for i, r in p.iterrows())
w("tab_sample.tex", f"""\\begin{{tabular}}{{lrrrr}}
\\toprule
Cohort & Mainstream & Augmented & Extended & Total \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}""")

# ---- Table 2: NSC band x NBT quartile --------------------------------------
c = d.dropna(subset=["nsc", "nbt"]).copy()
# School-mark bands are fixed 5-point bins; benchmark bands are sample QUARTILES. The two are
# not the same kind of cut, so these cells cannot be read as a like-for-like contest between
# the measures. They show the benchmark varies within school-mark band, which is the claim the
# paper actually makes.
# The first band is open below: admission asks for 70, but a student below it would otherwise drop
# out of the table silently (the sample has one).
c["band"] = pd.cut(c.nsc, [0, 74, 79, 84, 89, 101], labels=["below 75", "75--79", "80--84", "85--89", "90+"])
assert c.band.notna().all(), "a school mark falls outside the Table 2 bands"
c["q"] = pd.qcut(c.nbt, 4, labels=["Q1", "Q2", "Q3", "Q4"])
ct = pd.crosstab(c.band, c.q)
rows = "\n".join(f"{i} & {r.Q1} & {r.Q2} & {r.Q3} & {r.Q4} & {r.sum()} \\\\" for i, r in ct.iterrows())
below_top = int(ct.loc[["below 75", "75--79"], ["Q3", "Q4"]].values.sum())
above_bot = int(ct.loc[["80--84", "85--89", "90+"], "Q1"].sum())
w("tab_crosstab.tex", f"""\\begin{{tabular}}{{lrrrrr}}
\\toprule
NSC Mathematics & \\multicolumn{{4}}{{c}}{{NBT Mathematics quartile}} & \\\\
\\cmidrule(lr){{2-5}}
band & Q1 (low) & Q2 & Q3 & Q4 (high) & $n$ \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}""")
_m = c.nbt.median()
# "strictly above" and "at or below" are COMPLEMENTARY, so this is a partition: every
# student falls in exactly one side. Using strict on both sides would drop the
# students who sit exactly on the median.
vals_extra = {"MatchExcl": int((c[c.nsc < 80].nbt > _m).sum()),
              "MatchAdm": int((c[c.nsc >= 80].nbt <= _m).sum()),
              "NTied": int((c.nbt == _m).sum()),
              "NInclusive": int(((c.nsc < 80) & (c.nbt >= _m)).sum()),
              "PctExclAbove": f"{100*(c[c.nsc < 80].nbt > _m).mean():.0f}",
              "PctAdmBelow": f"{100*(c[c.nsc >= 80].nbt <= _m).mean():.0f}"}
print(f"  below-80 in top half NBT = {below_top}; 80+ in bottom quartile = {above_bot}")

# ---- Table 3: nested models ------------------------------------------------
m = d[d.stream.isin(["mainstream", "augmented"])]
mk = m.dropna(subset=["nsc", "nbt", "s1_mark"]); pa = m.dropna(subset=["nsc", "nbt", "s1_pass"])
# The nested models. Spline bases rather than straight lines, so the shape of each relationship
# is estimated rather than assumed linear; stream and cohort are adjusted for throughout. The
# variance-explained columns are for the WHOLE model including stream and cohort, not for the
# mathematics measure alone.
FM = "s1_mark ~ bs(nsc, df=4) + bs(nbt, df=4) + stream + cohort"
FP = "s1_pass ~ bs(nsc, df=4) + bs(nbt, df=4) + stream + cohort"
res = []
for lab, y, data, fam, full in (("Semester 1 mark ($R^2$)", "s1_mark", mk, None, FM),
                                ("Semester 1 pass (pseudo-$R^2$)", "s1_pass", pa, sm.families.Binomial(), FP)):
    fits = {}
    for k, f in (("nsc", f"{y} ~ bs(nsc, df=4) + stream + cohort"),
                 ("nbt", f"{y} ~ bs(nbt, df=4) + stream + cohort"), ("both", full)):
        fits[k] = smf.ols(f, data).fit() if fam is None else smf.glm(f, data, family=fam).fit()
    def r2(fit): return fit.rsquared if fam is None else 1 - fit.llf/fit.llnull
    lr = 2*(fits["both"].llf - fits["nsc"].llf); dd = fits["both"].df_model - fits["nsc"].df_model
    lr2 = 2*(fits["both"].llf - fits["nbt"].llf); dd2 = fits["both"].df_model - fits["nbt"].df_model
    res.append(f"{lab} & {len(data)} & {r2(fits['nsc']):.3f} & {r2(fits['nbt']):.3f} & "
               f"{r2(fits['both']):.3f} & {lr:.1f} & {stats.chi2.sf(lr,dd):.1e} & "
               f"{lr2:.1f} & {stats.chi2.sf(lr2,dd2):.1e} \\\\")
w("tab_nested.tex", """\\begin{tabular}{lrrrrrrrr}
\\toprule
 & & \\multicolumn{3}{c}{Variance explained} & \\multicolumn{2}{c}{NBT adds} & \\multicolumn{2}{c}{NSC adds} \\\\
\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}
Outcome & $n$ & NSC & NBT & both & $\\chi^2_4$ & $p$ & $\\chi^2_4$ & $p$ \\\\
\\midrule
""" + "\n".join(res) + """
\\bottomrule
\\end{tabular}""")

# ---- Table 4: surface predictions ------------------------------------------
mf = smf.ols(FM, mk).fit(); pf = smf.glm(FP, pa, family=sm.families.Binomial()).fit()
pts = [(78, 80), (85, 60), (85, 85)]
g = pd.DataFrame(pts, columns=["nsc", "nbt"]).assign(stream="mainstream", cohort="2025")
g["mark"] = mf.predict(g); g["pass"] = pf.predict(g)
rows = "\n".join(f"{int(r.nsc)} & {int(r.nbt)} & {r['mark']:.1f} & {r['pass']:.2f} \\\\"
                 for _, r in g.iterrows())
w("tab_surface.tex", f"""\\begin{{tabular}}{{rrrr}}
\\toprule
NSC Mathematics & NBT Mathematics & Predicted mark & Predicted $P$(pass) \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}""")

# ---- Table 5: placement ----------------------------------------------------
# Percentile bootstrap on a difference, resampling the two groups independently.
def boot(a, b, n=20000):
    a, b = np.asarray(a.dropna()), np.asarray(b.dropna())
    s = rng.choice(a, (n, len(a))).mean(1) - rng.choice(b, (n, len(b))).mean(1)
    return np.percentile(s, [2.5, 97.5])
def dcoh(a, b):
    a, b = a.dropna(), b.dropna()
    sp = np.sqrt(((len(a)-1)*a.var()+(len(b)-1)*b.var())/(len(a)+len(b)-2))
    return (a.mean()-b.mean())/sp
e, a = d[d.stream == "extended"], d[d.stream == "augmented"]
sup = d[d.stream.isin(["extended", "augmented"])]
rows, place_ci = [], {}
for var, lab in (("nsc", "NSC Mathematics"), ("nbt", "NBT Mathematics")):
    lo, hi = place_ci[var] = boot(e[var], a[var])
    s = sup.dropna(subset=[var]); gm = s[var].mean()
    ssb = sum(len(gg)*(gg[var].mean()-gm)**2 for _, gg in s.groupby("stream"))
    eta = ssb/((s[var]-gm)**2).sum()
    rows.append(f"{lab} & {e[var].mean():.1f} ({e[var].std():.1f}) & {a[var].mean():.1f} "
                f"({a[var].std():.1f}) & {e[var].mean()-a[var].mean():+.1f} & "
                f"[{lo:+.1f}, {hi:+.1f}] & {dcoh(e[var], a[var]):+.2f} & {eta:.3f} \\\\")
w("tab_placement.tex", f"""\\begin{{tabular}}{{lccccrr}}
\\toprule
 & Extended & Augmented & \\multicolumn{{2}}{{c}}{{Difference}} & & \\\\
\\cmidrule(lr){{4-5}}
Measure & ($n={len(e)}$) & ($n={len(a)}$) & mean & 95\\% CI & $d$ & $\\eta^2$ \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")

# ---- inline values ---------------------------------------------------------
b80 = d[d.below80]
vals = {
    "Nthree": len(d), "Ntwo": len(two),
    "PctRecovered": f"{100*(len(d)-len(two))/len(two):.0f}",
    "Nbelowthree": len(b80), "Nbelowtwo": int(two.below80.sum()),
    "PctExtBelow": f"{100*(b80.stream=='extended').mean():.0f}",
    "SDnsc": f"{d.nsc.std():.1f}", "SDnbt": f"{d.nbt.std():.1f}",
    "Rnscnbt": f"{d[['nsc','nbt']].corr().iloc[0,1]:.2f}",
    "PctVar": f"{100*d[['nsc','nbt']].corr().iloc[0,1]**2:.0f}",
}
g2 = g.set_index([g.nsc.astype(int), g.nbt.astype(int)])
for (a_, b_), lab in (((78, 80), "A"), ((85, 60), "B"), ((85, 85), "C")):
    vals[f"Mark{lab}"] = f"{g2.loc[(a_, b_), 'mark']:.1f}"
    vals[f"Pass{lab}"] = f"{g2.loc[(a_, b_), 'pass']:.2f}"
vals["ContrastAB"] = "{:.1f}".format(g2.loc[(78, 80), "mark"] - g2.loc[(85, 60), "mark"])
o = m[(m.stream == "mainstream") & (m.nsc.between(74, 82))].dropna(subset=["nbt", "s1_pass"])
lo_ = o[(o.nbt >= 50) & (o.nbt < 60)]; hi_ = o[o.nbt >= 70]
# placement values as macros so prose cannot drift from the table
for var, key in (("nsc", "NSC"), ("nbt", "NBT")):
    lo2, hi2 = place_ci[var]  # the same draw as Table 8, so table and text cannot differ
    sup2 = sup.dropna(subset=[var]); gm2 = sup2[var].mean()
    ssb2 = sum(len(gg)*(gg[var].mean()-gm2)**2 for _, gg in sup2.groupby("stream"))
    sst2 = ((sup2[var]-gm2)**2).sum(); k2 = sup2.stream.nunique(); n2 = len(sup2)
    msw = (sst2-ssb2)/(n2-k2)
    vals[f"Eta{key}"] = f"{100*ssb2/sst2:.1f}"
    vals[f"Omega{key}"] = f"{100*max((ssb2-(k2-1)*msw)/(sst2+msw), 0):.2f}"
    vals["EtaNull"] = f"{100*(k2-1)/(n2-1):.2f}"
    vals[f"Diff{key}"] = f"{e[var].mean()-a[var].mean():+.1f}"
    vals[f"CI{key}"] = f"[{lo2:+.1f}, {hi2:+.1f}]"
    vals[f"D{key}"] = f"{dcoh(e[var], a[var]):+.2f}"
    vals[f"Ext{key}"] = f"{e[var].mean():.1f}"
    vals[f"Aug{key}"] = f"{a[var].mean():.1f}"
sup3 = d[d.stream.isin(["extended", "augmented", "mainstream"])].dropna(subset=["nsc"])
gm3 = sup3.nsc.mean()
ssb3 = sum(len(gg)*(gg.nsc.mean()-gm3)**2 for _, gg in sup3.groupby("stream"))
vals["EtaNSCall"] = f"{100*ssb3/((sup3.nsc-gm3)**2).sum():.1f}"
cc = m.dropna(subset=["nsc", "s1_pass"])
for cut_a, cut_b, tag in ((78, 80, "Down"), (80, 85, "Up")):
    band = cc[(cc.nsc >= cut_a) & (cc.nsc < cut_b)]
    np_, nf_ = int(band.s1_pass.sum()), int((band.s1_pass == 0).sum())
    vals |= {f"N{tag}": len(band), f"Pass{tag}": np_, f"Fail{tag}": nf_,
             f"Rate{tag}": f"{np_/nf_:.1f}" if nf_ else "no failures"}
rates = []
for k in range(75, 86):
    if k == 80: continue
    lo2, hi2 = (k, 80) if k < 80 else (80, k)
    b2 = cc[(cc.nsc >= lo2) & (cc.nsc < hi2)]
    nf2 = int((b2.s1_pass == 0).sum())
    if len(b2): rates.append(b2.s1_pass.sum() / nf2 if nf2 else np.inf)
vals |= {"RateMin": f"{min(rates):.1f}", "RateMax": f"{max(rates):.1f}"}
vals |= {"MatchExclA": int(((cc.nsc<80)&(cc.s1_pass==1)).sum()),
         "MatchAdmA": int(((cc.nsc>=80)&(cc.s1_pass==0)).sum())}
vals |= vals_extra
vals |= {"ObsLoRate": f"{100*lo_.s1_pass.mean():.0f}", "ObsLoN": str(len(lo_)),
         "ObsHiRate": f"{100*hi_.s1_pass.mean():.0f}", "ObsHiN": str(len(hi_))}
w("values.tex", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in vals.items()))
print("\ninline values:", vals)

# ---- Table 6: change-point -------------------------------------------------
mk2 = m.dropna(subset=["nsc", "s1_mark"]).copy(); pa2 = m.dropna(subset=["nsc", "s1_pass"]).copy()
rows, gains = [], {}
for lab, y, data, fam in (("Semester 1 mark", "s1_mark", mk2, None),
                          ("Semester 1 pass", "s1_pass", pa2, sm.families.Binomial())):
    base = f"{y} ~ stream + cohort"
    b = smf.ols(base, data).fit() if fam is None else smf.glm(base, data, family=fam).fit()
    cells = []
    for cut in (80, 90):
        s = data.copy(); s["jump"] = (s.nsc >= cut).astype(float)
        f0 = f"{y} ~ bs(nsc, df=4) + stream + cohort"
        m0 = smf.ols(f0, s).fit() if fam is None else smf.glm(f0, s, family=fam).fit()
        m1 = (smf.ols(f0 + " + jump", s).fit() if fam is None
              else smf.glm(f0 + " + jump", s, family=fam).fit())
        lr = max(2*(m1.llf - m0.llf), 0)
        fc = f"{y} ~ jump + stream + cohort"
        mc = smf.ols(fc, s).fit() if fam is None else smf.glm(fc, s, family=fam).fit()
        cells += [f"{m1.params.get('jump', float('nan')):+.2f}", f"{lr:.2f}",
                  f"{stats.chi2.sf(lr,1):.2f}", f"{2*(mc.llf-b.llf):.1f}"]
    fs = f"{y} ~ bs(nsc, df=4) + stream + cohort"
    ms2 = smf.ols(fs, data).fit() if fam is None else smf.glm(fs, data, family=fam).fit()
    rows.append(f"{lab} & {len(data)} & " + " & ".join(cells) + f" & {2*(ms2.llf-b.llf):.1f} \\\\")
w("tab_changepoint.tex", """\\begin{tabular}{lrrrrrrrrrr}
\\toprule
 & & \\multicolumn{4}{c}{Discontinuity at 80} & \\multicolumn{4}{c}{Discontinuity at 90} & Smooth \\\\
\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}
Outcome & $n$ & coef. & $\\chi^2_1$ & $p$ & fit & coef. & $\\chi^2_1$ & $p$ & fit & fit \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}""")

# Table 7 (model comparison) is written by scripts/17_model_table.py, which owns the
# leave-one-cohort-out fits and the paired bootstrap. Do not write it from here.

# The admission comparison (tab_admission.tex, values_admission.tex) is written by
# scripts/22_admission_interval.py, so the table and the text's differences and
# intervals come from one set of fitted models. Do not write it from here.
