"""Numbers in the paper that no table or values file carries.

Some numbers in the paper are read from figures or computed directly rather than taken from
paper/tables: descriptive statistics, the observed rates in figure 1(b), the error counts behind
figure 3, Welch tests for figure 2, the single-cut fits against the smooth curve, the
held-out-cohort R2 by cohort with the cohort-level t interval, and the offset at the same school
mark. This script recomputes them from an analysis sample. Run from anywhere:

  python3 scripts/36_printed_numbers.py SAMPLE_3STREAMS_CSV OOF_PREDICTIONS_CSV

OOF_PREDICTIONS_CSV is the file scripts/17 writes (anonymised-data/model_oof_predictions.csv);
it must have been written from the same sample. Prints counts, rates and model statistics only.
"""
import sys, warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import r2_score
warnings.simplefilter("ignore")
if len(sys.argv) != 3:
    sys.exit(__doc__)
d = pd.read_csv(sys.argv[1])
pgc = "Population Grp"   # the name scripts/10 guarantees for every cohort
if pgc not in d.columns:
    raise SystemExit(f"no {pgc} column in the analysis sample")
d = d.rename(columns={"Math Grd12 Fin Rslt": "nsc", "Eng Grd12 Fin Rslt": "eng",
                      "Phy Sci Grd12 Fin Rslt": "phys", "NBT Math Score": "nbt",
                      "NBT QL Score": "ql", "NBT AL Score": "al", pgc: "grp"})
two = d[d.stream.isin(["mainstream", "augmented"])].copy()

# ---- descriptives (Results 3.1) --------------------------------------------
q = d.dropna(subset=["nsc", "nbt"])
lo = q[q.nsc < 70]
print(f"three-stream n {len(q)}: school mark SD {q.nsc.std():.2f}, minimum {q.nsc.min():.0f} "
      f"({len(lo)} below 70, cohorts {lo.cohort.value_counts().to_dict()}); "
      f"NBT SD {q.nbt.std():.2f}, range {q.nbt.min():.0f}-{q.nbt.max():.0f}; r {q.nsc.corr(q.nbt):.3f}")

# ---- figure 1(b) and the sparse corner -------------------------------------
pa = two.dropna(subset=["nsc", "nbt", "s1_pass"])
for lab, g in (("below 80", pa[pa.nsc < 80]), ("80 or above", pa[pa.nsc >= 80])):
    cells = [g[(g.nbt >= a) & (g.nbt < b)] for a, b in ((35, 55), (55, 65), (65, 75), (75, 101))]
    print(f"figure 1(b), school mark {lab}: " +
          "; ".join(f"{100*c.s1_pass.mean():.0f}% (n={len(c)})" for c in cells))
s = pa[(pa.stream == "mainstream") & (pa.cohort == 2025)]
print(f"mainstream 2025, school mark below 80 and NBT below 60: "
      f"{int(((s.nsc < 80) & (s.nbt < 60)).sum())}")

# ---- error counts by cut (figure 3) ----------------------------------------
c = two.dropna(subset=["nsc", "s1_pass"])
print("cut  successful-excluded  admitted-who-fail")
for k in range(76, 83):
    print(f"{k:4d}  {int(((c.nsc < k) & (c.s1_pass == 1)).sum()):5d}  "
          f"{int(((c.nsc >= k) & (c.s1_pass == 0)).sum()):5d}")
print(f"pass rate {100*c.s1_pass.mean():.1f}% of {len(c)}; admitted at 80: {100*(c.nsc >= 80).mean():.1f}%")

# ---- support streams: Welch t (Results 3.5 and figure 2 labels) ------------
e_, a_ = d[d.stream == "extended"], d[d.stream == "augmented"]
for v in ("nsc", "nbt"):
    r = stats.ttest_ind(e_[v].dropna(), a_[v].dropna(), equal_var=False)
    print(f"extended against augmented, {v}: Welch t = {r.statistic:.2f}, p = {r.pvalue:.2g}")

# ---- single cuts against the smooth curve (Results 3.3) --------------------
mk = two.dropna(subset=["nsc", "s1_mark"]).copy(); mk["cohort"] = mk.cohort.astype(str)
b0 = smf.ols("s1_mark ~ stream + cohort", mk).fit()
fits = {}
for k in range(75, 96):
    mk["cut"] = (mk.nsc >= k).astype(float)
    fits[k] = 2*(smf.ols("s1_mark ~ cut + stream + cohort", mk).fit().llf - b0.llf)
smooth = 2*(smf.ols("s1_mark ~ bs(nsc, df=4) + stream + cohort", mk).fit().llf - b0.llf)
best = max(fits, key=fits.get)
print(f"fit over stream+cohort: cut at 80 {fits[80]:.1f}, best cut {best} {fits[best]:.1f}, "
      f"smooth {smooth:.1f}; cut at 80 keeps {100*fits[80]/smooth:.0f}% of the smooth fit")

# ---- held-out cohort R2 and the cohort-level t interval (Results 3.4) ------
r6 = two.dropna(subset=["nsc", "eng", "phys", "nbt", "ql", "al", "s1_mark"]).reset_index(drop=True)
o = pd.read_csv(sys.argv[2])
if len(o) != len(r6) or not np.allclose(o.y.values, r6.s1_mark.values):
    sys.exit("the cached predictions were not written from this sample")
L, RG = "Linear, main effects only", "Ridge (splines + pairwise)"
models = [L, RG, "Elastic net (splines + pairwise)", "Random forest", "Gradient boosting"]
cohorts = sorted(r6.cohort.unique())
per = {m: [r2_score(o.y[r6.cohort == k], o[m][r6.cohort == k]) for k in cohorts] for m in models}
print(f"held-out cohort R2, cohorts {[int(k) for k in cohorts]} (pooled R2 to four places):")
for m in models:
    print(f"  {m:34s} " + "  ".join(f"{v:+.3f}" for v in per[m]) + f"   pooled {r2_score(o.y, o[m]):.4f}")
for a in ("Random forest", "Gradient boosting"):
    for ref in (L, RG):
        diff = np.array(per[a]) - np.array(per[ref])
        h = stats.t.ppf(.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
        print(f"  {a} minus {ref}: by cohort {np.round(diff, 3).tolist()}, mean {diff.mean():+.3f}, "
              f"t interval [{diff.mean()-h:+.3f}, {diff.mean()+h:+.3f}]")
print("mean semester-one mark by cohort: " + str(r6.groupby("cohort").s1_mark.mean().round(1).to_dict()))

# ---- the offset at the same school mark (Results 3.5, Cleary) --------------
big = [g for g, n in two.grp.value_counts().items() if n >= 40]
two["grp2"] = np.where(two.grp.isin(big), two.grp, "Other or not recorded")
g = two.dropna(subset=["nsc", "nbt", "s1_mark", "grp2"]).copy()
g["afr"] = (g.grp2 == "African").astype(int); g["cohort"] = g.cohort.astype(str)
for v in ("nsc", "nbt"):
    f = smf.ols(f"s1_mark ~ {v} + afr + stream + cohort", g).fit()
    print(f"African students at the same {v}: {f.params['afr']:+.1f} marks (SE {f.bse['afr']:.1f}), n {len(g)}")
