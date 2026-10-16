"""Does the benchmark test's advantage survive to the full year?

Outcome (i)  passed the semester-1 course.
Outcome (ii) passed a semester-1 course and then a semester-2 course of the mainstream or
             augmented streams, in any combination (a student can move from augmented in
             semester 1 to mainstream in semester 2). A semester-1 pass with no semester-2
             record counts as not passing both. Extended is excluded: MAM1005H runs a full year
             and is not a semester outcome. 2026 is excluded because its second-semester records
             were not final when the data were extracted.

Students whose semester-1 result is missing (no semester-1 record, or a semester-1 record with
a blank or unrecognised grade) are left out of both outcomes, as is a semester-1 passer whose
semester-2 grade is unrecognised: a missing result is not a fail. Holm-adjusted p-values treat
the two tests as one family.
"""
import warnings, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import chi2
from pathlib import Path
warnings.simplefilter("ignore")
if not Path("anonymised-data").is_dir():
    raise SystemExit("run from the project root")
T = Path("paper/tables"); T.mkdir(parents=True, exist_ok=True)

d = pd.read_csv("anonymised-data/paper1_analysis_sample_3streams.csv").rename(
    columns={"Math Grd12 Fin Rslt": "nsc", "NBT Math Score": "nbt"})
a = d[d.stream.isin(["mainstream", "augmented"]) & d.cohort.isin([2024, 2025])].dropna(
    subset=["nsc", "nbt"]).copy()
# A result still pending after scripts/10's enrolment rule is unknown, not a fail: leave the
# student out rather than count them as not passing.
unres = (a.s1_pending & a.s1_pass.isna()) | (a.s2_pending & a.s2_pass.isna())
print(f"left out with an unresolved pending result: {int(unres.sum())}")
a = a[~unres].copy()
unknown = a.s1_pass.isna() | ((a.s1_pass == 1) & a.s2_course.notna() & a.s2_pass.isna())
print(f"left out with a missing result: {int(unknown.sum())} "
      f"({int((a.s1_pass.isna() & a.s1_course.isna()).sum())} with no semester-1 record)")
a = a[~unknown].copy()
a["cohort"] = a.cohort.astype(str)
a["s1"]   = a.s1_pass.astype(int)
a["both"] = ((a.s1_pass == 1) & (a.s2_pass == 1)).astype(int)
print(f"n = {len(a)}  ({(a.stream=='mainstream').sum()} mainstream, "
      f"{(a.stream=='augmented').sum()} augmented; 2024 and 2025 only)")

vals, pv = {"TwoSemN": len(a)}, {}
for y, key, lab in (("s1", "S1", "passed the semester-1 course"),
                    ("both", "Both", "passed both semesters")):
    m0 = smf.glm(f"{y} ~ nsc + C(cohort) + stream", a, family=sm.families.Binomial()).fit()
    m1 = smf.glm(f"{y} ~ nsc + nbt + C(cohort) + stream", a, family=sm.families.Binomial()).fit()
    lr = 2 * (m1.llf - m0.llf); p = chi2.sf(lr, 1)
    vals[f"TwoSemChi{key}"] = f"{lr:.2f}"
    vals[f"TwoSemP{key}"]   = f"{p:.4f}"
    pv[key] = p
    print(f"  NBT adds over the school mark, {lab:<30} chi2(1) = {lr:5.2f}, p = {p:.4f}, "
          f"coef {m1.params['nbt']:+.4f} (SE {m1.bse['nbt']:.4f})")

# Holm adjustment with the two tests above as one family, as the manuscript reports it
lo_k, hi_k = sorted(pv, key=pv.get)
holm = {lo_k: min(1.0, 2 * pv[lo_k])}
holm[hi_k] = max(holm[lo_k], pv[hi_k])
for key, h in holm.items():
    vals[f"TwoSemHolm{key}"] = f"{h:.4f}"
print(f"  Holm-adjusted: S1 {holm['S1']:.4f}, both {holm['Both']:.4f}")

# conditional on having passed S1, does the NBT still say anything about S2?
c = a[(a.s1_pass == 1) & a.s2_pass.notna()].copy(); c["y"] = c.s2_pass.astype(int)
n0 = smf.glm("y ~ C(cohort) + stream", c, family=sm.families.Binomial()).fit()
nsc_only = smf.glm("y ~ nsc + C(cohort) + stream", c, family=sm.families.Binomial()).fit()
both_    = smf.glm("y ~ nsc + nbt + C(cohort) + stream", c, family=sm.families.Binomial()).fit()
print(f"\n  among the {len(c)} who passed semester 1 and have a semester-2 result:")
print(f"    school mark over cohort+stream : chi2(1) = {2*(nsc_only.llf-n0.llf):5.2f}")
print(f"    benchmark added on top of that : chi2(1) = {2*(both_.llf-nsc_only.llf):5.2f}")
(T/"values_twosem.tex").write_text(
    "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in vals.items()) + "\n")
print(f"\nwrote {T/'values_twosem.tex'}: {vals}")
