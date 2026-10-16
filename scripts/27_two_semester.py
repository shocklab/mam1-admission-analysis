"""Does the benchmark test's advantage survive to the full year?

Outcome (i)  passed the semester-1 course.
Outcome (ii) passed BOTH semesters of the same sequence (mainstream 1031F+1032S,
             augmented 1033F+1034S). Extended is excluded: MAM1005H runs a full year and
             is not a semester outcome. 2026 is excluded because its second-semester records
             were not final when the data were extracted.
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
a["cohort"] = a.cohort.astype(str)
a["s1"]   = (a.s1_pass == 1).astype(int)
a["both"] = ((a.s1_pass == 1) & (a.s2_pass == 1)).astype(int)
print(f"n = {len(a)}  ({(a.stream=='mainstream').sum()} mainstream, "
      f"{(a.stream=='augmented').sum()} augmented; 2024 and 2025 only)")

vals = {"TwoSemN": len(a)}
for y, key, lab in (("s1", "S1", "passed the semester-1 course"),
                    ("both", "Both", "passed both semesters")):
    m0 = smf.glm(f"{y} ~ nsc + C(cohort) + stream", a, family=sm.families.Binomial()).fit()
    m1 = smf.glm(f"{y} ~ nsc + nbt + C(cohort) + stream", a, family=sm.families.Binomial()).fit()
    lr = 2 * (m1.llf - m0.llf); p = chi2.sf(lr, 1)
    vals[f"TwoSemChi{key}"] = f"{lr:.2f}"
    vals[f"TwoSemP{key}"]   = f"{p:.4f}"
    print(f"  NBT adds over the school mark, {lab:<30} chi2(1) = {lr:5.2f}, p = {p:.4f}, "
          f"coef {m1.params['nbt']:+.4f} (SE {m1.bse['nbt']:.4f})")

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
