"""How much do the Section 3.2 window pass rates depend on the window?

Script 13 compares mainstream students with school-leaving Mathematics between 74 and 82 and NBT
Mathematics 50 to 59 against those at 70 or above. The window is centred on 78, the school mark
of the below-threshold profile in the fitted-surface table (school mark 78, NBT 80), and the two
NBT bands bracket the region where the fitted model understated observed pass rates (50 to 59)
and the profile's own NBT score (70 or above). This script repeats the comparison at other
half-widths around 78 and with the window centred on 80, and prints counts and rates only.
Run from the project root.
"""
import pandas as pd
from pathlib import Path
from statsmodels.stats.proportion import proportion_confint
if not Path("anonymised-data").is_dir():
    raise SystemExit("run from the project root")
d = pd.read_csv("anonymised-data/paper1_analysis_sample_3streams.csv").rename(
    columns={"Math Grd12 Fin Rslt": "nsc", "NBT Math Score": "nbt"})
m = d[d.stream == "mainstream"].dropna(subset=["nsc", "nbt", "s1_pass"])


def cell(g):
    k, n = int(g.s1_pass.sum()), len(g)
    if n == 0:
        return "   -"
    lo, hi = proportion_confint(k, n, method="wilson")
    return f"{100*k/n:3.0f}% (n {n:3d}) [{100*lo:.0f}, {100*hi:.0f}]"


# Does the fitted pass model (script 13's specification) still understate observed pass rates in
# the lower NBT band, as the caption of the fitted-surface table says?
import statsmodels.api as sm, statsmodels.formula.api as smf
d["cohort"] = d.cohort.astype(str)
pa = d[d.stream.isin(["mainstream", "augmented"])].dropna(subset=["nsc", "nbt", "s1_pass"])
fp = smf.glm("s1_pass ~ bs(nsc, df=4) + bs(nbt, df=4) + stream + cohort", pa,
             family=sm.families.Binomial()).fit()
o = m[m.nsc.between(74, 82) & (m.nbt >= 50) & (m.nbt < 60)]
o = o.assign(cohort=o.cohort.astype(str))
print(f"74-82, NBT 50-59: observed {100*o.s1_pass.mean():.0f}% (n {len(o)}); "
      f"mean fitted {100*fp.predict(o).mean():.0f}%; fitted at school mark 78, NBT 55, "
      f"mainstream 2025: "
      f"{100*fp.predict(pd.DataFrame({'nsc': [78], 'nbt': [55], 'stream': ['mainstream'], 'cohort': ['2025']}))[0]:.0f}%\n")
print("window          NBT 50-59                     NBT 70+")
for lo_, hi_ in ((76, 80), (75, 81), (74, 82), (73, 83), (72, 84), (77, 83), (76, 84)):
    o = m[m.nsc.between(lo_, hi_)]
    tag = " (as reported)" if (lo_, hi_) == (74, 82) else ""
    print(f"{lo_}-{hi_}{tag:<14} {cell(o[(o.nbt >= 50) & (o.nbt < 60)]):<28}  {cell(o[o.nbt >= 70])}")
