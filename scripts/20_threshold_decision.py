"""The decision-analytic view of the threshold: what does moving it cost and buy?

Emits values_threshold.tex. Three parts:

  1. Error counts at each candidate cut, and where the two curves cross.
  2. A PERMUTATION TEST on that crossing. Repeating the exercise with the pass outcome randomly
     shuffled tests whether the crossing's position is set by the mark distribution and the base
     rate rather than by who passes. If it is, the crossing is not evidence that a given cut is
     well chosen.
  3. The exchange rate: successful students lost per failure avoided, as the cut moves.

Also computes the standard error on the jump at 80, which is what supports "absence of evidence
for a jump" rather than "evidence there is none". Run from the project root.
"""
import warnings, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
warnings.simplefilter("ignore")
if not Path("anonymised-data").is_dir():
    raise SystemExit("run from the project root")
rng = np.random.default_rng(20260819)
d = pd.read_csv("anonymised-data/paper1_analysis_sample_3streams.csv").rename(
    columns={"Math Grd12 Fin Rslt": "nsc"})
d["cohort"] = d.cohort.astype(str)
c = d[d.stream.isin(["mainstream", "augmented"])].dropna(subset=["nsc", "s1_pass"])
v = {}


def crossing(nsc, y):
    for k in np.arange(70, 96, 0.5):
        if ((nsc < k) & (y == 1)).sum() >= ((nsc >= k) & (y == 0)).sum():
            return k
    return np.nan


# the crossing is a property of the mark distribution, not of its validity
perm = [crossing(c.nsc.values, rng.permutation(c.s1_pass.values)) for _ in range(400)]
v["PermCross"] = f"{np.median(perm):.1f}"
v["PermPct"] = f"{100*np.mean([79 <= x <= 80 for x in perm]):.0f}"
v["Quant"] = f"{np.quantile(c.nsc, 1-c.s1_pass.mean()):.0f}"

# equal weighting minimises total error by admitting everyone in range
tot = {k: int(((c.nsc < k) & (c.s1_pass == 1)).sum() + ((c.nsc >= k) & (c.s1_pass == 0)).sum())
       for k in (70, 80, 85)}
v |= {"TotSeventy": tot[70], "TotEighty": tot[80], "TotEightyfive": tot[85]}
for w_, key in ((1.0, "One"), (0.30, "Three"), (0.25, "Quarter")):
    v[f"OptCut{key}"] = min(range(70, 96), key=lambda k: w_*((c.nsc < k) & (c.s1_pass == 1)).sum()
                            + ((c.nsc >= k) & (c.s1_pass == 0)).sum())

# the discontinuity null needs an interval, not just a p-value
mk = d[d.stream.isin(["mainstream", "augmented"])].dropna(subset=["nsc", "s1_mark"]).copy()
mk["jump"] = (mk.nsc >= 80).astype(float)
fit = smf.ols("s1_mark ~ bs(nsc, df=4) + jump + stream + cohort", mk).fit()
b, se = fit.params["jump"], fit.bse["jump"]
v |= {"JumpSE": f"{se:.2f}", "JumpLo": f"{b-1.96*se:+.1f}", "JumpHi": f"{b+1.96*se:+.1f}",
      "JumpMDE": f"{2.8*se:.1f}"}

# exchange rate: an interval, and the marginal rather than cumulative view
def rate_ci(lo_, hi_):
    band = c[(c.nsc >= lo_) & (c.nsc < hi_)]
    y = band.s1_pass.values
    r = [max(yy.sum(), 0) / max((yy == 0).sum(), 1)
         for yy in (rng.choice(y, len(y)) for _ in range(4000))]
    return np.percentile(r, [2.5, 97.5])
lo_, hi_ = rate_ci(78, 80)
v |= {"RateDownLo": f"{lo_:.1f}", "RateDownHi": f"{hi_:.1f}"}
v["RateSeventyfive"] = f"{c[(c.nsc>=75)&(c.nsc<80)].s1_pass.sum()/max((c[(c.nsc>=75)&(c.nsc<80)].s1_pass==0).sum(),1):.1f}"
Path("paper/tables/values_threshold.tex").write_text(
    "\n".join(f"\\newcommand{{\\{k}}}{{{x}}}" for k, x in v.items()))
print("wrote values_threshold.tex:", v)
