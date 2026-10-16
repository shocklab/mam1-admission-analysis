"""The pre-specified model comparison: do flexible learners beat linear regression here?

Emits tab_models.tex. Two design choices carry the result:

  1. VALIDATION. Leave-one-cohort-out is the primary design, not pooled k-fold, because a model
     for admissions is used on a future cohort it has never seen. Cohort is also excluded as a
     predictor for the same reason: a future cohort's coefficient cannot be known. Pooled figures
     are printed underneath to show how much those two choices change performance.
  2. THE COMPARATOR. The linear model must be a fair opponent, so the regularised linear models
     are given spline bases plus pairwise products of the predictors, and the comparator is
     whichever of the three linear models has the highest held-out-year R2, as the pre-specified
     protocol requires.

The paired bootstrap resamples students while holding the fitted models fixed, so it measures
evaluation-set noise and not the variability of refitting. With three cohorts that second source
is large, which is why the paper also reports a cohort-level t interval.
Run from the project root.
"""
import warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
warnings.simplefilter("ignore")
T = Path("paper/tables")
d = pd.read_csv("anonymised-data/paper1_analysis_sample_3streams.csv").rename(
    columns={"Math Grd12 Fin Rslt": "nsc", "Eng Grd12 Fin Rslt": "eng",
             "Phy Sci Grd12 Fin Rslt": "phys", "NBT Math Score": "nbt",
             "NBT QL Score": "ql", "NBT AL Score": "al"})
d = d[d.stream.isin(["mainstream", "augmented"])]
P6 = ["nsc", "eng", "phys", "nbt", "ql", "al"]
r = d.dropna(subset=P6 + ["s1_mark"]).copy()


class SplinesPlusPairs(BaseEstimator, TransformerMixin):
    def __init__(self): self.sp = SplineTransformer(n_knots=4, degree=3)
    def fit(self, X, y=None): self.sp.fit(X); return self
    def transform(self, X):
        X = np.asarray(X); S = self.sp.transform(X)
        pr = [X[:, i]*X[:, j] for i in range(X.shape[1]) for j in range(i+1, X.shape[1])]
        return np.hstack([S, np.column_stack(pr)])


def tuned(e, g): return GridSearchCV(e, g, cv=KFold(5, shuffle=True, random_state=0), n_jobs=-1)
MODELS = {
    "Linear, main effects only": lambda: Pipeline([("sc", StandardScaler()), ("rg", RidgeCV(alphas=np.logspace(-2, 4, 40)))]),
    "Ridge (splines + pairwise)": lambda: Pipeline([("f", SplinesPlusPairs()), ("sc", StandardScaler()), ("rg", RidgeCV(alphas=np.logspace(-2, 4, 40)))]),
    "Elastic net (splines + pairwise)": lambda: Pipeline([("f", SplinesPlusPairs()), ("sc", StandardScaler()), ("en", ElasticNetCV(l1_ratio=[.1, .5, .9], n_alphas=60, cv=5, max_iter=20000))]),
    "Random forest": lambda: tuned(RandomForestRegressor(random_state=0), {"n_estimators": [400], "max_depth": [None, 6], "min_samples_leaf": [1, 5, 15]}),
    "Gradient boosting": lambda: tuned(GradientBoostingRegressor(random_state=0), {"n_estimators": [200, 500], "learning_rate": [0.03, 0.1], "max_depth": [2, 3]}),
}
oof, within = {}, {}
for name, mk_ in MODELS.items():
    p = pd.Series(index=r.index, dtype=float)
    for co in sorted(r.cohort.unique()):
        tr, te = r[r.cohort != co], r[r.cohort == co]
        m = mk_(); m.fit(tr[P6], tr["s1_mark"]); p.loc[te.index] = m.predict(te[P6])
    oof[name] = p
    within[name] = np.mean([r2_score(g.s1_mark, p.loc[g.index]) for _, g in r.groupby("cohort")])
    print(f"  {name:34s} {r2_score(r.s1_mark, p):+.3f}  within {within[name]:+.3f}")

y = r.s1_mark.values
LINEAR = ["Linear, main effects only", "Ridge (splines + pairwise)", "Elastic net (splines + pairwise)"]
base = max(LINEAR, key=lambda n: r2_score(y, oof[n]))
print(f"\ncomparator, the best regularised linear model by held-out-year R2: {base}")
print("\npaired bootstrap, 10000 reps x 5 outer seeds:")
ci = {}
for name in ("Random forest", "Gradient boosting"):
    allb = []
    for seed in range(5):
        g = np.random.default_rng(1000 + seed)
        idx = g.integers(0, len(y), (10000, len(y)))
        allb += [r2_score(y[i], oof[name].values[i]) - r2_score(y[i], oof[base].values[i]) for i in idx[:2000]]
    lo, hi = np.percentile(allb, [2.5, 97.5])
    ci[name] = (np.mean(allb), lo, hi)
    print(f"  {name:20s} dR2 {np.mean(allb):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

cvp = {}
for name in ("Elastic net (splines + pairwise)", "Random forest", "Gradient boosting"):
    cvp[name] = r2_score(r.s1_mark, cross_val_predict(MODELS[name](), r[P6], r["s1_mark"],
                                                      cv=KFold(5, shuffle=True, random_state=42)))
# and the same design WITH cohort included as a predictor.
rc = pd.get_dummies(r[P6 + ["cohort"]], columns=["cohort"], drop_first=True).astype(float)
cvp["Elastic net, cohort included"] = r2_score(
    r.s1_mark, cross_val_predict(MODELS["Elastic net (splines + pairwise)"](), rc, r["s1_mark"],
                                 cv=KFold(5, shuffle=True, random_state=42)))
print("  with cohort:", round(cvp["Elastic net, cohort included"], 3))
rows = []
for name in MODELS:
    pt = r2_score(y, oof[name]) - r2_score(y, oof[base])
    dr = "reference" if name == base else f"${pt:+.3f}$"
    cistr = f"$[{ci[name][1]:+.3f}, {ci[name][2]:+.3f}]$" if name in ci else "not bootstrapped"
    rows.append(f"{name} & {r2_score(r.s1_mark, oof[name]):.3f} & {within[name]:.3f} & {dr} & {cistr} \\\\")
extra = "\n".join(f"{n} & {v:.3f} & & & \\\\" for n, v in cvp.items())
(T/"tab_models.tex").write_text("""\\begin{tabular}{lrrcc}
\\toprule
Model & LOCO $R^2$ & within-cohort $R^2$ & $\\Delta R^2$ vs best linear & 95\\% CI \\\\
\\midrule
""" + "\n".join(rows) + """
\\midrule
\\multicolumn{5}{l}{\\emph{For comparison, pooled 5-fold cross-validation:}} \\\\
""" + extra + """
\\bottomrule
\\end{tabular}""")
pd.DataFrame(oof).assign(y=y).to_csv("anonymised-data/model_oof_predictions.csv", index=False)
print(f"\nwrote {T/'tab_models.tex'} from data, and cached OOF predictions")
