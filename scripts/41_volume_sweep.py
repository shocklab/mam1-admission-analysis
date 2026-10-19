"""Matched-volume admission comparison across a range of admission volumes.

Script 22 compares the rules at one volume only, the one the proposed 80% rule implies (598 of
710). This script repeats the comparison at more and less selective volumes. At a more selective
volume the marginal student is less likely to pass, so a better ranking has more room to help.

At each volume q, every cohort admits round(q * n_cohort) students, ranked by
  - the school-leaving Mathematics mark (the school-mark rule),
  - out-of-fold probabilities from school mark + NBT Mathematics (models fitted on the other two
    cohorts, as in script 22),
  - out-of-fold probabilities from all six measures,
  - the outcome itself (the oracle, the ceiling at that volume).
The school mark is an integer, so many students tie at the boundary. Ties are broken at random
and the expected number of passers is reported, which is why those counts carry a decimal. At
the volume the 80% rule implies, the boundary falls between 79 and 80 and no tie is broken.

Intervals refit and reselect inside every replicate (the same design as script 22's), resampling
students within cohort. All volumes in a replicate share one set of fits, so the curves are
comparable with each other. Prints aggregate counts only. Run from the project root.
"""
import warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
warnings.simplefilter("ignore")
if not Path("anonymised-data").is_dir():
    raise SystemExit("run from the project root")
rng = np.random.default_rng(20261019)
d = pd.read_csv("anonymised-data/paper1_analysis_sample_3streams.csv").rename(
    columns={"Math Grd12 Fin Rslt": "nsc", "Eng Grd12 Fin Rslt": "eng",
             "Phy Sci Grd12 Fin Rslt": "phys", "NBT Math Score": "nbt",
             "NBT QL Score": "ql", "NBT AL Score": "al"})
P6 = ["nsc", "eng", "phys", "nbt", "ql", "al"]
c = d[d.stream.isin(["mainstream", "augmented"])].dropna(subset=P6 + ["s1_pass"]).copy()
VOLS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, None, 0.90]   # None = the 80% rule's volume
N_REP = 300


def loco(data, feats):
    s = pd.Series(index=data.index, dtype=float)
    for co in sorted(data.cohort.unique()):
        tr, te = data[data.cohort != co], data[data.cohort == co]
        if tr.s1_pass.nunique() < 2:
            raise ValueError(f"holding out {co}: the training cohorts have one outcome class")
        m = Pipeline([("sc", StandardScaler()),
                      ("lr", LogisticRegressionCV(Cs=6, cv=4, max_iter=3000,
                                                  scoring="neg_log_loss"))])
        m.fit(tr[feats], tr.s1_pass)
        s.loc[te.index] = m.predict_proba(te[feats])[:, 1]
    return s


def quota(g, q):
    return int((g.nsc >= 80).sum()) if q is None else int(round(q * len(g)))


def expected_passers(score, y, k):
    """Expected passers among the top k by score, ties at the boundary broken at random."""
    if k <= 0:
        return 0.0
    if score.isna().any():
        raise ValueError("a student has no score")
    t = score.sort_values(ascending=False).iloc[k - 1]
    above, tied = score > t, score == t
    a = int(above.sum())
    return float(y[above].sum()) + (k - a) * float(y[tied].sum()) / int(tied.sum())


def passers(data, col, q):
    tot = 0.0
    for _, g in data.groupby("cohort"):
        k = quota(g, q)
        tot += (float(g.s1_pass.sort_values(ascending=False).head(k).sum()) if col == "oracle"
                else expected_passers(g[col], g.s1_pass, k))
    return tot


def curves(data):
    data = data.copy()
    data["p_both"] = loco(data, ["nsc", "nbt"]); data["p_full"] = loco(data, P6)
    return {q: {col: passers(data, col, q) for col in ("nsc", "p_both", "p_full", "oracle")}
            for q in VOLS}


point = curves(c)
# the matched volume must reproduce script 22 (526 passers for the school-mark rule)
assert point[None]["nsc"] == float(((c.nsc >= 80) & (c.s1_pass == 1)).sum())
reps = {q: {"p_both": [], "p_full": []} for q in VOLS}
print(f"refit-and-reselect bootstrap, {N_REP} replicates:")
for i in range(N_REP):
    b = pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1e9)))
                   for _, g in c.groupby("cohort")]).reset_index(drop=True)
    cb = curves(b)
    for q in VOLS:
        for col in ("p_both", "p_full"):
            reps[q][col].append(cb[q][col] - cb[q]["nsc"])

rows, out = [], []
print(f"\nn = {len(c)}; pass rate {100*c.s1_pass.mean():.1f}%")
print("volume  admitted  school  +NBT  diff [95% CI]       six  diff [95% CI]       oracle")
for q in VOLS:
    adm = sum(quota(g, q) for _, g in c.groupby("cohort"))
    p = point[q]; lab = f"{100*adm/len(c):.1f}" if q is None else f"{100*q:.0f}"
    cis = {col: np.percentile(reps[q][col], [2.5, 97.5]) for col in ("p_both", "p_full")}
    db, df = p["p_both"] - p["nsc"], p["p_full"] - p["nsc"]
    print(f"{lab:>6}  {adm:8d}  {p['nsc']:6.1f}  {p['p_both']:4.0f}  {db:+5.1f} "
          f"[{cis['p_both'][0]:+.0f}, {cis['p_both'][1]:+.0f}]  {p['p_full']:6.0f}  {df:+5.1f} "
          f"[{cis['p_full'][0]:+.0f}, {cis['p_full'][1]:+.0f}]  {p['oracle']:6.0f}")
    out.append(f"{lab} & {adm} & {p['nsc']:.1f} & {db:+.1f} & "
               f"[{cis['p_both'][0]:+.0f}, {cis['p_both'][1]:+.0f}] & {df:+.1f} & "
               f"[{cis['p_full'][0]:+.0f}, {cis['p_full'][1]:+.0f}] & {p['oracle']:.0f} \\\\")
Path("paper/tables/tab_volume_sweep.tex").write_text("""\\begin{tabular}{rrrrrrrr}
\\toprule
 & & \\multicolumn{1}{c}{School mark} & \\multicolumn{2}{c}{School mark and NBT} & \\multicolumn{2}{c}{All six measures} & \\\\
\\cmidrule(lr){3-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}
Admitted (\\%) & $n$ & passers & difference & 95\\% CI & difference & 95\\% CI & Oracle \\\\
\\midrule
""" + "\n".join(out) + """
\\bottomrule
\\end{tabular}""")
print("\nwrote paper/tables/tab_volume_sweep.tex")
