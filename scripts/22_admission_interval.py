"""How precise is the matched-volume admission null, and how much headroom is there?

Emits tab_admission.tex, values_admission.tex and values_admission2.tex. It is the only script
that fits the admission models, so the table and the text use the same predictions.

The interval refits the models and re-selects students within every bootstrap replicate, so each
replicate keeps the matched-volume constraint.

Also computes the ORACLE: how many passers a rule with perfect foresight would admit at the same
volume. That separates "there is nothing to gain" from "these measures capture none of what
there is to gain". Run from the project root.
"""
import warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
warnings.simplefilter("ignore")
if not Path("anonymised-data").is_dir():
    raise SystemExit("run from the project root")
rng = np.random.default_rng(20260819)
d = pd.read_csv("anonymised-data/paper1_analysis_sample_3streams.csv").rename(
    columns={"Math Grd12 Fin Rslt": "nsc", "Eng Grd12 Fin Rslt": "eng",
             "Phy Sci Grd12 Fin Rslt": "phys", "NBT Math Score": "nbt",
             "NBT QL Score": "ql", "NBT AL Score": "al"})
P6 = ["nsc", "eng", "phys", "nbt", "ql", "al"]
c = d[d.stream.isin(["mainstream", "augmented"])].dropna(subset=P6 + ["s1_pass"]).copy()


def loco(data, feats):
    s = pd.Series(index=data.index, dtype=float)
    for co in sorted(data.cohort.unique()):
        tr, te = data[data.cohort != co], data[data.cohort == co]
        if tr.s1_pass.nunique() < 2:
            continue
        m = Pipeline([("sc", StandardScaler()),
                      ("lr", LogisticRegressionCV(Cs=6, cv=4, max_iter=3000,
                                                  scoring="neg_log_loss"))])
        m.fit(tr[feats], tr.s1_pass)
        s.loc[te.index] = m.predict_proba(te[feats])[:, 1]
    return s


def passers(data, score=None):
    tot = 0
    for co, g in data.groupby("cohort"):
        n_c = int((g.nsc >= 80).sum())
        sel = (g.nsc >= 80) if score is None else (g[score].rank(ascending=False,
                                                                 method="first") <= n_c)
        tot += int(g.loc[sel, "s1_pass"].sum())
    return tot


v = {}
c["p_both"] = loco(c, ["nsc", "nbt"]); c["p_full"] = loco(c, P6)
base = passers(c)
v["AdmBase"] = base
for col, tag in (("p_both", "Both"), ("p_full", "Full")):
    v[f"Adm{tag}Diff"] = f"{passers(c, col) - base:+d}"
diffs = {"p_both": [], "p_full": []}
print("refit-and-reselect bootstrap, 300 replicates (slow by design):")
for i in range(300):
    b = pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1e9)))
                   for _, g in c.groupby("cohort")]).reset_index(drop=True)
    try:
        b["p_both"] = loco(b, ["nsc", "nbt"]); b["p_full"] = loco(b, P6)
        bb = passers(b)
        for col in diffs:
            diffs[col].append(passers(b, col) - bb)
    except Exception:
        continue
for col, tag in (("p_both", "Both"), ("p_full", "Full")):
    lo, hi = np.percentile(diffs[col], [2.5, 97.5])
    v[f"Adm{tag}CI"] = f"[{lo:+.0f}, {hi:+.0f}]"
    print(f"  {col}: mean {np.mean(diffs[col]):+.1f}  95% CI [{lo:+.0f}, {hi:+.0f}]  "
          f"({len(diffs[col])} reps)")
# Table 5: admitted, passers admitted and successful students excluded, from the same fits
def selected(data, score=None):
    sel = pd.Series(False, index=data.index)
    for co, g in data.groupby("cohort"):
        n_c = int((g.nsc >= 80).sum())
        sel.loc[g.index[(g.nsc >= 80) if score is None
                        else (g[score].rank(ascending=False, method="first") <= n_c)]] = True
    return sel


arows = []
for lab, col in (("School mark $\\geq 80$ (the proposed rule)", None),
                 ("School mark and benchmark test", "p_both"),
                 ("All six measures", "p_full")):
    sel = selected(c, col)
    assert int(c[sel].s1_pass.sum()) == passers(c, col)
    arows.append(f"{lab} & {int(sel.sum())} & {int(c[sel].s1_pass.sum())} & "
                 f"{int(c[~sel].s1_pass.sum())} \\\\")
Path("paper/tables/tab_admission.tex").write_text("""\\begin{tabular}{lrrr}
\\toprule
Rule & Admitted & Passers admitted & Successful students excluded \\\\
\\midrule
""" + "\n".join(arows) + """
\\bottomrule
\\end{tabular}""")
va = {"AdmN": int((c.nsc >= 80).sum()), "AdmRate": f"{100*(c.nsc >= 80).mean():.0f}",
      "BaseRate": f"{100*c.s1_pass.mean():.1f}", "AdmTotal": len(c)}
Path("paper/tables/values_admission.tex").write_text(
    "\n".join(f"\\newcommand{{\\{k}}}{{{x}}}" for k, x in va.items()))
print("Table 5:", *arows, sep="\n  ")
print("values_admission:", va)
# oracle ceiling at the same volume
orc = 0
for co, g in c.groupby("cohort"):
    n_c = int((g.nsc >= 80).sum())
    orc += int(g.sort_values("s1_pass", ascending=False).head(n_c).s1_pass.sum())
v |= {"AdmOracle": orc, "AdmHead": orc - base}
# out-of-sample ranking gain
au = []
for co, g in c.groupby("cohort"):
    au.append((roc_auc_score(g.s1_pass, g.nsc), roc_auc_score(g.s1_pass, g.p_both)))
gains = [b - a for a, b in au]
bs = [np.mean(rng.choice(gains, len(gains))) for _ in range(20000)]
v |= {"AucNSC": f"{np.mean([a for a, _ in au]):.3f}", "AucBoth": f"{np.mean([b for _, b in au]):.3f}",
      "AucGain": f"{np.mean(gains):+.3f}"}
Path("paper/tables/values_admission2.tex").write_text(
    "\n".join(f"\\newcommand{{\\{k}}}{{{x}}}" for k, x in v.items()))
print("\nvalues:", v)
