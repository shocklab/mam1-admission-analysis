"""Bounds on the two results that rest on pending results: the entry-course gap and the
two-semester test.

scripts/10 resolves a pending result (a mark with S, DE, OSS) from the next year's enrolment and
leaves out the ones it cannot resolve. The paper also reports the results with those counted
both ways, so this re-runs scripts/10 itself with UNRESOLVED=fail and UNRESOLVED=pass (writing the
samples to a temporary folder, never over anonymised-data/) and recomputes:

  - the entry-course gap, extended minus augmented, 2024 and 2025 (scripts/21, EntryDiff), and the
    same for students with a single semester-one stream record (SetEntryDiff);
  - the likelihood-ratio test of the NBT over the school mark for passing both semesters
    (scripts/27, TwoSemChiBoth).

Before reporting the bounds it checks that the same code applied to the sample as built
reproduces the point estimates in paper/tables, so the bounds are computed the same way as the
reported values. Emits values_pending.tex. Run from the project root.
"""
import os, re, subprocess, sys, tempfile, warnings
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from pathlib import Path
from scipy.stats import chi2
warnings.simplefilter("ignore")
if not Path("anonymised-data").is_dir():
    raise SystemExit("run from the project root")
T = Path("paper/tables")


def macros(path):
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (T/path).read_text()))


def entry_gap(d, rng, single=False):
    s = d[d.s1_pass.notna() & (d.cohort < 2026)]
    if single:
        s = s[s.s1_streams_n <= 1]
    e, a = s[s.stream == "extended"].s1_pass.values, s[s.stream == "augmented"].s1_pass.values
    bs = [100*(rng.choice(e, len(e)).mean() - rng.choice(a, len(a)).mean()) for _ in range(20000)]
    return 100*(e.mean() - a.mean()), *np.percentile(bs, [2.5, 97.5]), len(e), len(a)


def two_semester(d):
    a = d[d.stream.isin(["mainstream", "augmented"]) & d.cohort.isin([2024, 2025])].dropna(
        subset=["nsc", "nbt"]).copy()
    a = a[~((a.s1_pending & a.s1_pass.isna()) | (a.s2_pending & a.s2_pass.isna()))].copy()
    a["cohort"] = a.cohort.astype(str)
    a["both"] = ((a.s1_pass == 1) & (a.s2_pass == 1)).astype(int)
    m0 = smf.glm("both ~ nsc + C(cohort) + stream", a, family=sm.families.Binomial()).fit()
    m1 = smf.glm("both ~ nsc + nbt + C(cohort) + stream", a, family=sm.families.Binomial()).fit()
    lr = 2*(m1.llf - m0.llf)
    return lr, chi2.sf(lr, 1), len(a)


def load(path):
    return pd.read_csv(path).rename(columns={"Math Grd12 Fin Rslt": "nsc", "NBT Math Score": "nbt"})


samples = {"as built": load("anonymised-data/paper1_analysis_sample_3streams.csv")}
tmp = Path(tempfile.mkdtemp(prefix="pending_bounds_"))
for rule in ("fail", "pass"):
    out = tmp / f"sample_{rule}_3streams.csv"
    env = dict(os.environ, UNRESOLVED=rule, OUTCSV=str(out))
    r = subprocess.run([sys.executable, "scripts/10_build_analysis_sample.py"], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"scripts/10 failed with UNRESOLVED={rule}:\n{r.stderr[-2000:]}")
    print(next(l for l in r.stdout.splitlines() if l.startswith("Pending results")))
    samples[f"unresolved {rule}"] = load(out)

res = {}
for lab, d in samples.items():
    rng = np.random.default_rng(20260819)
    res[lab] = (entry_gap(d, rng), entry_gap(d, rng, single=True), two_semester(d))

# the same code on the sample as built must give the reported point estimates
pl, ts = macros("values_placement.tex"), macros("values_twosem.tex")
(g, _, _, _, _), (gs, _, _, _, _), (lr, _, _) = res["as built"]
assert f"{g:+.1f}" == pl["EntryDiff"], (g, pl["EntryDiff"])
assert f"{gs:+.1f}" == pl["SetEntryDiff"], (gs, pl["SetEntryDiff"])
assert f"{lr:.2f}" == ts["TwoSemChiBoth"], (lr, ts["TwoSemChiBoth"])
print("reproduces EntryDiff, SetEntryDiff and TwoSemChiBoth from paper/tables\n")

v = {}
for lab, ((g, lo, hi, ne, na), (gs, slo, shi, sne, sna), (lr, p, n)) in res.items():
    print(f"{lab:16s} entry gap {g:+.1f} [{lo:+.1f}, {hi:+.1f}] (extended {ne}, augmented {na}); "
          f"single record {gs:+.1f} [{slo:+.1f}, {shi:+.1f}] ({sne}, {sna}); "
          f"two semesters chi2 {lr:.2f}, p {p:.3f}, n {n}")
    if lab == "as built":
        continue
    k = "Fail" if lab.endswith("fail") else "Pass"
    v |= {f"PendEntry{k}": f"{g:+.1f}", f"PendEntry{k}CI": f"[{lo:+.1f}, {hi:+.1f}]",
          f"PendSetEntry{k}": f"{gs:+.1f}", f"PendSetEntry{k}CI": f"[{slo:+.1f}, {shi:+.1f}]",
          f"PendTwoSemChi{k}": f"{lr:.2f}", f"PendTwoSemP{k}": f"{p:.3f}", f"PendTwoSemN{k}": n}
(T/"values_pending.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{x}}}" for k, x in v.items()) + "\n")
print(f"\nwrote {T/'values_pending.tex'}")
