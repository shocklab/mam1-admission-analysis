"""Tie-break sensitivity for the placement result.

Re-runs the loader (scripts/10) under each tie-break rule rather than re-implementing it, so the
check uses exactly the code that built the sample."""
import os, subprocess, sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
warnings.simplefilter("ignore")
if not Path("anonymised-data").is_dir():
    raise SystemExit("run from the project root")


def eta(d, var):
    s = d[d.stream.isin(["extended", "augmented"])].dropna(subset=[var])
    gm = s[var].mean()
    ssb = sum(len(g) * (g[var].mean() - gm) ** 2 for _, g in s.groupby("stream"))
    return 100 * ssb / ((s[var] - gm) ** 2).sum()


res = {}
for tb in ("most", "least"):
    out = f"/tmp/tb_{tb}.csv"
    env = dict(os.environ, TIEBREAK=tb, OUTCSV=out)
    subprocess.run([sys.executable, "scripts/10_build_analysis_sample.py"],
                   env=env, capture_output=True, check=True)
    d = pd.read_csv(out).rename(columns={"Math Grd12 Fin Rslt": "nsc", "NBT Math Score": "nbt"})
    res[tb] = (eta(d, "nsc"), eta(d, "nbt"), len(d))
d = pd.read_csv("/tmp/tb_most.csv").rename(
    columns={"Math Grd12 Fin Rslt": "nsc", "NBT Math Score": "nbt"})
n_amb = int((d.s1_streams_n > 1).sum())
d2 = d[d.s1_streams_n <= 1]
res["exclude"] = (eta(d2, "nsc"), eta(d2, "nbt"), len(d2))
for k, (a, b, n) in res.items():
    print(f"  {k:9s} n={n:5d}   NSC eta2 {a:.2f}%   NBT eta2 {b:.2f}%")
ns = [v[0] for v in res.values()]
nb = [v[1] for v in res.values()]
out = {"TieNamb": n_amb, "TieNSCmin": f"{min(ns):.1f}", "TieNSCmax": f"{max(ns):.1f}",
       "TieNBTmin": f"{min(nb):.1f}", "TieNBTmax": f"{max(nb):.1f}"}
Path("paper/tables/values_tiebreak.tex").write_text(
    "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in out.items()))
print("\nreported rule must match the paper (0.2% / 7.8%):", f"{res['most'][0]:.2f}% / {res['most'][1]:.2f}%")
print("wrote paper/tables/values_tiebreak.tex:", out)
