"""The three manuscript figures.

Okabe-Ito hues throughout, validated for colour-vision deficiency (worst adjacent pair dE 11.0
under deuteranopia).

Figure 1 (a) the two measures with the proposed cut, (b) observed pass rates by benchmark band.
  Panel (a) is monochrome, so that colour has one meaning in the figure: in panel (b) it marks
  school marks of 80 or above against below 80. The two misallocation quadrants are shaded and
  carry only their count; the explanation is in the caption. Panel (b) shows observed rates with
  Wilson intervals rather than a fitted surface, because the additive spline goes non-monotone
  where the data are thin.

Figure 2 distributions of both measures by stream, with p-values.

Figure 3 the two error counts across candidate thresholds. The crossing is not evidence that 80
  is well chosen: see the permutation test in scripts/20.

Writes into paper/figures/. Run from the project root.
"""
import warnings, numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import statsmodels.api as sm, statsmodels.formula.api as smf
from pathlib import Path
warnings.simplefilter("ignore")
F = Path("paper/figures"); F.mkdir(parents=True, exist_ok=True)
BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
INK, MUTED = "#1a1a1a", "#767676"
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "legend.frameon": False,
})
d = pd.read_csv("anonymised-data/paper1_analysis_sample_3streams.csv").rename(
    columns={"Math Grd12 Fin Rslt": "nsc", "NBT Math Score": "nbt"})
d["cohort"] = d.cohort.astype(str)
two = d[d.stream.isin(["mainstream", "augmented"])].copy()

# ---------------- Figure 1: the plane, and observed rates by band ----------------
qs = d.dropna(subset=["nsc", "nbt"]); med = qs.nbt.median()
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.4)); fig.subplots_adjust(wspace=.30)

# ---------------- (a) the plane, monochrome, with the two quadrants drawn ----------------
a = ax[0]
XL, YL = (69, 101), (28, 101)
a.add_patch(Rectangle((XL[0], med), 80-XL[0], YL[1]-med, fc=VERM, alpha=.10, zorder=0, lw=0))
a.add_patch(Rectangle((80, YL[0]), XL[1]-80, med-YL[0], fc=VERM, alpha=.10, zorder=0, lw=0))
rng = np.random.default_rng(0)
a.scatter(qs.nsc + rng.normal(0, .20, len(qs)), qs.nbt, s=7, c=INK, alpha=.22,
          linewidths=0, zorder=2)
a.axvline(80, color=INK, lw=1.1, ls="--", zorder=3)
a.axhline(med, color=INK, lw=.8, ls=":", zorder=3)
n_excl = len(qs[(qs.nsc < 80) & (qs.nbt > med)]); n_adm = len(qs[(qs.nsc >= 80) & (qs.nbt <= med)])
# All text is kept OFF the data: the quadrants carry only their count, placed in the emptiest
# corner of each shaded block, and both line labels sit outside the axes. The sentence that
# explains what the counts mean belongs in the caption, not on top of 962 points.
SHADE = "#FBEFE6"          # VERM at alpha .10 composited over white
BOX = dict(fc=SHADE, ec="none", pad=1.0)
a.text(70.4, 99.5, f"{n_excl}", fontsize=15, color=VERM, ha="left", va="top",
       zorder=4, bbox=BOX)
# The left quadrant is only 11 marks wide, so this label is kept short enough that its
# background box stays inside the shading and does not spill past the cut line.
a.text(70.4, 92.5, "below the cut,\nabove median", fontsize=6.3, color=VERM,
       ha="left", va="top", linespacing=1.3, zorder=4, bbox=BOX)
a.text(99.6, 40.5, f"{n_adm}", fontsize=15, color=VERM, ha="right", va="bottom",
       zorder=4, bbox=BOX)
a.text(99.6, 39.0, "above the cut,\nbelow the median", fontsize=6.6, color=VERM,
       ha="right", va="top", linespacing=1.3, zorder=4, bbox=BOX)
a.annotate("proposed cut, 80%", xy=(80, YL[1]), xytext=(0, 4), textcoords="offset points",
           fontsize=7.4, color=INK, ha="center", va="bottom", annotation_clip=False)
a.set(xlabel="School-leaving Mathematics (%)", ylabel="NBT Mathematics", xlim=XL, ylim=YL)
# The median is a labelled tick rather than floating text, so nothing sits over the data.
a.set_yticks([30, 40, 50, 60, int(med), 80, 90, 100])
a.set_yticklabels(["30", "40", "50", "60", f"{med:.0f}", "80", "90", "100"])
[lb.set_color(INK) for lb in a.get_yticklabels() if lb.get_text() == f"{med:.0f}"]
# No in-plot label for the median: the tick at 71 marks it and the caption names it.
# Any text placed near the line would sit over data.
a.set_title("(a) Where the two measures disagree", loc="left", fontsize=9, color=INK, pad=16)

# ---------------- (b) observed rates ----------------
def wilson(k, n, z=1.96):
    ph = k/n; den = 1 + z*z/n
    c = (ph + z*z/(2*n))/den
    h = z*np.sqrt(ph*(1-ph)/n + z*z/(4*n*n))/den
    return c-h, c+h
b = ax[1]
pa = two.dropna(subset=["nsc", "nbt", "s1_pass"])
bands = [(35,55),(55,65),(65,75),(75,101)]; labs = ["35–54","55–64","65–74","75+"]
x = np.arange(len(bands))
for grp, col, mk, lab, off, ny in ((pa.nsc < 80, VERM, "^", "school mark below 80", -.10, 43.0),
                                   (pa.nsc >= 80, BLUE, "o", "school mark 80 or above", .10, 46.2)):
    g = pa[grp]; xs, ys, lo_, hi_ = [], [], [], []
    for i, (l, h) in enumerate(bands):
        cell = g[(g.nbt >= l) & (g.nbt < h)]
        if len(cell) < 10: continue
        k, n = int(cell.s1_pass.sum()), len(cell); wl, wh = wilson(k, n)
        xs.append(x[i]+off); ys.append(100*k/n); lo_.append(100*(k/n-wl)); hi_.append(100*(wh-k/n))
        b.text(x[i], ny, f"{n}", ha="center", va="center", fontsize=6.8, color=col)
    b.errorbar(xs, ys, yerr=[lo_, hi_], fmt=mk, color=col, ms=5, capsize=2.5, elinewidth=.9)
    b.plot(xs, ys, color=col, lw=1.0, alpha=.45)
    b.annotate(lab.replace("school mark ", ""), (xs[-1] + .14, ys[-1]), fontsize=7.4,
               color=col, va="center", ha="left")
b.text(-0.44, 44.6, "n", ha="center", va="center", fontsize=6.8, color=MUTED, style="italic")
b.set(xlabel="NBT Mathematics", ylabel="Semester 1 pass rate (%)", ylim=(40, 102), xlim=(-.55, 4.30))
b.set_xticks(x); b.set_xticklabels(labs)
b.set_yticks([50,60,70,80,90,100]); b.spines["left"].set_bounds(50, 100)
b.yaxis.grid(True); b.set_axisbelow(True)
b.set_title("(b) Pass rate by benchmark band", loc="left", fontsize=9, color=INK, pad=16)
fig.savefig(F/"fig1_plane.pdf"); plt.close(fig)
print("wrote fig1_plane.pdf")

# ---------------- Figure 2: placement ----------------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
order = ["mainstream", "augmented", "extended"]
cols = {"mainstream": BLUE, "augmented": VERM, "extended": GREEN}
for j, (var, lab) in enumerate((("nsc", "School-leaving Mathematics (%)"),
                                ("nbt", "NBT Mathematics"))):
    for i, st in enumerate(order):
        v = d[d.stream == st][var].dropna()
        parts = ax[j].violinplot([v], positions=[i], widths=.75, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(cols[st]); pc.set_alpha(.35); pc.set_edgecolor("none")
        ax[j].plot([i], [v.median()], "o", color=cols[st], ms=5, zorder=3)
        ax[j].plot([i, i], [v.quantile(.25), v.quantile(.75)], color=cols[st], lw=2, zorder=2)
    ax[j].set_xticks(range(3)); ax[j].set_xticklabels(order)
    ax[j].set_ylabel(lab)
    ax[j].set_ylim(25, 102)
    ax[j].grid(axis="y")
    ax[j].set_title(f"({'ab'[j]}) {lab.split(' (')[0]}", loc="left", fontsize=9, color=INK)
# Welch t, augmented against extended
from scipy.stats import ttest_ind
for j, (var, y_arrow, y_text) in enumerate((("nsc", (91, 91), 94), ("nbt", (66, 60), 70))):
    pv = ttest_ind(d[d.stream == "augmented"][var].dropna(), d[d.stream == "extended"][var].dropna(),
                   equal_var=False).pvalue
    colour = INK if pv < 0.05 else MUTED
    ax[j].annotate("", xy=(1, y_arrow[0]), xytext=(2, y_arrow[1]),
                   arrowprops=dict(arrowstyle="<->", color=colour, lw=.9 if pv < 0.05 else .8))
    ax[j].text(1.5, y_text, "p < 0.001" if pv < 0.001 else f"p = {pv:.2f}", fontsize=7.5,
               color=colour, ha="center")
    print(f"figure 2, {var}: Welch p = {pv:.3g}")
fig.tight_layout(); fig.savefig(F/"fig2_placement.pdf"); plt.close(fig)
print("wrote fig2_placement.pdf")

# ---------------- Figure 3: the exchange rate ----------------
c = two.dropna(subset=["nsc", "s1_pass"])
cuts = list(range(72, 94))
excl_pass = [int(c[(c.nsc < k) & (c.s1_pass == 1)].shape[0]) for k in cuts]
adm_fail = [int(c[(c.nsc >= k) & (c.s1_pass == 0)].shape[0]) for k in cuts]
fig, ax = plt.subplots(figsize=(4.4, 3.2))
ax.plot(cuts, excl_pass, "-o", color=VERM, lw=1.6, ms=3.5, label="successful students excluded")
ax.plot(cuts, adm_fail, "-s", color=BLUE, lw=1.6, ms=3.5, label="admitted students who fail")
ax.axvline(80, color=INK, lw=1, ls="--")
ax.text(80.6, ax.get_ylim()[1] * .55, "proposed\nthreshold", fontsize=7.5, color=INK)
i80 = cuts.index(80)
ax.annotate(f"{excl_pass[i80]}", (80, excl_pass[i80]), textcoords="offset points",
            xytext=(-22, 6), fontsize=8, color=VERM)
ax.annotate(f"{adm_fail[i80]}", (80, adm_fail[i80]), textcoords="offset points",
            xytext=(-22, -12), fontsize=8, color=BLUE)
ax.set(xlabel="Candidate threshold on school-leaving Mathematics (%)", ylabel="Students")
ax.grid(axis="y"); ax.legend(loc="upper center", fontsize=7.5, bbox_to_anchor=(.5, 1.16), ncol=1)
fig.tight_layout(); fig.savefig(F/"fig3_tradeoff.pdf"); plt.close(fig)
print("wrote fig3_tradeoff.pdf")
print(f"\nquadrant counts used in fig 1a: excluded above median {n_excl}, admitted at or below {n_adm}")
