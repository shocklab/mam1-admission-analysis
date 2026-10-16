"""Build the analysis sample from the faculty-wide extracts, with all three streams.

Population: first-time entrants who appear in any of the pathway course codes and have all six
pre-university measures. Outcome coding: numeric marks are parsed directly; a trailing A (absent
from the supplementary) is stripped and the mark kept, a fail; a mark recorded twice ("68; 68")
collapses to one mark, and two different marks become missing; UP and PA are passes with no
mark; AB, INC, LOA, DPR and UF are not passed, with no mark. A trailing S, DE and OSS are results
not yet in when the extract was pulled; they are resolved from the next year's enrolment
(CHOICE 6).

Every number in the paper comes from the CSV this script writes, so the judgement calls below
decide everything downstream. Each is marked in the code with "CHOICE":

  1. Population. A student is included if they appear in any of the pathway course codes, are
     flagged a first-time entrant and have all six pre-university measures. A student who
     appears only in semester 2 is included.
  2. MAM1006H is excluded from the population (see the note at THREE_STREAM).
  3. An NBT score of 0 means "did not write", not a score of zero.
  4. Outcome coding: how a Grade Code string becomes a mark and a pass/fail flag.
  5. Tie-break: which stream a student is assigned when they hold more than one record.
  6. Results not yet in (S, DE, OSS): resolved from the next year's enrolment.

Run from the project root. Writes the three-stream sample used by the analyses, and a
two-stream sample (mainstream and augmented only) used for comparison counts and for the
validation check at the end.
"""
import re, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.simplefilter("ignore")
A1 = Path("anonymised-data/Mokhithi & Shock/Paper 1")
P1 = A1 / "RAW - 2024, 2025, & 2026 data"
import os as _os0
OUT = Path(_os0.environ.get("OUTCSV",
           "anonymised-data/paper1_analysis_sample_3streams.csv"))
FILES = {2024: P1 / "MAM1 2024 Data_RAW.xlsx", 2025: P1 / "MAM1 2025 Data.xlsx",
         2026: A1 / "RAW - corrected, October 2026"
                  / "002 - FYE Early Assessment  Report Original_30Oct2026.xlsx"}
# The six pre-university measures. A student must have all six to enter the sample.
PRED = ["Math Grd12 Fin Rslt", "Eng Grd12 Fin Rslt", "Phy Sci Grd12 Fin Rslt",
        "NBT Math Score", "NBT QL Score", "NBT AL Score"]
# TWO_STREAM: mainstream and augmented, both semesters, without the extended stream.
TWO_STREAM = ["MAM1031F", "MAM1033F", "MAM1032S", "MAM1034S"]
# CHOICE 2. MAM1006H is the SECOND-year half of the extended sequence. Students appearing only
# there are not first-year entrants to the first-year sequence, whatever the
# year-of-study field says, so it does not define the population. MAM1005H does.
THREE_STREAM = TWO_STREAM + ["MAM1005H"]
# Course code -> stream. S1_OF also defines what counts as a semester-1 record, so a student
# whose only record is MAM1032S gets their stream from S2_OF further down instead.
S1_OF = {"MAM1031F": "mainstream", "MAM1033F": "augmented", "MAM1005H": "extended"}
S2_OF = {"MAM1032S": "mainstream", "MAM1034S": "augmented"}
# CHOICE 4a. Non-numeric result codes. UP (passed the supplementary) and PA (pass by aggregate)
# are passes with no mark attached; the fails are absence, incomplete, leave of absence,
# duly-performed refused and unsuccessful final. DE (deferred exam) and OSS (supplementary
# awarded, not yet written) are results not yet in, as is a mark with a trailing S.
# Anything in none of the sets becomes missing rather than being guessed at.
PASS_C, FAIL_C, PENDING_C = {"UP", "PA"}, {"AB", "INC", "LOA", "DPR", "UF"}, {"DE", "OSS"}


def parse_mark(v):
    """CHOICE 4b. Turn a Grade Code cell into a numeric mark, or NaN if it is not one.

    Handles three awkward cases found in the extracts:
      "45A"    a mark with a trailing letter -> 45.0 (the suffix is stripped)
      "68; 68" the same mark recorded twice  -> 68.0
      "68; 71" two DIFFERENT marks           -> NaN, because there is no basis for picking one
    """
    if pd.isna(v): return np.nan
    t = str(v).strip()
    if t == "" or t.lower() == "nan": return np.nan
    nums = []
    for part in [p.strip() for p in t.split(";") if p.strip()]:
        m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([A-Za-z]*)", part)
        if not m: return np.nan
        nums.append(float(m.group(1)))
    return nums[0] if nums and len(set(nums)) == 1 else np.nan


def is_pending(v):
    """True for a result not yet in when the extract was pulled: a mark with a trailing S, or
    DE or OSS. Such a result has neither a mark nor a pass flag until CHOICE 6 resolves it."""
    if pd.isna(v): return False
    t = str(v).strip().upper()
    return t in PENDING_C or bool(re.fullmatch(r"[0-9]+(?:\.[0-9]+)?S", t))


def outcome(v):
    """CHOICE 4c. -> (numeric mark, pass/eligible 1/0/nan, pending True/False).

    A numeric mark gives both a mark and a pass flag at the 50% line. A recognised letter code
    gives a pass flag but no mark, which is why the mark models are fitted on fewer students
    than the pass models. Unrecognised codes give neither, and a pending result gives neither
    until it is resolved.
    """
    if is_pending(v): return np.nan, np.nan, True
    mark = parse_mark(v)
    if not np.isnan(mark): return mark, float(mark >= 50), False
    code = str(v).strip().upper()
    if code in PASS_C: return np.nan, 1.0, False
    if code in FAIL_C: return np.nan, 0.0, False
    return np.nan, np.nan, False


# known cases, checked on every run
for _v, _want in (("47", (47.0, 0.0, False)), ("68; 68", (68.0, 1.0, False)),
                  ("46A", (46.0, 0.0, False)), ("47S", (np.nan, np.nan, True)),
                  ("DE", (np.nan, np.nan, True)), ("OSS", (np.nan, np.nan, True)),
                  ("UP", (np.nan, 1.0, False)), ("AB", (np.nan, 0.0, False)),
                  ("68; 71", (np.nan, np.nan, False)), (np.nan, (np.nan, np.nan, False))):
    _got = outcome(_v)
    assert all((a == b) or (pd.isna(a) and pd.isna(b)) for a, b in zip(_got, _want)), (_v, _got)


# CHOICE 6. A pending result is resolved from the student's courses in the NEXT year's extract
# (the 2025 S marks are the only pending results on
# pathway courses). Semester one or the extended entry course: a later MAM1006H or MAM2 course
# is a pass, a return to first-year mathematics a fail. Semester two: a later MAM2 course is a
# pass, a return to first-year mathematics a fail, and MAM1006H alone stays unresolved. No next
# year, or no mathematics enrolment in it, stays unresolved (missing). PENDING=missing, fail or
# pass in the environment replaces the rule, for sensitivity analyses. UNRESOLVED=fail or pass
# keeps the rule and counts only the results it cannot resolve as fails or as passes
# (scripts/37 uses it for the bounds reported in the paper).
FIRST_YEAR = {"MAM1031F", "MAM1033F", "MAM1032S", "MAM1034S", "MAM1005H"}
PENDING_RULE = _os0.environ.get("PENDING", "resolve")
UNRESOLVED = {"missing": np.nan, "fail": 0.0, "pass": 1.0}[_os0.environ.get("UNRESOLVED", "missing")]


def resolve(courses_next, semester):
    """-> 1.0, 0.0 or nan for a pending result, from the set of next-year course codes."""
    if PENDING_RULE != "resolve":
        return {"missing": np.nan, "fail": 0.0, "pass": 1.0}[PENDING_RULE]
    if courses_next is None: return UNRESOLVED
    mam2 = any(c.startswith("MAM2") for c in courses_next)
    if mam2 or (semester == 1 and "MAM1006H" in courses_next): return 1.0
    if courses_next & FIRST_YEAR: return 0.0
    return UNRESOLVED


def _is_unresolved(x):
    return np.isnan(x) if np.isnan(UNRESOLVED) else x == UNRESOLVED


if PENDING_RULE == "resolve":
    assert resolve({"MAM2011F"}, 2) == 1.0 and resolve({"MAM1006H"}, 1) == 1.0
    assert _is_unresolved(resolve({"MAM1006H"}, 2)) and resolve({"MAM1031F", "MAM1032S"}, 1) == 0.0
    assert _is_unresolved(resolve({"CSC1015F"}, 1)) and _is_unresolved(resolve(None, 2))


# ---------------------------------------------------------------------------------------
# One pass per cohort file. Each workbook has two sheets: "Early Assessment" carries one row
# per student with the pre-university measures, "Detail" carries one row per course record.
# ---------------------------------------------------------------------------------------
rows, COURSES = [], {}
for yr, fn in FILES.items():
    ea = pd.read_excel(fn, sheet_name="Early Assessment")
    det = pd.read_excel(fn, sheet_name="Detail")
    COURSES[yr] = (det.assign(_c=det["Course Code"].astype(str).str.strip().str.upper())
                   .groupby("anon_id")["_c"].apply(set).to_dict())
    yos = next(c for c in ea.columns if "academic year of study" in str(c).lower())
    for c in PRED: ea[c] = pd.to_numeric(ea[c], errors="coerce")
    # CHOICE 3. A zero on an NBT is "did not write", not a score of zero. Two students are
    # affected. Leaving them as zeros would drag the NBT mean down and, more importantly, put
    # them in the bottom NBT band of every table they appear in.
    for c in ("NBT Math Score", "NBT AL Score", "NBT QL Score"):
        ea.loc[ea[c] == 0, c] = np.nan
    ea = ea.assign(cohort=yr, first_time=ea[yos].astype(str).str.strip().str.lower().str.startswith("first"))
    det = det.assign(course=det["Course Code"].astype(str))
    keep = ["anon_id", "cohort", "first_time"] + PRED
    demo = {c: c for c in ea.columns if re.search(r"population|gender|^sex$", str(c), re.I)}
    # One row per student. demo picks up whichever spelling of the demographic columns this
    # cohort's extract uses (2025 says "Gender" and "Population Grp"; 2024 and 2026 say "Sex"
    # and "Population Group").
    base = ea[keep + list(demo)].drop_duplicates("anon_id")
    # Build the sample twice: once restricted to the four mainstream and augmented course codes
    # (the two-stream sample), once including MAM1005H (the three-stream sample).
    for scope, label in ((TWO_STREAM, "two"), (THREE_STREAM, "three")):
        sub = det[det.course.isin(scope)]
        # CHOICE 5. Some students hold more than one semester-1 pathway record. The rule is
        # explicit rather than left to sort order: by default the most supported stream wins
        # (extended, then augmented, then mainstream).
        # Set TIEBREAK=least in the environment to reverse the rule; scripts/18 uses that to
        # show the placement result survives either convention.
        import os as _os
        _TB = _os.environ.get("TIEBREAK", "most")
        RANK = ({"extended": 0, "augmented": 1, "mainstream": 2} if _TB == "most"
                else {"mainstream": 0, "augmented": 1, "extended": 2})
        s1r = sub[sub.course.isin(S1_OF)].copy()
        s1r["_r"] = s1r.course.map(S1_OF).map(RANK)
        n_s1 = s1r.groupby("anon_id").course.nunique()
        s1 = s1r.sort_values("_r").drop_duplicates("anon_id").set_index("anon_id")
        s2r = sub[sub.course.isin(S2_OF)].copy()
        s2r["_r"] = s2r.course.map(S2_OF).map(RANK)
        s2 = s2r.sort_values("_r").drop_duplicates("anon_id").set_index("anon_id")
        ids = set(sub.anon_id)
        b = base[base.anon_id.isin(ids)].copy()
        b["scope"] = label
        b["s1_course"] = b.anon_id.map(s1["course"])
        b["s2_course"] = b.anon_id.map(s2["course"])
        # Stream comes from the semester-1 record where there is one, and falls back to the
        # semester-2 record otherwise, which keeps students who appear only in semester 2.
        b["stream"] = b.s1_course.map(S1_OF).fillna(b.s2_course.map(S2_OF))
        g1 = b.anon_id.map(s1["Grade Code"]); g2 = b.anon_id.map(s2["Grade Code"])
        b[["s1_mark", "s1_pass", "s1_pending"]] = pd.DataFrame([outcome(v) for v in g1], index=b.index)
        b[["s2_mark", "s2_pass", "s2_pending"]] = pd.DataFrame([outcome(v) for v in g2], index=b.index)
        # observed_s2 is enrolment in a semester-2 course, not a pass. It is used as the
        # continuation outcome and reads as "still doing mathematics", nothing stronger.
        b["observed_s2"] = b.s2_course.notna()
        b["s1_streams_n"] = b.anon_id.map(n_s1).fillna(0).astype(int)
        rows.append(b)

# ---------------------------------------------------------------------------------------
# CHOICE 1. The population filter. "complete" is the all-six-predictors requirement; it is
# applied together with first_time in the two selections below, not here, so that the counts
# at each stage stay visible.
# ---------------------------------------------------------------------------------------
D = pd.concat(rows, ignore_index=True)
D["complete"] = D[PRED].notna().all(axis=1)
D["below80"] = D["Math Grd12 Fin Rslt"] < 80

# CHOICE 6, applied: resolve pending results from the next year's extract.
for sem in (1, 2):
    pend = D[f"s{sem}_pending"].astype(bool)
    D[f"s{sem}_pending"] = pend
    D.loc[pend, f"s{sem}_pass"] = [
        resolve(COURSES.get(co + 1, {}).get(a) if co + 1 in COURSES else None, sem)
        for a, co in zip(D.loc[pend, "anon_id"], D.loc[pend, "cohort"])]
_sel = D[(D.scope == "three") & D.first_time & D.complete]
print(f"Pending results (rule: {PENDING_RULE}; unresolved counted as {UNRESOLVED}), three-stream sample:")
for sem in (1, 2):
    p = _sel[_sel[f"s{sem}_pending"]]
    print(f"  semester {sem}: {len(p)} pending -> "
          f"pass {int((p[f's{sem}_pass'] == 1).sum())}, fail {int((p[f's{sem}_pass'] == 0).sum())}, "
          f"unresolved {int(p[f's{sem}_pass'].isna().sum())}; by cohort and stream "
          f"{p.groupby(['cohort', 'stream']).size().to_dict()}")

# ---------------------------------------------------------------------------------------
# Validation: the two-stream counts for 2024 and 2025 are checked against the original
# two-stream analysis of these data. 2026 comes from a corrected extract and is not expected
# to match.
# ---------------------------------------------------------------------------------------
print("=" * 74); print("VALIDATION: does the loader reproduce the original two-stream sample?"); print("=" * 74)
two = D[(D.scope == "two") & D.first_time & D.complete]
t = two.groupby("cohort").agg(n=("anon_id", "size"), numeric=("s1_mark", "count"),
                              below80=("below80", "sum"))
t.loc["Total"] = t.sum()
print(t.to_string())
print("\nOriginal: 202/360/305 (total 867); numeric 146/310/249 (705); below-80 35/59/69 (163)")
print("2026 now comes from the corrected extract, so only the 2024 and 2025 rows should match.")

print("\n" + "=" * 74); print("THE REBUILT THREE-STREAM SAMPLE"); print("=" * 74)
three = D[(D.scope == "three") & D.first_time & D.complete]
print(three.groupby(["cohort", "stream"]).agg(
    n=("anon_id", "size"), nsc=("Math Grd12 Fin Rslt", "mean"),
    nbt=("NBT Math Score", "mean"), pct_below80=("below80", lambda s: 100*s.mean()),
    pass_rate=("s1_pass", lambda s: 100*s.mean())).round(1).to_string())
print(f"\ntotal three-stream N = {len(three)}  against two-stream N = {len(two)}  "
      f"(+{len(three)-len(two)} students recovered)")
print(f"below-80 students: three-stream {int(three.below80.sum())} vs two-stream {int(two.below80.sum())}")

two.to_csv(str(OUT).replace("3streams", "2streams"), index=False)
three.to_csv(OUT, index=False)
print(f"\nwrote {OUT}")
