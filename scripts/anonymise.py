"""Procedural anonymiser for the institutional extracts.

Reads the un-anonymised folder and writes a clean copy elsewhere. It prints column names and
aggregate counts only, never cell values.

  dry run (writes nothing):  python3 scripts/anonymise.py --dry-run
  for real:                  python3 scripts/anonymise.py --write

Design:
  * ALLOWLIST of columns to keep. Everything else is dropped by default, so a new identifier
    column in a future extract is dropped rather than leaked.
  * Identifiers are used only to make the pseudonym, then discarded. No field is derived from
    a birth date.
  * Student IDs become a truncated HMAC under a secret salt held only by the authors, giving
    consistent pseudonyms across files and years with no mapping file in existence.
  * Unrecognised columns are counted and dropped. A dry run prints only the shape of each
    unrecognised header (letters and digits masked) and writes nothing; --write also lists
    their names in unknown_columns.txt so they can be promoted to the allowlist.
"""
import argparse, hashlib, hmac, os, re, secrets, sys, warnings
from pathlib import Path
import pandas as pd

warnings.simplefilter("ignore")
SRC = Path("Full google doc data")
DST = Path("anonymised-data")
SALT_FILE = Path.home() / ".mam1_anon_salt"

# --- columns to keep, normalised to lowercase-no-punctuation -----------------
ALLOW = {
    # cohort / enrolment context
    "enrol term", "enrol academic year", "academic year of study", "admit term",
    "academic standing action", "academic plan code", "academic program code",
    "academic program", "course career code", "entrance category code",
    "history term", "history term code",
    # demographics kept for the differential-prediction analysis
    "population grp", "population group", "gender", "sex",
    "international or sa", "non sa citizenship",
    # school context (school NAME is dropped; quintile/decile are safe and wanted)
    "quintile", "decile", "fin aid student",
    # pre-university measures
    "nbt al score", "nbt math score", "nbt ql score",
    "nbt al performance", "nbt ql performance", "nbt math performance",
    "redress category code", "redress category",
    "eng grd12 fin rslt", "math grd12 fin rslt", "phy sci grd12 fin rslt",
    "eng grd12 fin", "math grd12 fin",
    # courses and outcomes
    "course code", "course description", "courses", "grade code", "subject",
    "catalog nbr", "term", "official grade code", "official grade",
    "official grade performance group", "grade points per unit", "earn credit",
    "units attempted code", "mid term 2020 grade", "course mark with mid term sub.",
    "grade", "combined gpa", "gpa", "total courses", "courses passed", "courses failed",
    "program related faculty score",
    # "faculty scores" removed: 85-86% unique per cohort, a near-key with little value
    "course faculty code", "program plan faculty code",
}
# the wide 2025 layout strings course/mark pairs across these
# Columns are allowlisted by name, never by position, so a future wide extract cannot
# export whatever happens to sit in a given column.
ALLOW |= {
    # exposure and dose
    "augmented", "whether on the augmented course or not", "attendance", "tutorial groups",
    # within-semester assessment
    "class record", "exam", "final grade", "dp", "total score", "max points",
    "test 1s", "test 2s", "cr s", "exam s", "final s",
    "test 1f", "test 2f", "cr f", "exam f", "final f", "fts or rs",
    # behavioural (non-identifying)
    "status", "lateness h:m:s", "lateness",
    # admin
    "class nbr", "acad prog", "faculty", "level", "course", "course_role",
    "grade_original", "grade_clean", "numeric_mark", "pass",
    "first_time_entry", "returning_rs", "number of courses", "pathway_category",
    # free-text note columns ("note", "coding_note") are not kept: free text can carry names
}
ALLOW_RX = [re.compile(r) for r in (
    r"^assignment \d", r"^test ?\d+\b", r"^\d+(\.\d+)*[.:] ", r"^mam[23]_",
    r"^\w*honours_?\w*$", r"^extended_", r"^standard_",
)]

# linkage keys: hashed to a pseudonym, originals discarded. Priority order matters:
# all sheets must hash the SAME underlying value or the files will not join.
# email local part and username are the student number at UCT; OrgDefinedId is an
# internal LMS id and does NOT join, so it goes last.
ID_PRIORITY = ["campus id", "email", "email.1", "username", "sid", "student id",
               "id number", "emplid", "employee id", "orgdefinedid"]


# --- identifier columns: used to derive, then dropped ------------------------
ID_COLS   = {"campus id", "employee id", "emplid", "id number", "application number",
             "sid", "orgdefinedid", "student id", "username", "email", "email.1"}
DROP_PII  = ID_COLS | {"name", "cell phone", "email", "email.1", "first name",
                       "last name", "birth date", "sa school", "school", "school id",
                       "submission id", "submission time", "view count",
                       "submission count", "end-of-line indicator", "endofline indicator"}

RX = [re.compile(p, re.I) for p in (
    r"[\w.+-]+@[\w-]+\.[\w.]+", r"\b\d{13}\b", r"\b[A-Z]{6}\d{3}\b",
    r"\b(?:\+27|0)\d{9}\b", r"\b\d{8,}\b")]


def norm(c):
    return re.sub(r"[^a-z0-9._: ]", "", str(c).strip().lower())


def redact(s):
    """Mask the SHAPE of an unrecognised header. Never echo it: under header misdetection a
    cell value becomes a column name, and a name or spaced phone number matches no regex."""
    t = str(s)[:40]
    return "".join("9" if ch.isdigit() else "a" if ch.islower()
                   else "A" if ch.isupper() else ch for ch in t)


def get_salt():
    if SALT_FILE.exists():
        return SALT_FILE.read_bytes()
    salt = secrets.token_bytes(32)
    SALT_FILE.write_bytes(salt)
    os.chmod(SALT_FILE, 0o600)
    print(f"  ! new salt written to {SALT_FILE} (mode 600). Keep it to give later files the "
          f"same pseudonyms; deleting it means no further file can be linked to these ones.")
    return salt


def pseudonym(value, salt):
    if pd.isna(value):
        return None
    if isinstance(value, float) and float(value).is_integer():
        value = int(value)            # 12345678.0 -> "12345678", not "123456780"
    v = str(value).strip().upper()
    if "@" in v:                      # student email -> local part is the student number
        v = v.split("@")[0]
    v = re.sub(r"[^A-Z0-9]", "", v)   # normalise so the same student hashes alike everywhere
    if len(v) < 4 or v in {"NA", "NONE", "NULL", "0", "UNKNOWN", "NAN"}:
        return None                   # never merge degenerate keys under one pseudonym
    return "S" + hmac.new(salt, v.encode(), hashlib.sha256).hexdigest()[:10]


def header_row(df):
    """Index of the header row, or None if we cannot tell confidently."""
    known = ALLOW | DROP_PII | ID_COLS
    for i in range(min(12, len(df))):
        row = [v for v in df.iloc[i].tolist() if pd.notna(v)]
        if len(row) < 3:
            continue
        strings = [v for v in row if isinstance(v, str)]
        if not strings:
            continue
        shaped = len(strings)/len(row) >= 0.8 and len(set(strings))/len(strings) >= 0.8
        # A data row scores near zero here; a real header matches known vocabulary.
        recognised = sum(1 for v in strings
                         if norm(v) in known or any(rx.search(norm(v)) for rx in ALLOW_RX))
        if shaped and recognised / len(strings) >= 0.4:
            return i
    return None


def process(path, salt, write):
    rel = path.relative_to(SRC)
    print(f"\n--- {rel}")
    try:
        book = pd.read_excel(path, sheet_name=None, header=None)
    except Exception as e:
        print(f"    SKIP (unreadable: {type(e).__name__})")
        return
    out = {}
    for sheet, raw in book.items():
        h = header_row(raw)
        if h is None:
            print(f"    [{sheet}] header not confidently detected; sheet dropped")
            continue
        df = pd.read_excel(path, sheet_name=sheet, header=h)
        df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
        cols = {c: norm(c) for c in df.columns}
        # The deny list wins: a column matching both an allow pattern and a risky name
        # pattern is dropped.
        RISKY = re.compile(r"name|e?mail|phone|cell|\bid\b|number|address|birth|school|"
                           r"surname|initial|contact|dob", re.I)
        NOLETTER = re.compile(r"^[\d.:\s]+$")   # "082 555 1234" is not a question number
        keep = [c for c, n in cols.items()
                if (n in ALLOW or any(rx.search(n) for rx in ALLOW_RX))
                and n not in DROP_PII and n not in ID_COLS
                and not (any(rx.search(n) for rx in ALLOW_RX)
                         and (RISKY.search(n) or NOLETTER.match(n)))]
        ids = sorted([c for c, n in cols.items() if n in ID_COLS],
                     key=lambda c: ID_PRIORITY.index(cols[c])
                     if cols[c] in ID_PRIORITY else 99)
        pii = [c for c, n in cols.items() if n in DROP_PII]
        unknown = [c for c, n in cols.items()
                   if n not in ALLOW and n not in DROP_PII
                   and not any(rx.search(n) for rx in ALLOW_RX)]

        clean = df[keep].copy()
        if ids:
            clean.insert(0, "anon_id", df[ids[0]].map(lambda v: pseudonym(v, salt)))

        print(f"    [{sheet}] rows={len(df)}  kept={len(keep)}  "
              f"pii_dropped={len(pii)}  unknown_dropped={len(unknown)}"
              f"  key={cols[ids[0]] if ids else 'NONE'}")
        if unknown and not write:
            print(f"        unknown -> {len(unknown)} columns, shapes only: "
                  f"{', '.join(redact(cc) for cc in unknown)}")
        elif unknown:
            print(f"        unknown -> {len(unknown)} columns, names written to "
                  f"unknown_columns.txt (open it yourself; not echoed here)")
            with open("unknown_columns.txt", "a") as fh:
                fh.write(f"\n{rel} [{sheet}]\n")
                for cc in unknown:
                    fh.write(f"    {cc}\n")
        out[sheet[:31]] = clean

    if write and out:
        dest = DST / rel.with_suffix(".xlsx")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(dest) as w:
            for s, d in out.items():
                d.to_excel(w, sheet_name=s, index=False)
        print(f"    -> wrote {dest}")


ap = argparse.ArgumentParser()
mode = ap.add_mutually_exclusive_group(required=True)
mode.add_argument("--write", action="store_true")
mode.add_argument("--dry-run", action="store_true")
ap.add_argument("--only", help="process only this file or folder inside Full google doc data/")
a = ap.parse_args()
root = Path(a.only) if a.only else SRC
if not root.resolve().is_relative_to(SRC.resolve()):
    sys.exit("--only must point inside Full google doc data/")
root = SRC / root.resolve().relative_to(SRC.resolve())

salt = get_salt() if a.write else b"DRYRUN" * 8
print(f"{'WRITING' if a.write else 'DRY RUN: nothing will be written'}")
for p in sorted([root] if root.is_file() else root.rglob("*")):
    skip = (any(t in p.name.lower() for t in ("anonymised", "anonymized", "mapping"))
            or "superseded" in (q.lower() for q in p.parts))
    if p.suffix.lower() in (".xlsx", ".xls") and not p.name.startswith("~") and not skip:
        process(p, salt, a.write)
