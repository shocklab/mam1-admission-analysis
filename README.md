# Analysis code: When Better Prediction Does Not Improve Admission

This repository contains the analysis code for Mashudu Mokhithi and Jonathan P. Shock, "When Better Prediction Does Not Improve Admission: Predictive Validity and Decision Utility in First-Year University Mathematics", *Education Sciences*.

The scripts build the analysis sample from institutional student records for the 2024, 2025 and 2026 first-year mathematics cohorts at the University of Cape Town. They produce the tables, inline values and figures reported in the paper.

## Data

The confidential institutional student records are not included. The analysis scripts read de-identified extracts from `anonymised-data/` in the repository root, produced by `scripts/anonymise.py`. Input paths are set at the top of `scripts/10_build_analysis_sample.py`.

There is one workbook per cohort, with two sheets:

- "Early Assessment": one row per student, identified by the pseudonym `anon_id`. The six pre-university measures are `Math Grd12 Fin Rslt`, `Eng Grd12 Fin Rslt`, `Phy Sci Grd12 Fin Rslt`, `NBT Math Score`, `NBT QL Score` and `NBT AL Score`. The sheet also contains `Population Grp`, `Gender` and an academic-year-of-study column whose value begins with "first" for a first-time entrant.
- "Detail": one row per course record, with `anon_id`, `Course Code` and `Grade Code`.

Course codes identify the streams: mainstream MAM1031F (Semester 1) and MAM1032S (Semester 2), augmented MAM1033F (Semester 1) and MAM1034S (Semester 2), and extended MAM1005H, which runs over the full year. The grade codes and their outcome coding are listed at the top of `scripts/10_build_analysis_sample.py`.

`scripts/anonymise.py` retains only columns on an allowlist, removing names, contact details, birth dates, school names, identity numbers and free-text notes. It replaces student numbers with a keyed hash using HMAC-SHA256 and a secret key held by the authors. Each student therefore has the same pseudonym across files and years. No mapping file exists. The first `--write` run creates the key at the path set by `SALT_FILE` in the script.

Run the anonymiser with `--dry-run` to report column counts and unrecognised column names with letters and digits masked. It writes no files. Run it with `--write` to write the de-identified extracts and list unrecognised column names in `unknown_columns.txt` for review against the allowlist.

The analysis samples, tables, figures and printed output are all derived from student records. Check them for small cells before sharing them.

## Requirements

Python 3.10 with the packages in `requirements.txt`. The paper's results were produced with the versions pinned there. The random forest and gradient boosting results in `scripts/17_model_table.py` depend on the scikit-learn version.

## Running the analysis

Run every script from the repository root, in this order:

```
python3 scripts/10_build_analysis_sample.py
python3 scripts/13_make_tables.py
python3 scripts/16_figures.py
python3 scripts/17_model_table.py
python3 scripts/18_tiebreak_sensitivity.py
python3 scripts/19_group_and_continuation.py
python3 scripts/20_threshold_decision.py
python3 scripts/21_placement_detail.py
python3 scripts/22_admission_interval.py
python3 scripts/27_two_semester.py
python3 scripts/37_pending_bounds.py
python3 scripts/36_printed_numbers.py anonymised-data/paper1_analysis_sample_3streams.csv anonymised-data/model_oof_predictions.csv
```

Script 10 runs first because every other script reads the sample it writes. Script 13 runs next because it creates `paper/tables/`. Script 37 runs after scripts 21 and 27 because it checks its point estimates against values they write. Script 36 reads the held-out predictions written by script 17. Scripts 17 and 22 take the longest. Scripts 18 and 37 re-run script 10 in a temporary folder inside `anonymised-data/` and delete the folder when they finish.

Tables and LaTeX macros are written to `paper/tables/`, figures to `paper/figures/` and analysis samples to `anonymised-data/`. Script 36 prints its results.

## Scripts

| Script | What it computes | Writes |
|---|---|---|
| `10_build_analysis_sample.py` | The analysis sample of first-time entrants in the mainstream, augmented and extended first-year mathematics streams with all six pre-university measures, their stream, and their Semester 1 and Semester 2 outcomes. The six decisions defining the sample and outcome coding are marked `CHOICE` in the code. | `anonymised-data/paper1_analysis_sample_3streams.csv`, `anonymised-data/paper1_analysis_sample_2streams.csv` |
| `13_make_tables.py` | The sample table, school-leaving Mathematics band by NBT Mathematics quartile table, nested spline models, fitted-surface predictions, comparison of the two support streams, and discontinuity tests at 80 and 90. | `tab_sample.tex`, `tab_crosstab.tex`, `tab_nested.tex`, `tab_surface.tex`, `tab_placement.tex`, `tab_changepoint.tex`, `values.tex` |
| `16_figures.py` | Figures 1 to 3. | `fig1_plane.pdf`, `fig2_placement.pdf`, `fig3_tradeoff.pdf` |
| `17_model_table.py` | A comparison under leave-one-cohort-out validation, with paired bootstrap intervals, of a main-effects linear model, ridge regression and elastic net with spline and pairwise terms, random forest and gradient boosting. It also computes pooled five-fold cross-validation for comparison. | `tab_models.tex`, `anonymised-data/model_oof_predictions.csv` |
| `18_tiebreak_sensitivity.py` | The placement result under the alternative rule for students with more than one stream record, by re-running script 10. | `values_tiebreak.tex` |
| `19_group_and_continuation.py` | Differential prediction by population group, with correlations and slopes by group and Cleary tests comparing African students with all other students. It also computes false exclusion at a cut of 80 and Semester 2 course enrolment rates, including students in the sample who appear only in a Semester 2 course. | `tab_groups.tex`, `tab_falseexcl.tex`, `tab_continuation.tex`, `values_groups.tex` |
| `20_threshold_decision.py` | Error counts at each candidate cut, a check of the identity that determines where the two error curves cross, the exchange rate between the two kinds of error, and the standard error of the jump at 80. | `values_threshold.tex` |
| `21_placement_detail.py` | The support-stream comparison by cohort, entry-course pass rates for all three streams, sensitivity to students with more than one Semester 1 stream record, omega-squared and the minimum detectable difference. | `tab_placement_cohort.tex`, `tab_entrycourse.tex`, `values_placement.tex` |
| `22_admission_interval.py` | The matched-volume admission comparison, with intervals from refitting and re-selecting within each bootstrap replicate, the number of passers a rule with perfect foresight would admit, and the out-of-sample ranking gain. | `tab_admission.tex`, `values_admission.tex`, `values_admission2.tex` |
| `27_two_semester.py` | Whether NBT Mathematics adds to school-leaving Mathematics in predicting a Semester 1 pass and a pass in both semesters, with Holm-adjusted p-values. Students with no Semester 1 result are excluded. | `values_twosem.tex` |
| `37_pending_bounds.py` | The entry-course gap and the two-semester test with results unresolved by the next year's enrolment counted first all as passes, then all as fails. It re-runs script 10 under each rule. | `values_pending.tex` |
| `36_printed_numbers.py` | The numbers in the paper computed from figures or direct calculation rather than from a table, including the cohort-level t interval for the model comparison. | printed output |
| `anonymise.py` | De-identification of the institutional extracts, as described above. It reads the protected source folder, which is not part of the repository. | `anonymised-data/` |

## Citation

Mokhithi, M. and Shock, J. P. When Better Prediction Does Not Improve Admission: Predictive Validity and Decision Utility in First-Year University Mathematics. *Education Sciences*.
