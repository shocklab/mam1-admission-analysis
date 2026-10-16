# Analysis code: When Better Prediction Does Not Improve Admission

This repository contains the analysis code for Mashudu Mokhithi and Jonathan P. Shock, "When Better Prediction Does Not Improve Admission: Predictive Validity and Decision Utility in First-Year University Mathematics", *Education Sciences*.

The scripts build the analysis sample from institutional student records for the 2024, 2025 and 2026 first-year mathematics cohorts at the University of Cape Town. They produce the tables, inline values and figures reported in the paper.

## Data

The confidential institutional student records are not included. The analysis scripts read de-identified extracts from `anonymised-data/` in the repository root, produced by `scripts/anonymise.py`.

There is one workbook per cohort. The "Early Assessment" sheet contains one row per student, with six pre-university measures, demographic fields and the first-time-entrant flag. The "Detail" sheet contains one row per course record, with the course code and grade code. Input paths are set at the top of `scripts/10_build_analysis_sample.py`.

`scripts/anonymise.py` retains only columns on an allowlist, removing names, contact details, birth dates, school names and identity numbers. It replaces student numbers with a keyed hash using HMAC-SHA256 and a secret key held by the authors. Each student therefore has the same pseudonym across files and years. No mapping file exists. Run it with `--dry-run` to report column names and counts without writing files, or with `--write` to write the de-identified extracts.

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
```

Script 10 runs first because every other script reads the sample it writes. Script 13 runs next because it creates `paper/tables/`. Script 37 runs last because it checks its point estimates against values written by scripts 21 and 27. Scripts 17 and 22 take several minutes each.

Tables and LaTeX macros are written to `paper/tables/`, figures to `paper/figures/` and analysis samples to `anonymised-data/`.

## Scripts

| Script | What it computes | Writes |
|---|---|---|
| `10_build_analysis_sample.py` | The analysis sample of first-time entrants in the mainstream, augmented and extended first-year mathematics streams with all six pre-university measures, their stream, and their Semester 1 and Semester 2 outcomes. The six decisions defining the sample and outcome coding are marked `CHOICE` in the code. | `anonymised-data/paper1_analysis_sample_3streams.csv`, `anonymised-data/paper1_analysis_sample_2streams.csv` |
| `13_make_tables.py` | The sample table, school-leaving Mathematics band by NBT Mathematics quartile table, nested spline models, fitted-surface predictions, comparison of the two support streams, and discontinuity tests at 80 and 90. | `tab_sample.tex`, `tab_crosstab.tex`, `tab_nested.tex`, `tab_surface.tex`, `tab_placement.tex`, `tab_changepoint.tex`, `values.tex` |
| `16_figures.py` | Figures 1 to 3. | `fig1_plane.pdf`, `fig2_placement.pdf`, `fig3_tradeoff.pdf` |
| `17_model_table.py` | A comparison under leave-one-cohort-out validation, with paired bootstrap intervals, of a main-effects linear model, ridge regression and elastic net with spline and pairwise terms, random forest and gradient boosting. It also computes pooled five-fold cross-validation for comparison. | `tab_models.tex`, `anonymised-data/model_oof_predictions.csv` |
| `18_tiebreak_sensitivity.py` | The placement result under the alternative rule for students with more than one stream record, by re-running script 10. | `values_tiebreak.tex` |
| `19_group_and_continuation.py` | Differential prediction by population group (correlations, slopes and Cleary tests), false exclusion at a cut of 80, and continuation into Semester 2. | `tab_groups.tex`, `tab_falseexcl.tex`, `tab_continuation.tex`, `values_groups.tex` |
| `20_threshold_decision.py` | Error counts at each candidate cut, a permutation test of where the two error curves cross, the exchange rate between the two kinds of error, and the standard error of the jump at 80. | `values_threshold.tex` |
| `21_placement_detail.py` | The support-stream comparison by cohort, entry-course pass rates for all three streams, sensitivity to students who moved between streams, omega-squared and the minimum detectable difference. | `tab_placement_cohort.tex`, `tab_entrycourse.tex`, `values_placement.tex` |
| `22_admission_interval.py` | The matched-volume admission comparison, with intervals from refitting and re-selecting within each bootstrap replicate, the number of passers a rule with perfect foresight would admit, and the out-of-sample ranking gain. | `tab_admission.tex`, `values_admission.tex`, `values_admission2.tex` |
| `27_two_semester.py` | Whether NBT Mathematics adds to school-leaving Mathematics when success means passing both semesters. | `values_twosem.tex` |
| `37_pending_bounds.py` | The entry-course gap and two-semester test when results still pending at extraction are counted first all as passes, then all as fails, by re-running script 10 under each rule. | `values_pending.tex` |
| `anonymise.py` | De-identification of the institutional extracts, as described above. It reads the protected source folder, which is not part of the repository. | `anonymised-data/` |

## Citation

Mokhithi, M. and Shock, J. P. When Better Prediction Does Not Improve Admission: Predictive Validity and Decision Utility in First-Year University Mathematics. *Education Sciences*.
