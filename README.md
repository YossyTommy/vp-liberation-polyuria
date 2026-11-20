# Early Polyuria During Vasopressor Liberation: Association With Vasopressin Discontinuation in Two ICU Databases

Analysis code for the manuscript of the same title (retrospective cohort study in MIMIC-IV v3.1
and eICU-CRD v2.0). The code compares ICU stays in which vasopressin was the last vasopressor
stopped (VP group) with stays treated with norepinephrine without vasopressin (NE group), and
estimates the association with new-onset polyuria and hypernatremia after liberation from
vasopressors.

Repository: https://github.com/YossyTommy/vp-liberation-polyuria

**No patient-level data are included in this repository.** The scripts read the PhysioNet files
from a local folder; every row-level intermediate is written to a working folder that is excluded
from version control.

## Repository layout

```
config.py                    central path configuration (environment variables, repo-relative defaults)
run_all.sh                   runs the steps of the primary pipeline in order
run_exploratory_v16.sh       runs the post hoc exploratory analyses (mimic/10-12, eicu/08-10, pooled/02-04)
run_additions_v17.sh         runs the standard-error check and the bootstrap bands (mimic/13-14, eicu/11-12, pooled/05)
early_late_helpers.py        functions shared by the post hoc scripts (Wilson intervals, masking of cells < 10,
                             summary tables, bootstrap of differences, fixed-effect pooling)
early_late_bands.py          functions shared by mimic/14 and eicu/12 (stay-level bootstrap bands)
requirements.txt             Python package versions used
mimic/                       MIMIC-IV v3.1
  01_extract.sh                one pass over the csv.gz files -> row subsets used by the later scripts
  02_primary_analysis.py       cohort, Table 1, primary two-group analysis, sensitivity analyses
  03_cif_bootstrap.py          bootstrap CIs of the weighted cumulative incidence curves
  04_window_test.py            comparison of the 0-8 hr and >8-24 hr windows (ratio of odds ratios)
  05_usage_groups.py           usage-pattern / stopping-order groups (time zero T0b)
  06_fluid_balance.py          cumulative fluid balance analyses
  07_vpstop_timing.py          timing relative to the vasopressin stop, hypernatremia timing,
                               NE-first vs VP-first, switch sensitivity, closed cohort
  08_sodium_harmonized.py      sodium >= 150 mmol/L on the denominator shared with eICU-CRD
  09_full_charting_table.py    characteristics of stays with urine recorded in all 24 hourly bins
  10_early_late.py             post hoc: stays with polyuria by time of onset; polyuria with and without a sodium
                               rise (three-category outcome)
  11_early_late_extra.py       post hoc: five-category outcome, values measured from the onset of polyuria,
                               mortality, vasopressin dose and duration, urine tests
  12_interaction.py            post hoc: interaction between group and polyuria in the association with a sodium rise
  13_se_check.py               clustered and model-based standard errors of the sodium-rise models (same fits)
  14_fig4_bands.py             bootstrap confidence bands of the closed-cohort time course
  mimic_common.py              definitions, cached extraction, data loading, cohort construction
  mimic_models.py              imputation, regression with clustered SE, weights, cumulative incidence
  mimic_groups.py              stopping-order classification and group models
  mimic_fluid.py               fluid balance, overlap weighting
eicu/                        eICU-CRD v2.0
  01_primary_analysis.py       cohort, Table 1, primary two-group analysis, sensitivity analyses
  02_cif_ci_timing.py          bootstrap CIs of the cumulative incidence curves; timing-rule sensitivity
  03_window_test.py            comparison of the 0-8 hr and >8-24 hr windows
  04_usage_groups.py           usage-pattern / stopping-order groups (time zero T0b)
  05_fluid_balance.py          cumulative fluid balance analyses
  06_vpstop_timing.py          timing relative to the vasopressin stop, hypernatremia timing, NE-first vs VP-first
  07_switch_closed.py          switch sensitivity, closed cohort, numbers at risk
  08_early_late.py             post hoc: as mimic/10
  09_early_late_extra.py       post hoc: as mimic/11
  10_interaction.py            post hoc: as mimic/12
  11_se_check.py               as mimic/13
  12_fig4_bands.py             as mimic/14
  eicu_common.py               paths, options, run log
  eicu_extract.py              loading and filtering of the raw tables
  eicu_cohort.py               cohort construction (two-group and usage-group cohorts)
  eicu_stats.py                imputation, regression with clustered SE, weights, cumulative incidence, bootstrap
  eicu_fluid.py                fluid balance, overlap weighting
  eicu_vpstop.py               analyses anchored at the vasopressin stop (shared by 06 and 07)
pooled/
  01_pooled_estimates.py       fixed-effect pooled estimates across the two databases
  02_early_late.py             pooled estimates of mimic/10 and eicu/08; Holm adjustment of the four secondary comparisons
  03_early_late_extra.py       pooled estimates of mimic/11 and eicu/09
  04_interaction.py            pooled estimates of mimic/12 and eicu/10
  05_se_check.py               pooled estimates of mimic/13 and eicu/11 (wider interval in each database)
```

Numbered files are the steps of the analysis, run in that order within each folder. Files without
a number are modules with the functions shared by the steps; they are imported, not run.
Plotting code is not included in this repository; every number shown in the figures is written to
the aggregate output tables named below.

## Scripts and manuscript items

The table names the step (and, for workbooks, the sheet) that computes the numbers. Tables and
figures in the manuscript and the supplement were prepared from these outputs.

| Manuscript item | Step(s) | Output used |
|---|---|---|
| Table 1 (baseline characteristics) | `mimic/02`, `eicu/01` | sheet `03_table1_primary` of each primary workbook |
| Primary results in the text (adjusted odds ratios; pooled estimates) | `mimic/02`, `eicu/01`, `pooled/01` | sheets `04_primary_logistic` (MIMIC-IV), `05_primary_OR` (eICU-CRD); `pooled_estimates.csv` |
| Figure 1 (study design and flow) | `mimic/02`, `eicu/01` | sheet `01_flow` of each primary workbook |
| Figure 2 (cumulative incidence; odds ratios by time window) | `mimic/03`, `eicu/02`; `mimic/02`, `mimic/04`, `eicu/01`, `eicu/03`, `pooled/01` | `fig2_cif_*.csv` (curves, CIs, numbers at risk); primary odds ratios, `window_test_*.csv`, `pooled_estimates.csv` (panel c) |
| Figure 3 (serum sodium; panel a, sodium rise by group and polyuria; panel b, sodium >= 150 mmol/L and polyuria with and without a sodium rise) | panel a: `mimic/12`, `eicu/10`, `pooled/04`; panel b: `mimic/13`, `eicu/11`, `pooled/01`, `pooled/05`, `mimic/10`, `eicu/08` | panel a: `interaction_cells.csv` (proportions with Wilson 95% CIs), `interaction_pooled.csv`; panel b: `na150_counts.csv`, `pooled_estimates.csv`, `B_category_counts.csv`, `se_check_pooled.csv` |
| Figure 4 (timing relative to the vasopressin stop, closed cohort; percentage of recorded stay-hours in a polyuria run) | `mimic/07`, `eicu/07`; `mimic/14`, `eicu/12` | `closed_cohort_hourly.csv` (hourly values; the figure pools them over 2-hr intervals), `closed_cohort.csv`; `fig4_bands.csv` (95% confidence bands) |
| Supplementary Table S1 (definitions and data items) | no output (text); the definitions are implemented in `mimic/01_extract.sh`, `mimic/mimic_common.py`, `eicu/eicu_extract.py` and `eicu/eicu_cohort.py` | |
| Supplementary Table S2 (sensitivity analyses, MIMIC-IV) | `mimic/02` | sheets `11_sensitivity`, `18_added_sensitivity` |
| Supplementary Table S3 (sensitivity analyses, eICU-CRD) | `eicu/01`, `eicu/02` | sheets `14_sensitivity`, `19_robustness`; `eicu_timing_sensitivity.csv` |
| Supplementary Table S4 (0-8 hr vs >8-24 hr windows and complementary analyses) | `mimic/02`, `mimic/03`, `mimic/04`, `eicu/01`, `eicu/02`, `eicu/03`, `pooled/01` | odds ratios: primary sheets; cause-specific hazard ratios: sheets `08_csCox` (MIMIC-IV), `09_csCox_time_split` (eICU-CRD); multinomial model and ratio of odds ratios: `window_test_*.csv`, `pooled_estimates.csv`; weighted cumulative incidence: `fig2_cif_*.csv`; risk differences and E-value: sheets `07b_CIF_IPW_RD_boot`, `05_IPW_ATT` (MIMIC-IV) |
| Supplementary Table S5 (cumulative incidence and competing events) | `mimic/02`, `mimic/03`, `eicu/01`, `eicu/02` | sheets `07_CIF_summary`, `07b_CIF_IPW_RD_boot` (MIMIC-IV), `10_CIF_8h_24h` (eICU-CRD); `fig2_cif_*.csv` |
| Supplementary Table S6 (sodium outcomes; testing frequency) | `mimic/08`, `mimic/02`, `eicu/01` | `sodium_harmonized.csv`; sheets `12_clinical`, `12b_na150_detection` (MIMIC-IV), `13_sodium_clinical` (eICU-CRD) |
| Supplementary Table S7 (hypernatremia by early polyuria and timing) | `mimic/07`, `eicu/06` | sheets `Na_*` |
| Supplementary Table S8 (fluid-balance sensitivity analyses) | `mimic/06`, `mimic/08`, `eicu/05`, `pooled/01` | fluid-balance workbooks; `sodium_harmonized.csv`; pooled values and I2 in `pooled_secondary.csv` |
| Supplementary Table S9 (polyuria by tertile of fluid overload) | `mimic/06`, `eicu/05` | sheets `05_c_FO_tertiles`, `05b_c_interaction` (MIMIC-IV); `c_tertiles`, `c_interaction` (eICU-CRD) |
| Supplementary Table S10 (co-interventions after T0) | `mimic/02`, `eicu/01` | sheets `13_cointerventions` (MIMIC-IV), `16_cointerventions` (eICU-CRD) |
| Supplementary Table S11 (stays with all 24 hourly urine bins recorded, MIMIC-IV) | `mimic/09`; `mimic/02` | `etable11_full_charting.csv` (file name of the output; the table is Supplementary Table S11); the corresponding odds ratios are row "(s3)" of sheet `18_added_sensitivity` |
| Supplementary Table S12 (subgroups: sepsis, cardiac surgery) | `mimic/02`, `eicu/01` | sheets `10_subgroups` (MIMIC-IV), `15_subgroups` (eICU-CRD) |
| Supplementary Table S13 (polyuria relative to the vasopressin stop; closed cohort; cumulative incidence) | `mimic/07`, `eicu/06`, `eicu/07` | sheets `VPstop_*`; `closed_cohort.csv`, `vpstop_cif.csv`, `risk_table.csv` |
| Supplementary Table S14 (characteristics by usage pattern and stopping order) | `mimic/05`, `eicu/04` | sheets `03_table1_5groups` (MIMIC-IV), `03_background_5grp` (eICU-CRD) |
| Supplementary Table S15 (polyuria and hypernatremia by usage pattern and stopping order) | `mimic/05`, `mimic/08`, `eicu/04`, `pooled/01` | sheets `06_outcomes` (MIMIC-IV), `06_outcomes_OR`, `07b_sodium_OR` (eICU-CRD); rows `5grp: ...` of `sodium_harmonized.csv`; pooled values and I2 in `pooled_secondary.csv` |
| Supplementary Table S16 (NE-first vs VP-first by time anchor, overlap weighting) | `mimic/07`, `eicu/06` | sheets `Order_*`; `order_estimates.csv` |
| Supplementary Table S17 (excluding switches from norepinephrine to vasopressin) | `mimic/07`, `eicu/07` | `switch_sensitivity.csv` |
| Supplementary Figure S1 (fluid balance and early polyuria) | `mimic/06`, `eicu/05`, `pooled/01` | sheets `05_c_FO_tertiles` (MIMIC-IV), `c_tertiles` (eICU-CRD); `pooled_secondary.csv` |
| Supplementary Figure S2 (time points and populations) | no output (schematic drawn without data) | |
| Supplementary Figure S3 (open cohort around the vasopressin stop; cumulative incidence) | `mimic/07`, `eicu/06` | `vpstop_hourly.csv`, `vpstop_cif.csv`, `vpstop_t0b_hist.csv` |
| Supplementary Figure S4 (odds ratios by usage pattern and stopping order) | `mimic/05`, `mimic/08`, `eicu/04`, `pooled/01` | as Supplementary Table S15 |
| Supplementary Figure S5 (mean hourly urine output in the closed cohort) | `mimic/07`, `eicu/07`; `mimic/14`, `eicu/12` | `closed_cohort_hourly.csv`; `fig4_bands.csv` |
| Supplementary Table S18 (Holm adjustment of the four secondary comparisons) | `pooled/02` | `C_holm_secondary.csv` |
| Supplementary Table S19 (polyuria with and without a sodium rise: three-category outcome) | `mimic/10`, `eicu/08`; `mimic/13`, `eicu/11`, `pooled/05` | `B_category_counts.csv`; `se_check_pooled.csv` (pooled from `se_check_estimates.csv` of each database) |
| Supplementary Table S20 (interaction between group and polyuria) | `mimic/12`, `eicu/10`, `pooled/04` | `interaction_cells.csv`; `interaction_pooled.csv` (pooled from `interaction_estimates.csv` of each database) |
| Supplementary Table S21 (five-category outcome) | `mimic/11`, `eicu/09`; `mimic/13`, `eicu/11`, `pooled/05` | `E1_category5_counts.csv`; `se_check_pooled.csv` (pooled from `se_check_estimates.csv` of each database) |
| Supplementary Table S22 (stays with polyuria by time of onset, measured from T0) | `mimic/10`, `eicu/08`, `pooled/02` | `A_summary.csv`, `A_group_sizes.csv`, `A_contrasts_bootstrap.csv`; `A_interaction_pooled.csv` |
| Supplementary Table S23 (the same stays, measured from the onset of polyuria) | `mimic/11`, `eicu/09`, `pooled/03` | `E2_summary.csv`; `E2_regression_pooled.csv` |
| Supplementary Table S24 (urine osmolality, urine sodium, urine specific gravity) | `mimic/11`, `eicu/09` | `E5_urine_labs_by_group.csv` |
| Supplementary Table S25 (hospital mortality and ICU days) | `mimic/11`, `eicu/09`, `pooled/03` | `E3_outcomes_by_category.csv`; `E3_death_models_pooled.csv` |
| Supplementary Table S26 (duration and rate of the vasopressin infusion) | `mimic/11`, `eicu/09`, `pooled/03` | `E4_vp_dose_summary.csv`; `E4_vp_dose_models_pooled.csv` |
| Results of the sodium-rise analyses in the text | `pooled/04`, `pooled/05`, `pooled/03` | `interaction_pooled.csv`, `se_check_pooled.csv`, `E4_vp_dose_models_pooled.csv` |

## Primary, secondary and post hoc analyses

No protocol was registered. As stated in the manuscript, the comparator design, the time windows and the
exploratory analyses were developed after initial examination of the data. The scripts fall into four groups.

| Group | Scripts | Status in the manuscript |
|---|---|---|
| Primary and secondary outcomes and their sensitivity analyses | `mimic/02`-`04`, `06`, `08`, `09`; `eicu/01`-`03`, `05`; `pooled/01` (`run_all.sh`) | Primary outcome (polyuria within 24 hr), secondary outcomes (onset at 0-8 hr and at >8-24 hr; serum sodium >= 150 mmol/L within 72 hr), sensitivity analyses and subgroups. The analysis by tertile of fluid overload (`mimic/06`, `eicu/05`; Supplementary Table S9) is reported as post hoc. |
| Exploratory analyses of timing and stopping order | `mimic/05`, `07`; `eicu/04`, `06`, `07` (`run_all.sh`) | Exploratory (Supplementary Tables S13-S17, Supplementary Figures S3-S5, Figure 4). |
| Post hoc analyses added after the primary and secondary results were known | `mimic/10`-`12`; `eicu/08`-`10`; `pooled/02`-`04` (`run_exploratory_v16.sh`) | Post hoc and exploratory, not adjusted for multiple comparisons: polyuria with and without a sodium rise, interaction between group and polyuria, time of onset, mortality, infusion duration, urine tests (Supplementary Tables S19-S26, Figure 3). `pooled/02` also applies the Holm procedure to the four secondary comparisons (Supplementary Table S18). |
| Additions without a new analysis | `mimic/13`, `14`; `eicu/11`, `12`; `pooled/05` (`run_additions_v17.sh`) | The sodium-rise models are fitted again unchanged to obtain clustered and model-based standard errors (the wider interval in each database is reported); bootstrap confidence bands are added to the closed-cohort time course. |

The headers of the added scripts keep the labels of the manuscript drafts in which they were written (v15, v16, v17):
v15 is the manuscript version of the primary pipeline. Where a header says "not part of the manuscript analyses",
the script was written before its results were added to the manuscript; these analyses are now reported as post hoc
exploratory analyses. Where the header of `mimic/12` and `eicu/10` says "pre-specified", it means that the outcome,
the model and the sensitivity definition were fixed before the model was fitted, after the descriptive proportions
had been seen; the analysis is post hoc.

## Data access

The data are not redistributed here and may not be shared by the authors. Both databases are
available from PhysioNet to credentialed users who have completed the required training and
signed the data use agreement of each database.

* MIMIC-IV v3.1: https://physionet.org/content/mimiciv/3.1/ (doi:10.13026/kpb9-mt58)
* eICU Collaborative Research Database v2.0: https://physionet.org/content/eicu-crd/2.0/ (doi:10.13026/C2WM1R)

Download the compressed csv files and keep the PhysioNet folder structure:

```
<MIMIC_DIR>/hosp/*.csv.gz   <MIMIC_DIR>/icu/*.csv.gz
<EICU_DIR>/*.csv.gz
```

Do not commit data, the working folder or the output folder. `.gitignore` excludes `data/`,
`work/`, `out/` and the usual table and figure file types.

## Environment

Python 3.11 with the package versions in `requirements.txt`. The extraction steps call `gunzip`
and `awk` through a POSIX shell, so the pipeline needs macOS or Linux (on Windows: WSL). The
analyses were run on macOS with its BSD awk; mawk or gawk is used when available.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

All paths are defined in `config.py` and can be set with environment variables. Defaults are
folders inside the repository.

| Variable | Meaning | Default |
|---|---|---|
| `MIMIC_DIR` | MIMIC-IV v3.1 root (contains `hosp/` and `icu/`) | `data/mimiciv/3.1` |
| `EICU_DIR` | eICU-CRD v2.0 root | `data/eicu-crd/2.0` |
| `WORK_DIR` | row-level intermediates (extracts, caches, cohort tables) | `work` |
| `OUT_DIR` | aggregate results and logs | `out` |

Optional: `NBOOT` (bootstrap replicates), `M_IMP` / `MICE_M` (number of imputations, MIMIC-IV /
eICU-CRD scripts) and `NJOBS` (parallel workers). Leave them unset to reproduce the manuscript;
the defaults differ between steps (500 replicates in `mimic/02`, `mimic/03`, `eicu/01` and
`eicu/02`; 200 in `mimic/06`-`08` and `eicu/05`-`07`; 10 imputations). The added scripts use `NBOOT_EL`
(bootstrap of differences in `mimic/10`, `11` and `eicu/08`, `09`; default 2,000) and `NBOOT_BANDS`
(confidence bands in `mimic/14` and `eicu/12`; default 1,000). Random seeds are fixed in the code.

## Running

```bash
export MIMIC_DIR=/path/to/mimiciv/3.1
export EICU_DIR=/path/to/eicu-crd/2.0
bash run_all.sh         # or: bash run_all.sh mimic | eicu | pooled
bash run_exploratory_v16.sh     # post hoc exploratory analyses; needs the results of run_all.sh
bash run_additions_v17.sh       # standard-error check and bootstrap bands; needs both of the above
```

Each step can also be run on its own from the repository root, e.g.
`python mimic/02_primary_analysis.py`.

1. MIMIC-IV: `01` -> `02` -> `03`, `04`, `09` (each needs 02); `05` (needs 01); `06` (needs 02 and 05); `07`, `08` (need 02, 05 and 06)
2. eICU-CRD: `01`; `02` (re-uses the extracts of 01); `03` (needs 01); `04`; `05` (needs 01 and 04); `06` (needs 01, 04 and 05); `07` (needs 01 and 04)
3. Pooled estimates: `pooled/01` (needs `mimic/02`, `04`, `05`, `06`, `08` and `eicu/01`, `03`, `04`, `05`)
4. Post hoc analyses (`run_exploratory_v16.sh`): `mimic/10` (needs `mimic/02` and `07`) -> `mimic/11`, `mimic/12`; `eicu/08` (needs `eicu/01`) -> `eicu/09`, `eicu/10`; then `pooled/02` (needs `mimic/10`, `eicu/08` and `pooled/01`), `pooled/03` (needs `mimic/11`, `eicu/09`), `pooled/04` (needs `mimic/12`, `eicu/10`)
5. Additions (`run_additions_v17.sh`): `mimic/13` (needs `mimic/10`, `11`), `eicu/11` (needs `eicu/08`, `09`), `pooled/05` (needs `mimic/13`, `eicu/11`, `pooled/02`, `pooled/03`); `mimic/14` (needs `mimic/07`), `eicu/12` (needs `eicu/07`)

The two database chains are independent of each other and can be run at the same time. A later
step reads the cohort tables that earlier steps wrote to `WORK_DIR` and imports the shared
functions from the modules in its folder.

Several steps end with a block of checks that compare key counts and estimates with the values
reported in the manuscript and stop, or flag the row, when a value differs. These expected values
hold for the default settings only. Every added script first recomputes key values of the primary
pipeline, or compares its point estimates with the earlier outputs, and stops when a value differs.

## Run time

Measured on a 12-core laptop with 32 GB RAM (macOS), the two chains running side by side:

| Step | Time |
|---|---|
| `mimic/01_extract.sh` | 73 min 47 s |
| `mimic/02_primary_analysis.py` | 18 min 16 s |
| `mimic/03_cif_bootstrap.py` | 3 min 05 s |
| `mimic/04_window_test.py` | 10 s |
| `mimic/05_usage_groups.py` | 7 min 16 s |
| `mimic/06_fluid_balance.py` | 10 min 56 s |
| `mimic/07_vpstop_timing.py` | 5 min 23 s |
| `mimic/08_sodium_harmonized.py` | 4 min 13 s |
| `mimic/09_full_charting_table.py` | 3 s |
| `eicu/01_primary_analysis.py` | 23 min 12 s |
| `eicu/02_cif_ci_timing.py` | 10 min 48 s |
| `eicu/03_window_test.py` | 7 s |
| `eicu/04_usage_groups.py` | 12 min 46 s |
| `eicu/05_fluid_balance.py` | 62 min 25 s |
| `eicu/06_vpstop_timing.py` | 7 min 08 s |
| `eicu/07_switch_closed.py` | 12 min 46 s |
| `pooled/01_pooled_estimates.py` | 5 s |

MIMIC-IV chain 123 min 09 s, eICU-CRD chain 129 min 12 s; 2 hr 9 min wall-clock with the two chains in parallel.
The extraction step (`mimic/01_extract.sh`) was run with the BSD awk of macOS; mawk and gawk were
not timed. Disk space: about 3.2 GB in `WORK_DIR`. The run times of the added scripts (steps 4 and 5) were not
recorded and are not included in the totals above.

## Outputs

* `WORK_DIR/mimic/data` (csv extracts), `WORK_DIR/mimic/ext` (parquet caches),
  `WORK_DIR/mimic/cohort` (analysis tables); `WORK_DIR/eicu/ex`, `WORK_DIR/eicu/ex_fluid`
  (filtered raw tables), `WORK_DIR/eicu/cohort`, `WORK_DIR/eicu/cohort_timing_rules`. These files
  contain one row per stay or per record and must be handled under the PhysioNet data use
  agreement.
* `OUT_DIR/mimic_02` ... `mimic_09`, `OUT_DIR/eicu_01` ... `eicu_07`: aggregate results of the
  step with that number - an Excel workbook with one sheet per analysis
  (`results_mimic_*.xlsx`, `eICU_*_results.xlsx`), csv tables with the numbers shown in the
  figures, a run log and, for some steps, a table of reproduction checks.
* `OUT_DIR/mimic_10` ... `mimic_14`, `OUT_DIR/eicu_08` ... `eicu_12`, `OUT_DIR/pooled_02` ... `pooled_05`:
  aggregate csv tables of the added scripts and a table of reproduction checks. In the tables of
  `mimic/10`, `11` and `eicu/08`, `09`, counts below 10 are written as `<10`. Their row-level files are
  written to `WORK_DIR/mimic/early_late` and `WORK_DIR/eicu/early_late`.
* `OUT_DIR/pooled/pooled_estimates.csv` (primary analysis) and `pooled_secondary.csv`
  (fluid-balance sensitivity analyses and stopping-order groups). All pooled values are computed
  from the database-specific estimates at full precision; `est`, `lo` and `hi` hold the pooled
  estimate and its 95% CI before rounding.
* `OUT_DIR/logs/`: console log of every step and `runtime_seconds.tsv` (written by `run_all.sh`).

Aggregate tables can contain small cell counts; check the data use agreement before sharing them.

## Citation

If you use this code, please cite the article (citation to be added on publication) and this
repository: https://github.com/YossyTommy/vp-liberation-polyuria

Please also cite the databases:

* Johnson A, Bulgarelli L, Pollard T, et al. MIMIC-IV (version 3.1). PhysioNet. 2024. doi:10.13026/kpb9-mt58
* Pollard T, Johnson A, Raffa J, et al. eICU Collaborative Research Database (version 2.0). PhysioNet. 2019. doi:10.13026/C2WM1R

## License

MIT (see `LICENSE`). The license covers the code only, not the data.
