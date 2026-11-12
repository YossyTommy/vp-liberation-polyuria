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
run_all.sh                   runs every step in order
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
  eicu_common.py               paths, options, run log
  eicu_extract.py              loading and filtering of the raw tables
  eicu_cohort.py               cohort construction (two-group and usage-group cohorts)
  eicu_stats.py                imputation, regression with clustered SE, weights, cumulative incidence, bootstrap
  eicu_fluid.py                fluid balance, overlap weighting
  eicu_vpstop.py               analyses anchored at the vasopressin stop (shared by 06 and 07)
pooled/
  01_pooled_estimates.py       fixed-effect pooled estimates across the two databases
```

Numbered files are the steps of the analysis, run in that order within each folder. Files without
a number are modules with the functions shared by the steps; they are imported, not run.
Every number shown in the figures is written to the aggregate output tables named below.

## Scripts and manuscript items

The table names the step (and, for workbooks, the sheet) that computes the numbers. Tables and
figures in the manuscript and the supplement were prepared from these outputs.

| Manuscript item | Step(s) | Output used |
|---|---|---|
| Table 1 (baseline characteristics) | `mimic/02`, `eicu/01` | sheet `03_table1_primary` of each primary workbook |
| Primary results in the text (adjusted odds ratios; pooled estimates) | `mimic/02`, `eicu/01`, `pooled/01` | sheets `04_primary_logistic` (MIMIC-IV), `05_primary_OR` (eICU-CRD); `pooled_estimates.csv` |
| Figure 1 (study design and flow) | `mimic/02`, `eicu/01` | sheet `01_flow` of each primary workbook |
| Figure 2 (cumulative incidence; odds ratios by time window) | `mimic/03`, `eicu/02`; `mimic/02`, `mimic/04`, `eicu/01`, `eicu/03`, `pooled/01` | `fig2_cif_*.csv` (curves, CIs, numbers at risk); primary odds ratios, `window_test_*.csv`, `pooled_estimates.csv` (panel c) |
| Figure 3 (serum sodium) | `mimic/08`, `eicu/01`, `pooled/01` | `sodium_harmonized.csv`; sheet `13_sodium_clinical` (eICU-CRD); `pooled_estimates.csv` (error bars in panel a are Wilson 95% CIs of the reported counts) |
| Figure 4 (timing relative to the vasopressin stop, closed cohort) | `mimic/07`, `eicu/07` | `closed_cohort_hourly.csv` (hourly values; the figure pools them over 2-hr intervals), `closed_cohort.csv` |
| eTable 1 (definitions and data items) | no output (text); the definitions are implemented in `mimic/01_extract.sh`, `mimic/mimic_common.py`, `eicu/eicu_extract.py` and `eicu/eicu_cohort.py` | |
| eTable 2 (sensitivity analyses, MIMIC-IV) | `mimic/02` | sheets `11_sensitivity`, `18_added_sensitivity` |
| eTable 3 (sensitivity analyses, eICU-CRD) | `eicu/01`, `eicu/02` | sheets `14_sensitivity`, `19_robustness`; `eicu_timing_sensitivity.csv` |
| eTable 4 (0-8 hr vs >8-24 hr windows and complementary analyses) | `mimic/02`, `mimic/03`, `mimic/04`, `eicu/01`, `eicu/02`, `eicu/03`, `pooled/01` | odds ratios: primary sheets; cause-specific hazard ratios: sheets `08_csCox` (MIMIC-IV), `09_csCox_time_split` (eICU-CRD); multinomial model and ratio of odds ratios: `window_test_*.csv`, `pooled_estimates.csv`; weighted cumulative incidence: `fig2_cif_*.csv`; risk differences and E-value: sheets `07b_CIF_IPW_RD_boot`, `05_IPW_ATT` (MIMIC-IV) |
| eTable 5 (cumulative incidence and competing events) | `mimic/02`, `mimic/03`, `eicu/01`, `eicu/02` | sheets `07_CIF_summary`, `07b_CIF_IPW_RD_boot` (MIMIC-IV), `10_CIF_8h_24h` (eICU-CRD); `fig2_cif_*.csv` |
| eTable 6 (sodium outcomes; testing frequency) | `mimic/08`, `mimic/02`, `eicu/01` | `sodium_harmonized.csv`; sheets `12_clinical`, `12b_na150_detection` (MIMIC-IV), `13_sodium_clinical` (eICU-CRD) |
| eTable 7 (hypernatremia by early polyuria and timing) | `mimic/07`, `eicu/06` | sheets `Na_*` |
| eTable 8 (fluid-balance sensitivity analyses) | `mimic/06`, `mimic/08`, `eicu/05`, `pooled/01` | fluid-balance workbooks; `sodium_harmonized.csv`; pooled values and I2 in `pooled_secondary.csv` |
| eTable 9 (polyuria by tertile of fluid overload) | `mimic/06`, `eicu/05` | sheets `05_c_FO_tertiles`, `05b_c_interaction` (MIMIC-IV); `c_tertiles`, `c_interaction` (eICU-CRD) |
| eTable 10 (co-interventions after T0) | `mimic/02`, `eicu/01` | sheets `13_cointerventions` (MIMIC-IV), `16_cointerventions` (eICU-CRD) |
| eTable 11 (stays with all 24 hourly urine bins recorded, MIMIC-IV) | `mimic/09`; `mimic/02` | `etable11_full_charting.csv`; the corresponding odds ratios are row "(s3)" of sheet `18_added_sensitivity` |
| eTable 12 (subgroups: sepsis, cardiac surgery) | `mimic/02`, `eicu/01` | sheets `10_subgroups` (MIMIC-IV), `15_subgroups` (eICU-CRD) |
| eTable 13 (polyuria relative to the vasopressin stop; closed cohort; cumulative incidence) | `mimic/07`, `eicu/06`, `eicu/07` | sheets `VPstop_*`; `closed_cohort.csv`, `vpstop_cif.csv`, `risk_table.csv` |
| eTable 14 (characteristics by usage pattern and stopping order) | `mimic/05`, `eicu/04` | sheets `03_table1_5groups` (MIMIC-IV), `03_background_5grp` (eICU-CRD) |
| eTable 15 (polyuria and hypernatremia by usage pattern and stopping order) | `mimic/05`, `mimic/08`, `eicu/04`, `pooled/01` | sheets `06_outcomes` (MIMIC-IV), `06_outcomes_OR`, `07b_sodium_OR` (eICU-CRD); rows `5grp: ...` of `sodium_harmonized.csv`; pooled values and I2 in `pooled_secondary.csv` |
| eTable 16 (NE-first vs VP-first by time anchor, overlap weighting) | `mimic/07`, `eicu/06` | sheets `Order_*`; `order_estimates.csv` |
| eTable 17 (excluding switches from norepinephrine to vasopressin) | `mimic/07`, `eicu/07` | `switch_sensitivity.csv` |
| eFigure 1 (fluid balance and early polyuria) | `mimic/06`, `eicu/05`, `pooled/01` | sheets `05_c_FO_tertiles` (MIMIC-IV), `c_tertiles` (eICU-CRD); `pooled_secondary.csv` |
| eFigure 2 (time points and populations) | no output (schematic drawn without data) | |
| eFigure 3 (open cohort around the vasopressin stop; cumulative incidence) | `mimic/07`, `eicu/06` | `vpstop_hourly.csv`, `vpstop_cif.csv`, `vpstop_t0b_hist.csv` |
| eFigure 4 (odds ratios by usage pattern and stopping order) | `mimic/05`, `mimic/08`, `eicu/04`, `pooled/01` | as eTable 15 |

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
`eicu/02`; 200 in `mimic/06`-`08` and `eicu/05`-`07`; 10 imputations). Random seeds are fixed in
the code.

## Running

```bash
export MIMIC_DIR=/path/to/mimiciv/3.1
export EICU_DIR=/path/to/eicu-crd/2.0
./run_all.sh            # or: ./run_all.sh mimic | eicu | pooled
```

Each step can also be run on its own from the repository root, e.g.
`python mimic/02_primary_analysis.py`.

1. MIMIC-IV: `01` -> `02` -> `03`, `04`, `09` (each needs 02); `05` (needs 01); `06` (needs 02 and 05); `07`, `08` (need 02, 05 and 06)
2. eICU-CRD: `01`; `02` (re-uses the extracts of 01); `03` (needs 01); `04`; `05` (needs 01 and 04); `06` (needs 01, 04 and 05); `07` (needs 01 and 04)
3. Pooled estimates: `pooled/01` (needs `mimic/02`, `04`, `05`, `06`, `08` and `eicu/01`, `03`, `04`, `05`)

The two database chains are independent of each other and can be run at the same time. A later
step reads the cohort tables that earlier steps wrote to `WORK_DIR` and imports the shared
functions from the modules in its folder.

Several steps end with a block of checks that compare key counts and estimates with the values
reported in the manuscript and stop, or flag the row, when a value differs. These expected values
hold for the default settings only.

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
not timed. Disk space: about 3.2 GB in `WORK_DIR`.

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
