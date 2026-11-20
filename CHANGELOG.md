# Changelog

## 2026-11-20 - additions for the revised manuscript

The primary pipeline of the first public version is unchanged: every file of that version except `README.md` is
byte-identical.

### Added

* Post hoc exploratory analyses (run with `run_exploratory_v16.sh`):
  `mimic/10_early_late.py`, `mimic/11_early_late_extra.py`, `mimic/12_interaction.py`,
  `eicu/08_early_late.py`, `eicu/09_early_late_extra.py`, `eicu/10_interaction.py`,
  `pooled/02_early_late.py`, `pooled/03_early_late_extra.py`, `pooled/04_interaction.py`.
  They cover polyuria with and without a rise in serum sodium (three- and five-category outcomes), the interaction
  between group and polyuria, stays with polyuria by time of onset, hospital mortality, duration and rate of the
  vasopressin infusion, urine tests, and the Holm adjustment of the four secondary comparisons.
* Additions without a new analysis (run with `run_additions_v17.sh`):
  `mimic/13_se_check.py`, `eicu/11_se_check.py`, `pooled/05_se_check.py` (clustered and model-based standard errors of
  the sodium-rise models; the wider interval in each database is reported);
  `mimic/14_fig4_bands.py`, `eicu/12_fig4_bands.py` (stay-level bootstrap confidence bands of the closed-cohort time
  course).
* Shared functions: `early_late_helpers.py`, `early_late_bands.py`.
* Batch scripts: `run_exploratory_v16.sh`, `run_additions_v17.sh`.
* `CHANGELOG.md`.

### Changed

* Header comments of `mimic/10`, `mimic/11`, `mimic/12`, `eicu/08`, `eicu/09`, `eicu/10` and `pooled/02` reworded to
  state that these analyses are reported in the manuscript as post hoc exploratory analyses (comments only; no
  change to the code or to any result).
* `README.md`: repository layout with the added files; table of scripts and manuscript items renamed to the current
  item names (Supplementary Table S1-S26, Supplementary Figure S1-S5) and extended to the added tables and figures;
  sources of Figures 3 and 4 updated; new section on primary, secondary and post hoc analyses; run order of the added
  scripts; environment variables `NBOOT_EL` and `NBOOT_BANDS`; outputs of the added scripts; a sentence stating that
  plotting code is not included.

### Removed

* Nothing.

## First public version

Primary pipeline: `run_all.sh`, `config.py`, `mimic/01`-`09`, `eicu/01`-`07`, `pooled/01` and their modules.
