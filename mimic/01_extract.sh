#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 01_extract.sh
# Streams the MIMIC-IV v3.1 csv.gz files once and writes the row subsets that
# 02_primary_analysis.py and the later MIMIC scripts read.
#
# Output (WORK_DIR/mimic/data, written without a header unless stated):
#   vaso2.csv        subject_id,stay_id,starttime,endtime,itemid,amount,rate,rateuom,patientweight
#                    inputevents rows of vasopressin 222315, norepinephrine 221906,
#                    phenylephrine 221749, epinephrine 221289, dopamine 221662,
#                    dobutamine 221653, angiotensin II 229709/229764; "Rewritten" rows dropped
#   diuretic_iv.csv  stay_id,starttime,itemid,amount - intravenous loop diuretics
#                    (221794 furosemide, 228340 furosemide 250/50, 229639 bumetanide), amount > 0
#   urine.csv        stay_id,charttime,itemid,value - outputevents urine items
#   rrt.csv          stay_id,starttime,endtime,itemid - renal replacement therapy procedures
#   labs.csv         subject_id,hadm_id,itemid,charttime,valuenum - sodium 50983/52623/50824,
#                    osmolality 50964/52031/51701, urine osmolality 51093/52044,
#                    creatinine 50912, urine sodium 51100/51099
#   labs_extra.csv   same layout - glucose 50931/50809, lactate 50813
#   map.csv          stay_id,charttime,itemid,valuenum - mean arterial pressure
#                    220052/220181/225312, values in (10, 200) mm Hg
#   icustays.csv, patients.csv, admissions.csv   decompressed copies (with header)
#
# Usage:  MIMIC_DIR=/path/to/mimiciv/3.1 WORK_DIR=/path/to/work bash mimic/01_extract.sh
# Time:   about 1 hour with the BSD awk of macOS (labevents 2.6 GB and chartevents 3.5 GB
#         dominate); mawk or gawk is used when available but was not timed.
# Column numbers refer to the MIMIC-IV v3.1 csv layouts; none of the columns used
# contains a comma.
# ---------------------------------------------------------------------------
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MIMIC="${MIMIC_DIR:-$ROOT/data/mimiciv/3.1}"
OUT="${WORK_DIR:-$ROOT/work}/mimic/data"
mkdir -p "$OUT"
if   command -v mawk >/dev/null; then AWK=mawk
elif command -v gawk >/dev/null; then AWK=gawk
else AWK=awk; fi
if   command -v gzcat >/dev/null; then ZCAT=gzcat
else ZCAT="gunzip -c"; fi
echo "awk=$AWK  zcat=$ZCAT  mimic=$MIMIC  out=$OUT"
step () { echo; echo "[$(date +%H:%M:%S)] $1"; }

# --- 1. inputevents: vasopressors and intravenous loop diuretics -------------
# cols: 1 subject_id, 3 stay_id, 5 starttime, 6 endtime, 8 itemid, 9 amount,
#       11 rate, 12 rateuom, 19 patientweight, 24 statusdescription
step "1/5 inputevents: vaso2, diuretic_iv"
rm -f "$OUT/vaso2.csv" "$OUT/diuretic_iv.csv"
$ZCAT "$MIMIC/icu/inputevents.csv.gz" | $AWK -F, -v out="$OUT" '
  BEGIN{w[222315];w[221906];w[221749];w[221289];w[221662];w[221653];w[229709];w[229764];
        d[221794];d[228340];d[229639]}
  NR>1 {
    if (($8 in w) && $24!="Rewritten")
      print $1","$3","$5","$6","$8","$9","$11","$12","$19 > (out "/vaso2.csv");
    if (($8 in d) && $9+0>0) print $3","$5","$8","$9 > (out "/diuretic_iv.csv");
  }'

# --- 2. outputevents: urine ----------------------------------------------------
# cols: 3 stay_id, 5 charttime, 7 itemid, 8 value
step "2/5 outputevents: urine"
$ZCAT "$MIMIC/icu/outputevents.csv.gz" | $AWK -F, '
  BEGIN{u[226559];u[226560];u[226561];u[226563];u[226564];u[226565];u[226566];
        u[226567];u[226557];u[226558];u[227489];u[226627];u[226631]}
  NR>1 && ($7 in u) && $8!="" {print $3","$5","$7","$8}
' > "$OUT/urine.csv"

# --- 3. procedureevents: renal replacement therapy -----------------------------
# cols: 3 stay_id, 5 starttime, 6 endtime, 8 itemid
step "3/5 procedureevents: rrt"
$ZCAT "$MIMIC/icu/procedureevents.csv.gz" | $AWK -F, '
  BEGIN{r[225441];r[225802];r[225803];r[225805];r[225809];r[225955];r[224270]}
  NR>1 && ($8 in r) {print $3","$5","$6","$8}
' > "$OUT/rrt.csv"

# --- 4. labevents: laboratory values -------------------------------------------
# cols: 2 subject_id, 3 hadm_id, 5 itemid, 7 charttime, 10 valuenum
step "4/5 labevents: labs, labs_extra (several minutes)"
rm -f "$OUT/labs.csv" "$OUT/labs_extra.csv"
$ZCAT "$MIMIC/hosp/labevents.csv.gz" | $AWK -F, -v out="$OUT" '
  BEGIN{k[50983];k[52623];k[50824];k[50964];k[52031];k[51701];
        k[51093];k[52044];k[50912];k[51100];k[51099];
        x[50931];x[50809];x[50813]}
  NR>1 && $10!="" {
    if ($5 in k) print $2","$3","$5","$7","$10 > (out "/labs.csv");
    if ($5 in x) print $2","$3","$5","$7","$10 > (out "/labs_extra.csv");
  }'

# --- 5. chartevents: mean arterial pressure ------------------------------------
# cols: 3 stay_id, 5 charttime, 7 itemid, 9 valuenum
step "5/5 chartevents: map (several minutes)"
$ZCAT "$MIMIC/icu/chartevents.csv.gz" | $AWK -F, '
  BEGIN{m[220052];m[220181];m[225312]}
  NR>1 && ($7 in m) && $9!="" && $9+0>10 && $9+0<200 {print $3","$5","$7","$9}
' > "$OUT/map.csv"

# --- small tables --------------------------------------------------------------
for f in icu/icustays hosp/patients hosp/admissions; do
  $ZCAT "$MIMIC/$f.csv.gz" > "$OUT/$(basename "$f").csv"
done

echo
for f in vaso2 diuretic_iv urine rrt labs labs_extra map icustays patients admissions; do
  echo "  $f.csv  $(wc -l < "$OUT/$f.csv") rows"
done
echo "done. Next: python mimic/02_primary_analysis.py"
