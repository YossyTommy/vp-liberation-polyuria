#!/usr/bin/env python3
"""
05_se_check.py - pooled estimates of the post hoc sodium-rise models with the "wider interval"
rule (manuscript v17). Inputs are the two standard-error versions of every estimate written by
mimic/13_se_check.py and eicu/11_se_check.py (same fits as v16; no new analysis).

For each estimate three pooled values are written:
  clustered                         clustered SE in both databases (the values of v16)
  model-based                       model-based SE in both databases
  wider interval in each database   in each database the SE type with the wider interval; this is
                                    the rule already used for the interaction analysis
                                    (pooled/04_interaction.py) and the value reported in v17

Pooling follows pooled/02_early_late.py and pooled/03_early_late_extra.py: fixed-effect
inverse-variance weighting on the log scale with SE = (ln upper - ln lower) / 3.92; I2 from
Cochran's Q.

Before pooling, the script checks that the reproduction checks of both database scripts all
matched and that the "clustered" pooled values equal the pooled values of v16
(pooled_02/B_multinomial_pooled.csv, pooled_03/E1_multinomial5_pooled.csv). It stops otherwise.

It also writes a comparison of the v16 intervals (clustered) with the v17 intervals (wider) and
flags every estimate for which the interval changes side with respect to 1 (a change in the
direction of a conclusion).

Outputs  OUT_DIR/pooled_05/reproduction_checks.csv, se_check_pooled.csv, se_check_comparison.csv
Run:     python pooled/05_se_check.py   (after mimic/13 and eicu/11)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from early_late_helpers import pool_fixed  # noqa: E402

FMT = "%.2f (%.2f–%.2f)"


def width(r):
    return np.log(r.hi) - np.log(r.lo)


def side(lo, hi):
    return "above 1" if lo > 1 else ("below 1" if hi < 1 else "includes 1")


def main():
    o = lambda *p: os.path.join(config.OUT_DIR, *p)
    out = config.out_dir("pooled_05")
    M, E = pd.read_csv(o("mimic_13", "se_check_estimates.csv")), pd.read_csv(o("eicu_11", "se_check_estimates.csv"))
    rows = []
    for (model, dn, term), mm_ in M.groupby(["model", "definition", "term"], sort=False):
        ee = E[(E.model == model) & (E.definition == dn) & (E.term == term)]
        assert len(mm_) == 2 and len(ee) == 2, (model, dn, term)
        pick = lambda d, t: d[d.se_type == t].iloc[0]
        variants = [("clustered", pick(mm_, "clustered"), pick(ee, "clustered")), ("model-based", pick(mm_, "model-based"), pick(ee, "model-based")),
                    ("wider interval in each database", mm_.loc[mm_.apply(width, axis=1).idxmax()], ee.loc[ee.apply(width, axis=1).idxmax()])]
        for k, m, e in variants:
            q, sp, p, i2 = pool_fixed([np.log(m.OR), np.log(e.OR)], [width(m) / 3.92, width(e) / 3.92])
            est, lo, hi = np.exp(q), np.exp(q - 1.96 * sp), np.exp(q + 1.96 * sp)
            rows.append(dict(model=model, definition=dn, term=term, se_type=k, mimic=FMT % (m.OR, m.lo, m.hi), mimic_p=m.p, mimic_se_type=m.se_type,
                             eicu=FMT % (e.OR, e.lo, e.hi), eicu_p=e.p, eicu_se_type=e.se_type, pooled=FMT % (est, lo, hi), pooled_p=p,
                             i2_percent=round(i2), est=est, lo=lo, hi=hi, mimic_lo=m.lo, mimic_hi=m.hi, eicu_lo=e.lo, eicu_hi=e.hi))
    T = pd.DataFrame(rows)

    # ---- checks ------------------------------------------------------------------------------------
    rep = []
    for db, f in (("MIMIC-IV", o("mimic_13", "reproduction_checks.csv")), ("eICU-CRD", o("eicu_11", "reproduction_checks.csv"))):
        r = pd.read_csv(f)
        rep.append(dict(check=f"{db}: reproduction checks of the database script all matched", value=str(bool(r.match.all())), expected="True",
                        match=bool(r.match.all())))
    B16 = pd.read_csv(o("pooled_02", "B_multinomial_pooled.csv"))
    E16 = pd.read_csv(o("pooled_03", "E1_multinomial5_pooled.csv"))
    C = T[T.se_type == "clustered"]
    bad = []
    for _, r in C.iterrows():
        if r.model == "three-category":
            c = {"polyuria without Na rise": "1", "polyuria with Na rise": "2"}[r.term]
            ref = B16[(B16.definition == r.definition) & (B16.category.astype(str) == c)].iloc[0]
        else:
            ref = E16[(E16.analysis == r.definition) & (E16.term == r.term)].iloc[0]
        if not (r.pooled == ref.pooled and r.mimic == ref.mimic and r.eicu == ref.eicu and int(r.i2_percent) == int(ref.i2_percent)):
            bad.append((r.model, r.definition, r.term, r.pooled, ref.pooled))
    rep.append(dict(check="clustered values (MIMIC-IV, eICU-CRD, pooled, I2) equal the v16 pooled tables for all %d estimates" % len(C),
                    value=str(len(bad) == 0), expected="True", match=len(bad) == 0))
    REP = pd.DataFrame(rep)
    REP.to_csv(os.path.join(out, "reproduction_checks.csv"), index=False)
    print(REP.to_string(index=False))
    if not REP.match.all():
        print("REPRODUCTION FAILED - stopped", bad[:5])
        sys.exit(1)
    T.to_csv(os.path.join(out, "se_check_pooled.csv"), index=False)

    # ---- comparison v16 (clustered) vs v17 (wider interval) ------------------------------------------
    W = T[T.se_type == "wider interval in each database"].set_index(["model", "definition", "term"])
    cmp_rows = []
    for _, r in C.iterrows():
        w = W.loc[(r.model, r.definition, r.term)]
        cmp_rows.append(dict(model=r.model, definition=r.definition, term=r.term,
                             mimic_v16=r.mimic, mimic_v17=w.mimic, mimic_v17_se_type=w.mimic_se_type,
                             eicu_v16=r.eicu, eicu_v17=w.eicu, eicu_v17_se_type=w.eicu_se_type,
                             pooled_v16=r.pooled, pooled_v17=w.pooled, i2_v16=int(r.i2_percent), i2_v17=int(w.i2_percent),
                             mimic_side_v16=side(r.mimic_lo, r.mimic_hi), mimic_side_v17=side(w.mimic_lo, w.mimic_hi),
                             eicu_side_v16=side(r.eicu_lo, r.eicu_hi), eicu_side_v17=side(w.eicu_lo, w.eicu_hi),
                             pooled_side_v16=side(r.lo, r.hi), pooled_side_v17=side(w.lo, w.hi)))
    K = pd.DataFrame(cmp_rows)
    for s in ("mimic", "eicu", "pooled"):
        K[f"{s}_side_changed"] = K[f"{s}_side_v16"] != K[f"{s}_side_v17"]
    K.to_csv(os.path.join(out, "se_check_comparison.csv"), index=False)
    print(K[["model", "definition", "term", "mimic_v16", "mimic_v17", "eicu_v16", "eicu_v17", "pooled_v16", "pooled_v17", "i2_v16", "i2_v17"]].to_string(index=False))
    ch = K[K.mimic_side_changed | K.eicu_side_changed | K.pooled_side_changed]
    print("\nESTIMATES WHOSE INTERVAL CHANGES SIDE WITH RESPECT TO 1: %d" % len(ch))
    if len(ch):
        print(ch[["model", "definition", "term", "mimic_side_v16", "mimic_side_v17", "eicu_side_v16", "eicu_side_v17", "pooled_side_v16", "pooled_side_v17"]].to_string(index=False))


if __name__ == "__main__":
    main()
