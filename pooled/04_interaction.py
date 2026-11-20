#!/usr/bin/env python3
"""
04_interaction.py - pooled estimates of the group x polyuria interaction on the sodium rise
(mimic/12_interaction.py, eicu/10_interaction.py; post hoc, exploratory; manuscript v16).

Pooling: fixed-effect inverse-variance weighting of the database-specific estimates at full
precision. Odds ratios are pooled on the log scale and differences of standardised proportions
on the proportion scale, each with the standard error stored by the database script
(est_raw, se_raw); I2 from Cochran's Q (early_late_helpers.pool_fixed, the function used by
pooled/03_early_late_extra.py). The standardised proportions of the single cells are not pooled.

Each estimate is pooled three times: with the clustered SE of both databases (v15 convention),
with the model-based SE of both databases, and with the larger of the two SEs in each database
("wider interval in each database"). The third is the one to be used for interpretation: in
eICU-CRD the hospital-clustered SE of the interaction term is about one third smaller than the
model-based SE (few hospitals contribute VP stays with polyuria), so the clustered SE alone
would overstate the precision of the eICU-CRD estimate and its weight in the pooled estimate.

Before pooling, the script checks that the reproduction checks of both database scripts all
matched, and that the pooled odds ratios of the three-category model of analysis B
(pooled/02_early_late.py) are reproduced from the database files (2.51, 2.04-3.10 and
0.89, 0.70-1.14). It stops when a check fails.

Inputs   OUT_DIR/mimic_12/ and OUT_DIR/eicu_10/: interaction_estimates.csv, reproduction_checks.csv
         OUT_DIR/mimic_10/B_multinomial.csv, OUT_DIR/eicu_08/B_multinomial.csv
Outputs  OUT_DIR/pooled_04/reproduction_checks.csv, interaction_pooled.csv
Run:     python pooled/04_interaction.py   (after mimic/12 and eicu/10)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from early_late_helpers import pool_fixed  # noqa: E402


def main():
    o = lambda *p: os.path.join(config.OUT_DIR, *p)
    out = config.out_dir("pooled_04")

    # ---- 0. checks -----------------------------------------------------------------------------
    rep = []
    for db, f in (("MIMIC-IV", o("mimic_12", "reproduction_checks.csv")), ("eICU-CRD", o("eicu_10", "reproduction_checks.csv"))):
        r = pd.read_csv(f)
        rep.append(dict(check=f"{db}: reproduction checks of the database script all matched", value=str(bool(r.match.all())),
                        expected="True", match=bool(r.match.all())))
    M, E = pd.read_csv(o("mimic_10", "B_multinomial.csv")), pd.read_csv(o("eicu_08", "B_multinomial.csv"))
    for c, exp in ((2, "2.51 (2.04–3.10)"), (1, "0.89 (0.70–1.14)")):
        m = M[(M.analysis == "main: rise >= 5 or peak >= 150") & (M.category == c)].iloc[0]
        e = E[(E.analysis == "main: rise >= 5 or peak >= 150") & (E.category == c)].iloc[0]
        q, sp, _, _ = pool_fixed([np.log(m.OR), np.log(e.OR)], [(np.log(m.hi) - np.log(m.lo)) / 3.92, (np.log(e.hi) - np.log(e.lo)) / 3.92])
        val = "%.2f (%.2f–%.2f)" % (np.exp(q), np.exp(q - 1.96 * sp), np.exp(q + 1.96 * sp))
        rep.append(dict(check=f"analysis B pooled OR, category {c}", value=val, expected=exp, match=val == exp))
    REP = pd.DataFrame(rep)
    REP.to_csv(os.path.join(out, "reproduction_checks.csv"), index=False)
    print(REP.to_string(index=False))
    if not REP.match.all():
        print("REPRODUCTION FAILED - pooling stopped")
        sys.exit(1)

    # ---- 1. pooling ----------------------------------------------------------------------------
    M, E = pd.read_csv(o("mimic_12", "interaction_estimates.csv")), pd.read_csv(o("eicu_10", "interaction_estimates.csv"))
    kind = lambda x: "model-based" if x == "model-based" else "clustered (v15 convention)"
    M["se_kind"], E["se_kind"] = M.se_type.map(kind), E.se_type.map(kind)
    rows = []
    for (dn, term), mm_ in M.groupby(["definition", "term"], sort=False):
        ee = E[(E.definition == dn) & (E.term == term)]
        assert len(mm_) == 2 and len(ee) == 2, (dn, term)
        variants = [(k, mm_[mm_.se_kind == k].iloc[0], ee[ee.se_kind == k].iloc[0]) for k in ("clustered (v15 convention)", "model-based")]
        # third variant: in each database the SE type with the larger SE (wider interval)
        variants.append(("wider interval in each database", mm_.loc[mm_.se_raw.idxmax()], ee.loc[ee.se_raw.idxmax()]))
        for k, m, e in variants:
            r = dict(definition=dn, se_type=k, scale=m.scale, term=term, mimic=m.text, mimic_p=m.p, mimic_se_type=m.se_kind,
                     eicu=e.text, eicu_p=e.p, eicu_se_type=e.se_kind)
            if term.startswith("standardised proportion"):
                r["note"] = "not pooled (database-specific standardised proportion)"
                rows.append(r)
                continue
            q, sp, p, i2 = pool_fixed([m.est_raw, e.est_raw], [m.se_raw, e.se_raw])
            if m.raw_unit == "log odds ratio":
                est, lo, hi = np.exp(q), np.exp(q - 1.96 * sp), np.exp(q + 1.96 * sp)
                txt = "%.2f (%.2f–%.2f)" % (est, lo, hi)
            else:
                est, lo, hi = 100 * q, 100 * (q - 1.96 * sp), 100 * (q + 1.96 * sp)
                txt = "%.1f (%.1f to %.1f)" % (est, lo, hi)
            r.update(pooled=txt, pooled_p=p, i2_percent=round(i2), est=est, lo=lo, hi=hi)
            rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(out, "interaction_pooled.csv"), index=False)
    print(T[["definition", "se_type", "term", "mimic", "eicu", "pooled", "pooled_p", "i2_percent"]].to_string(index=False))


if __name__ == "__main__":
    main()
