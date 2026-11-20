"""
early_late_helpers.py - descriptive helpers shared by the exploratory scripts added after v15
(mimic/11_early_late_extra.py, eicu/09_early_late_extra.py). Not used by the manuscript analyses.

The functions are those defined inside mimic/10_early_late.py and eicu/08_early_late.py (copied
unchanged): Wilson intervals, masking of cells < 10, median [IQR] and proportion tables, percentile
bootstrap of differences (stays resampled within group) and fixed-effect pooling.
Optional environment variable: NBOOT_EL (bootstrap replicates, default 2000).
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

NBOOT = int(os.environ.get("NBOOT_EL", 2000))
SEED = 20261117


def wilson(k, n, z=1.959964):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def small(k, n=None):
    """True when a count (or its complement in n) is below 10."""
    return k < 10 or (n is not None and n - k < 10)


def fmt_prop(k, n):
    if n < 10:
        return "n<10", np.nan, np.nan, np.nan
    if small(k, n):
        return ("<10" if k < 10 else "masked (n-k <10)"), np.nan, np.nan, np.nan
    lo, hi = wilson(k, n)
    return "%d/%d (%.1f%%; %.1f–%.1f)" % (k, n, 100 * k / n, 100 * lo, 100 * hi), 100 * k / n, 100 * lo, 100 * hi


def fmt_med(x, dec=1):
    x = pd.Series(x).dropna()
    if len(x) < 10:
        return "n<10", np.nan, np.nan, np.nan, len(x)
    q = x.quantile([.25, .5, .75]).values
    f = "%%.%df [%%.%df–%%.%df]" % (dec, dec, dec)
    return f % (q[1], q[0], q[2]), q[1], q[0], q[2], len(x)


def describe(df, gcol, spec, groups):
    rows = []
    for v, lab, kind, dec in spec:
        r = dict(variable=v, label=lab, kind=kind)
        for g in groups:
            s = df.loc[df[gcol] == g, v]
            if kind == "bin":
                s = s.dropna()
                txt, p, lo, hi = fmt_prop(int(s.sum()), len(s))
                r[g] = txt
                r[g + "_pct"], r[g + "_lo"], r[g + "_hi"] = p, lo, hi
                r[g + "_n"] = len(s) if len(s) >= 10 else "<10"
            else:
                txt, m, q1, q3, n = fmt_med(s, dec)
                r[g] = txt
                r[g + "_median"], r[g + "_q1"], r[g + "_q3"] = m, q1, q3
                r[g + "_n"] = n if n >= 10 else "<10"
        rows.append(r)
    return pd.DataFrame(rows)


def boot_contrasts(df, gcol, spec, pairs, nboot=NBOOT, seed=SEED):
    """Difference (group a - group b) of medians (continuous) or proportions (binary, percentage
    points) with percentile bootstrap 95% CIs; stays resampled with replacement within each group."""
    rng = np.random.default_rng(seed)
    vars_ = [s[0] for s in spec]
    kinds = {s[0]: s[2] for s in spec}
    arrs = {g: df.loc[df[gcol] == g, vars_].to_numpy(float) for g in set(sum(map(list, pairs), []))}

    def stat(a):
        out = []
        for j, v in enumerate(vars_):
            x = a[:, j]
            x = x[~np.isnan(x)]
            if len(x) == 0:
                out.append(np.nan)
            elif kinds[v] == "bin":
                out.append(100 * x.mean())
            else:
                out.append(np.median(x))
        return np.array(out)

    rows = []
    for a, b in pairs:
        A, B = arrs[a], arrs[b]
        est = stat(A) - stat(B)
        bs = np.empty((nboot, len(vars_)))
        for i in range(nboot):
            bs[i] = stat(A[rng.integers(0, len(A), len(A))]) - stat(B[rng.integers(0, len(B), len(B))])
        lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
        for j, (v, lab, kind, dec) in enumerate(spec):
            na_, nb_ = (~np.isnan(A[:, j])).sum(), (~np.isnan(B[:, j])).sum()
            ka = np.nansum(A[:, j]) if kind == "bin" else np.nan
            kb = np.nansum(B[:, j]) if kind == "bin" else np.nan
            masked = na_ < 10 or nb_ < 10 or (kind == "bin" and (small(ka, na_) or small(kb, nb_)))
            unit = "percentage points" if kind == "bin" else "difference in medians"
            rows.append(dict(contrast=f"{a} - {b}", variable=v, label=lab, measure=unit,
                             diff=np.nan if masked else est[j], lo=np.nan if masked else lo[j],
                             hi=np.nan if masked else hi[j],
                             text="masked (cell <10)" if masked else ("%.*f (%.*f to %.*f)" % (dec, est[j], dec, lo[j], dec, hi[j])),
                             n_a=na_ if na_ >= 10 else "<10", n_b=nb_ if nb_ >= 10 else "<10", nboot=nboot))
    return pd.DataFrame(rows)


def box_stats(df, gcol, vars_, groups):
    rows = []
    for v in vars_:
        for g in groups:
            x = df.loc[df[gcol] == g, v].dropna()
            if len(x) < 10:
                rows.append(dict(variable=v, group=g, n="<10"))
                continue
            q = x.quantile([.1, .25, .5, .75, .9]).values
            rows.append(dict(variable=v, group=g, n=len(x), p10=q[0], q1=q[1], median=q[2], q3=q[3], p90=q[4],
                             mean=x.mean(), sd=x.std()))
    return pd.DataFrame(rows)


def pool_fixed(est, se):
    """Fixed-effect inverse-variance pooled estimate, SE, two-sided p and I2 (%)."""
    est, se = np.asarray(est, float), np.asarray(se, float)
    w = 1 / se ** 2
    p_ = (w * est).sum() / w.sum()
    sp = w.sum() ** -.5
    q = float((w * (est - p_) ** 2).sum())
    i2 = 100 * max(0.0, (q - (len(est) - 1)) / q) if q > 0 else 0.0
    return p_, sp, 2 * norm.sf(abs(p_ / sp)), i2
