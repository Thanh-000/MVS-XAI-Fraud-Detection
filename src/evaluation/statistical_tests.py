"""Statistical comparison helpers used by the evaluation harness."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2, ttest_rel


def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar's test on paired binary predictions."""
    y_true = np.asarray(y_true).astype(int)
    pred_a = np.asarray(pred_a).astype(int)
    pred_b = np.asarray(pred_b).astype(int)

    a_correct = pred_a == y_true
    b_correct = pred_b == y_true

    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    discordant = b + c

    if discordant == 0:
        return {
            "b": b,
            "c": c,
            "statistic": 0.0,
            "p_value": 1.0,
            "discordant": 0,
        }

    statistic = (abs(b - c) - 1.0) ** 2 / discordant
    p_value = float(1.0 - chi2.cdf(statistic, df=1))
    return {
        "b": b,
        "c": c,
        "statistic": float(statistic),
        "p_value": p_value,
        "discordant": discordant,
    }


def cohens_d(sample_a, sample_b):
    """Paired Cohen's d for repeated-seed comparisons."""
    sample_a = np.asarray(sample_a, dtype=float)
    sample_b = np.asarray(sample_b, dtype=float)
    if len(sample_a) != len(sample_b):
        raise ValueError("Cohen's d requires paired samples with equal length.")
    if len(sample_a) < 2:
        return float("nan")

    diff = sample_a - sample_b
    std = np.std(diff, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(diff) / std)


def paired_ttest(sample_a, sample_b):
    """Paired t-test used alongside Cohen's d for multi-seed comparisons."""
    sample_a = np.asarray(sample_a, dtype=float)
    sample_b = np.asarray(sample_b, dtype=float)
    if len(sample_a) != len(sample_b):
        raise ValueError("Paired t-test requires equal-length paired samples.")
    if len(sample_a) < 2:
        return {"statistic": float("nan"), "p_value": float("nan")}

    stat, p_value = ttest_rel(sample_a, sample_b)
    return {"statistic": float(stat), "p_value": float(p_value)}
