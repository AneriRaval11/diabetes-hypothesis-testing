"""Reusable statistical helpers for the diabetes hypothesis-testing project."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import confint_proportions_2indep, test_proportions_2indep

ALPHA = 0.05
SEED = 42


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["income_level"] = pd.Categorical(
        df["income_level"], ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"], ordered=True
    )
    df["education_level"] = pd.Categorical(
        df["education_level"], ["No formal", "Highschool", "Graduate", "Postgraduate"], ordered=True
    )
    return df


def cohens_d(a, b) -> tuple[float, float, float]:
    """Cohen's d (pooled SD) for a - b, with a large-sample 95% CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / sp
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    return d, d - 1.96 * se, d + 1.96 * se


def welch_test(a, b) -> dict:
    """Two-sided Welch t-test of mean(a) vs mean(b); CI for the difference; Cohen's d."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    res = stats.ttest_ind(a, b, equal_var=False)
    diff = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    tcrit = stats.t.ppf(0.975, dof)
    d, d_lo, d_hi = cohens_d(a, b)
    return dict(n=len(a) + len(b), estimate=diff, ci_low=diff - tcrit * se, ci_high=diff + tcrit * se,
                effect_size=d, es_low=d_lo, es_high=d_hi, p_raw=res.pvalue)


def tost_mean(a, b, margin: float) -> dict:
    """Two one-sided tests: is |mean(a) - mean(b)| < margin? (equivalence)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    p_lower = 1 - stats.norm.cdf((diff + margin) / se)   # H0: diff <= -margin
    p_upper = stats.norm.cdf((diff - margin) / se)       # H0: diff >= +margin
    return dict(margin=margin, p_tost=max(p_lower, p_upper),
                ci90_low=diff - 1.645 * se, ci90_high=diff + 1.645 * se)


def two_prop(y_a, y_b) -> dict:
    """Risk difference and odds ratio (a vs b) for binary outcomes, plus chi-square p."""
    ya, yb = np.asarray(y_a), np.asarray(y_b)
    n1, n2, x1, x2 = len(ya), len(yb), int(ya.sum()), int(yb.sum())
    p1, p2 = x1 / n1, x2 / n2
    ct = np.array([[x1, n1 - x1], [x2, n2 - x2]])
    _, p, _, _ = stats.chi2_contingency(ct, correction=False)
    odds = (x1 * (n2 - x2)) / ((n1 - x1) * x2)
    se_log = np.sqrt(1 / x1 + 1 / (n1 - x1) + 1 / x2 + 1 / (n2 - x2))
    lo, hi = confint_proportions_2indep(x1, n1, x2, n2, method="wald", compare="diff")
    return dict(n=n1 + n2, p1=p1, p2=p2, estimate=p1 - p2, ci_low=lo, ci_high=hi,
                odds_ratio=odds, or_low=np.exp(np.log(odds) - 1.96 * se_log),
                or_high=np.exp(np.log(odds) + 1.96 * se_log), p_raw=p)


def tost_prop(y_a, y_b, margin: float) -> dict:
    """Equivalence test for a difference in proportions within +/- margin."""
    ya, yb = np.asarray(y_a), np.asarray(y_b)
    x1, n1, x2, n2 = int(ya.sum()), len(ya), int(yb.sum()), len(yb)
    p_lo = test_proportions_2indep(x1, n1, x2, n2, value=-margin, method="wald", compare="diff",
                                   alternative="larger").pvalue
    p_hi = test_proportions_2indep(x1, n1, x2, n2, value=margin, method="wald", compare="diff",
                                   alternative="smaller").pvalue
    d = x1 / n1 - x2 / n2
    se = np.sqrt((x1 / n1) * (1 - x1 / n1) / n1 + (x2 / n2) * (1 - x2 / n2) / n2)
    return dict(margin=margin, p_tost=max(p_lo, p_hi), ci90_low=d - 1.645 * se, ci90_high=d + 1.645 * se)


def cramers_v(table: pd.DataFrame) -> tuple[float, float]:
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1)))), float(p)


def holm(pvals) -> np.ndarray:
    return multipletests(pvals, alpha=ALPHA, method="holm")[1]


def interpret_d(d: float) -> str:
    d = abs(d)
    return "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"
