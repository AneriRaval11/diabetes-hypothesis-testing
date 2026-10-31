"""Hypothesis definitions for the diabetes case study.

Confirmatory family (H1-H7) is corrected with Holm's method; everything else is labelled exploratory.
Outcome definitions
    * hba1c               - continuous glycaemic marker (%)
    * diagnosed_diabetes  - binary, as recorded in the dataset
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis_utils import (ALPHA, SEED, cramers_v, holm, interpret_d, tost_mean, tost_prop, two_prop,
                            welch_test)

WHO_MINUTES = 150          # WHO guideline: >=150 min moderate activity per week
OBESE_BMI = 30
EQUIV_MARGIN_PP = 0.03     # judgement call: gaps under 3 percentage points treated as "practically equal"
EQUIV_MARGIN_HBA1C = 0.10  # HbA1c percentage points


def confirmatory_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (main results table, equivalence-test table)."""
    rows, equiv = [], []

    def add(hid, domain, label, kind, res, es=None, es_label=""):
        rows.append(dict(id=hid, domain=domain, hypothesis=label, outcome=kind, n=res["n"],
                         estimate=res["estimate"], ci_low=res["ci_low"], ci_high=res["ci_high"],
                         effect_size=es if es is not None else res.get("effect_size"),
                         es_low=res.get("es_low", res.get("or_low") if es_label == "Odds ratio" else np.nan),
                         es_high=res.get("es_high", res.get("or_high") if es_label == "Odds ratio" else np.nan),
                         effect_measure=es_label, p_raw=res["p_raw"]))

    # H1 physical activity -> HbA1c
    act = df.loc[df.physical_activity_minutes_per_week >= WHO_MINUTES, "hba1c"]
    inact = df.loc[df.physical_activity_minutes_per_week < WHO_MINUTES, "hba1c"]
    r = welch_test(act, inact)
    add("H1", "Behaviour", f"Meeting WHO activity guideline (>={WHO_MINUTES} min/wk) lowers HbA1c",
        "HbA1c diff (pts), active - inactive", r, es_label="Cohen's d")

    # H2 diet quality -> HbA1c (top vs bottom quartile)
    q1, q3 = df.diet_score.quantile([0.25, 0.75])
    r = welch_test(df.loc[df.diet_score >= q3, "hba1c"], df.loc[df.diet_score <= q1, "hba1c"])
    add("H2", "Behaviour", "Top-quartile diet score is associated with lower HbA1c than bottom quartile",
        "HbA1c diff (pts), Q4 - Q1", r, es_label="Cohen's d")

    # H3 family history -> diagnosis
    r = two_prop(df.loc[df.family_history_diabetes == 1, "diagnosed_diabetes"],
                 df.loc[df.family_history_diabetes == 0, "diagnosed_diabetes"])
    add("H3", "Biology", "Family history raises the probability of a diabetes diagnosis",
        "Risk diff, family hx - none", r, es=r["odds_ratio"], es_label="Odds ratio")

    # H4 obesity -> HbA1c
    r = welch_test(df.loc[df.bmi >= OBESE_BMI, "hba1c"], df.loc[df.bmi < OBESE_BMI, "hba1c"])
    add("H4", "Biology", f"Obesity (BMI >= {OBESE_BMI}) is associated with higher HbA1c",
        "HbA1c diff (pts), obese - non-obese", r, es_label="Cohen's d")

    # H5 income gradient in diagnosis (omnibus chi-square across 5 levels)
    tab = pd.crosstab(df.income_level, df.diagnosed_diabetes)
    v, p = cramers_v(tab)
    lo_, hi_ = df.loc[df.income_level == "Low", "diagnosed_diabetes"], df.loc[df.income_level == "High", "diagnosed_diabetes"]
    r = two_prop(lo_, hi_)
    r.update(p_raw=p, n=len(df))
    add("H5", "Socioeconomic", "Diagnosis prevalence differs across income levels",
        "Risk diff, Low - High income", r, es=v, es_label="Cramer's V")
    e = tost_prop(lo_, hi_, EQUIV_MARGIN_PP)
    equiv.append(dict(id="H5", comparison="Low vs High income (diagnosis rate)", **e))

    # H6 education gradient in diagnosis
    tab = pd.crosstab(df.education_level, df.diagnosed_diabetes)
    v, p = cramers_v(tab)
    a_, b_ = df.loc[df.education_level == "No formal", "diagnosed_diabetes"], df.loc[df.education_level == "Postgraduate", "diagnosed_diabetes"]
    r = two_prop(a_, b_)
    r.update(p_raw=p, n=len(df))
    add("H6", "Socioeconomic", "Diagnosis prevalence differs across education levels",
        "Risk diff, No formal - Postgraduate", r, es=v, es_label="Cramer's V")
    e = tost_prop(a_, b_, EQUIV_MARGIN_PP)
    equiv.append(dict(id="H6", comparison="No formal vs Postgraduate (diagnosis rate)", **e))

    # H7 smoking -> HbA1c
    cur, nev = df.loc[df.smoking_status == "Current", "hba1c"], df.loc[df.smoking_status == "Never", "hba1c"]
    r = welch_test(cur, nev)
    add("H7", "Behaviour", "Current smokers have different HbA1c than never-smokers",
        "HbA1c diff (pts), current - never", r, es_label="Cohen's d")
    e = tost_mean(cur, nev, EQUIV_MARGIN_HBA1C)
    equiv.append(dict(id="H7", comparison="Current vs never smokers (HbA1c)", **e))

    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p_raw.values)
    out["reject_h0"] = out.p_holm < ALPHA
    out["size_label"] = [interpret_d(x) if m == "Cohen's d" else
                         ("strong" if x > 2 else "moderate" if x > 1.5 else "weak") if m == "Odds ratio" else
                         ("negligible" if x < 0.1 else "small" if x < 0.3 else "medium")
                         for x, m in zip(out.effect_size, out.effect_measure)]
    eq = pd.DataFrame(equiv)
    eq["equivalent"] = eq.p_tost < ALPHA
    return out, eq


def dose_response(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-1, 30, 60, 100, 150, 250, 10_000]
    labels = ["0-30", "31-60", "61-100", "101-150", "151-250", "250+"]
    d = df.assign(activity_bin=pd.cut(df.physical_activity_minutes_per_week, bins, labels=labels))
    g = d.groupby("activity_bin", observed=True).hba1c.agg(["count", "mean", "std"])
    g["ci95"] = 1.96 * g["std"] / np.sqrt(g["count"])
    rho, p = stats.spearmanr(df.physical_activity_minutes_per_week, df.hba1c)
    g.attrs.update(spearman_rho=rho, spearman_p=p)
    return g.reset_index()


def heterogeneity(df: pd.DataFrame) -> pd.DataFrame:
    """Exploratory: does the activity-HbA1c slope differ by income or age group? (interaction F-tests)"""
    d = df.assign(act100=df.physical_activity_minutes_per_week / 100,
                  age_group=pd.cut(df.age, [17, 39, 59, 120], labels=["18-39", "40-59", "60+"]),
                  income=df.income_level.astype(str))
    out = []
    for var in ["income", "age_group"]:
        base = smf.ols("hba1c ~ act100 + C(%s)" % var, d).fit()
        full = smf.ols("hba1c ~ act100 * C(%s)" % var, d).fit()
        f, p, _ = full.compare_f_test(base)
        for lvl, sub in d.groupby(var, observed=True):
            m = smf.ols("hba1c ~ act100", sub).fit()
            out.append(dict(moderator=var, level=str(lvl), n=len(sub), slope_per_100min=m.params["act100"],
                            ci_low=m.conf_int().loc["act100", 0], ci_high=m.conf_int().loc["act100", 1],
                            interaction_F=f, interaction_p=p))
    return pd.DataFrame(out)


def adjusted_logit(df: pd.DataFrame) -> pd.DataFrame:
    """Logistic regression for diagnosis; continuous predictors standardised (OR per 1 SD).

    Glycaemic markers (HbA1c, glucose, insulin) are deliberately excluded: they are the diagnostic
    criteria, so including them would make the model circular.
    """
    d = df.copy()
    cont = ["age", "bmi", "waist_to_hip_ratio", "physical_activity_minutes_per_week", "diet_score",
            "sleep_hours_per_day", "screen_time_hours_per_day", "alcohol_consumption_per_week",
            "systolic_bp", "ldl_cholesterol", "triglycerides"]
    for c in cont:
        d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std()
    binary = ["family_history_diabetes", "hypertension_history", "cardiovascular_history"]
    d["income"] = d.income_level.astype(str)
    d["education"] = d.education_level.astype(str)
    f = ("diagnosed_diabetes ~ " + " + ".join(c + "_z" for c in cont) + " + " + " + ".join(binary) +
         " + C(gender) + C(income, Treatment('Middle')) + C(education, Treatment('Highschool'))"
         " + C(smoking_status, Treatment('Never')) + C(employment_status, Treatment('Employed'))"
         " + C(ethnicity, Treatment('White'))")
    m = smf.logit(f, d).fit(disp=False, cov_type="HC1")
    ci = m.conf_int()
    res = pd.DataFrame({"term": m.params.index, "odds_ratio": np.exp(m.params.values),
                        "or_low": np.exp(ci[0].values), "or_high": np.exp(ci[1].values), "p": m.pvalues.values})
    res = res[res.term != "Intercept"].reset_index(drop=True)
    res.attrs["pseudo_r2"] = m.prsquared
    res.attrs["n"] = int(m.nobs)
    return res


def predictive_ladder(df: pd.DataFrame) -> pd.DataFrame:
    """5-fold CV AUC for diagnosed_diabetes as predictor sets are added."""
    sets = {
        "1. Demographics + SES": (["age"], ["gender", "ethnicity", "education_level", "income_level", "employment_status"]),
        "2. + Behaviour": (["age", "physical_activity_minutes_per_week", "diet_score", "sleep_hours_per_day",
                           "screen_time_hours_per_day", "alcohol_consumption_per_week"],
                          ["gender", "ethnicity", "education_level", "income_level", "employment_status", "smoking_status"]),
        "3. + Body & clinical": (["age", "physical_activity_minutes_per_week", "diet_score", "sleep_hours_per_day",
                                 "screen_time_hours_per_day", "alcohol_consumption_per_week", "bmi", "waist_to_hip_ratio",
                                 "systolic_bp", "diastolic_bp", "ldl_cholesterol", "hdl_cholesterol", "triglycerides",
                                 "family_history_diabetes", "hypertension_history", "cardiovascular_history"],
                                ["gender", "ethnicity", "education_level", "income_level", "employment_status", "smoking_status"]),
    }
    glyc = ["hba1c", "glucose_fasting", "glucose_postprandial", "insulin_level"]
    sets["4. + Glycaemic markers (circular)"] = (sets["3. + Body & clinical"][0] + glyc, sets["3. + Body & clinical"][1])
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []
    for name, (num, cat) in sets.items():
        pre = ColumnTransformer([("n", StandardScaler(), num), ("c", OneHotEncoder(drop="first"), cat)])
        pipe = make_pipeline(pre, LogisticRegression(max_iter=1000))
        auc = cross_val_score(pipe, df, df.diagnosed_diabetes, cv=cv, scoring="roc_auc")
        rows.append(dict(model=name, auc_mean=auc.mean(), auc_sd=auc.std()))
    return pd.DataFrame(rows)


def data_quality(df: pd.DataFrame) -> dict:
    """Quick integrity checks reported in the README."""
    stage = pd.crosstab(df.diabetes_stage, df.diagnosed_diabetes)
    t2 = df[df.diabetes_stage == "Type 2"].hba1c
    return dict(
        n=len(df), missing_cells=int(df.isna().sum().sum()), duplicate_rows=int(df.duplicated().sum()),
        prevalence_diagnosed=float(df.diagnosed_diabetes.mean()),
        stage_counts=df.diabetes_stage.value_counts().to_dict(),
        stage_vs_diagnosed=stage,
        share_hba1c_ge_6_5_among_t2=float((t2 >= 6.5).mean()),
        share_hba1c_ge_6_5_overall=float((df.hba1c >= 6.5).mean()),
    )
