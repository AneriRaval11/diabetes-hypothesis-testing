"""Run the full pipeline: python src/run_analysis.py  ->  results/tables/*.csv and results/figures/*.png"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from analysis_utils import load_data  # noqa: E402
import hypotheses as H  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG, TAB = ROOT / "results" / "figures", ROOT / "results" / "tables"
sns.set_theme(style="whitegrid", context="notebook")


def fig_dose_response(g):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.errorbar(g.activity_bin.astype(str), g["mean"], yerr=g.ci95, fmt="o-", capsize=4, color="#8a1538")
    ax.set_xlabel("Physical activity (minutes / week)")
    ax.set_ylabel("Mean HbA1c (%)")
    ax.set_title("HbA1c falls steadily with weekly physical activity")
    fig.tight_layout(); fig.savefig(FIG / "01_activity_dose_response.png", dpi=160); plt.close(fig)


def fig_forest(main):
    cont = main[main.effect_measure == "Cohen's d"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    y = np.arange(len(cont))[::-1]
    ax.errorbar(cont.effect_size, y, xerr=[cont.effect_size - cont.es_low, cont.es_high - cont.effect_size],
                fmt="s", color="#8a1538", capsize=4)
    ax.axvline(0, color="grey", lw=1)
    for x in (-0.2, 0.2):
        ax.axvline(x, color="grey", lw=0.6, ls=":")
    short = {"H1": "H1 Activity >=150 min/wk", "H2": "H2 Diet score Q4 vs Q1", "H4": "H4 Obesity (BMI>=30)", "H7": "H7 Current vs never smoker"}
    ax.set_yticks(y); ax.set_yticklabels([short[i] for i in cont.id], fontsize=9)
    ax.set_xlabel("Cohen's d on HbA1c (95% CI)\ndotted lines = |d| of 0.2 (small)")
    ax.set_title("Continuous outcome (HbA1c)")
    rd = main[main.id.isin(["H3", "H5", "H6"])]
    ax = axes[1]
    y = np.arange(len(rd))[::-1]
    ax.errorbar(rd.estimate * 100, y, xerr=[(rd.estimate - rd.ci_low) * 100, (rd.ci_high - rd.estimate) * 100],
                fmt="s", color="#2f5d8a", capsize=4)
    ax.axvline(0, color="grey", lw=1)
    short2 = {"H3": "H3 Family history vs none", "H5": "H5 Low vs High income", "H6": "H6 No formal vs Postgrad"}
    ax.set_yticks(y); ax.set_yticklabels([short2[i] for i in rd.id], fontsize=9)
    ax.set_xlabel("Diagnosis-rate gap, pp (95% CI)")
    ax.set_title("Binary outcome (diagnosis)")
    fig.tight_layout(); fig.savefig(FIG / "02_effect_sizes_forest.png", dpi=160); plt.close(fig)


def fig_ses(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, col, title in zip(axes, ["income_level", "education_level"], ["Income level", "Education level"]):
        g = df.groupby(col, observed=True).diagnosed_diabetes.agg(["mean", "count"])
        ci = 1.96 * np.sqrt(g["mean"] * (1 - g["mean"]) / g["count"])
        ax.bar(g.index.astype(str), g["mean"] * 100, yerr=ci * 100, capsize=4, color="#8a1538", alpha=0.85)
        ax.set_title(f"Diagnosis rate by {title.lower()}"); ax.set_xlabel(title)
        ax.tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Diagnosed with diabetes (%)"); axes[0].set_ylim(0, 80)
    fig.tight_layout(); fig.savefig(FIG / "03_ses_gradient_is_flat.png", dpi=160); plt.close(fig)


def fig_logit(res):
    r = res[~res.term.str.contains("C\\(")].copy()
    r["label"] = r.term.str.replace("_z", " (per 1 SD)").str.replace("_", " ")
    r = r.sort_values("odds_ratio")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    y = np.arange(len(r))
    ax.errorbar(r.odds_ratio, y, xerr=[r.odds_ratio - r.or_low, r.or_high - r.odds_ratio], fmt="o", color="#8a1538", capsize=3)
    ax.axvline(1, color="grey", lw=1); ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(r.label, fontsize=8)
    ax.set_xlabel("Adjusted odds ratio (log scale, 95% CI)")
    ax.set_title("Adjusted associations with diagnosed diabetes")
    fig.tight_layout(); fig.savefig(FIG / "04_adjusted_odds_ratios.png", dpi=160); plt.close(fig)


def fig_auc(lad):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.barh(lad.model[::-1], lad.auc_mean[::-1], xerr=lad.auc_sd[::-1], color="#2f5d8a")
    ax.axvline(0.5, color="grey", lw=1, ls=":")
    ax.set_xlim(0.45, 1.0); ax.set_xlabel("5-fold cross-validated AUC")
    for i, v in enumerate(lad.auc_mean[::-1]):
        ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_title("Predicting diagnosis: what each predictor block adds")
    fig.tight_layout(); fig.savefig(FIG / "05_auc_ladder.png", dpi=160); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    df = load_data(ROOT / "data" / "raw" / "diabetes_dataset.csv")
    main_t, eq = H.confirmatory_tests(df)
    main_t.to_csv(TAB / "confirmatory_results.csv", index=False)
    eq.to_csv(TAB / "equivalence_tests.csv", index=False)
    dr = H.dose_response(df); dr.to_csv(TAB / "activity_dose_response.csv", index=False)
    het = H.heterogeneity(df); het.to_csv(TAB / "heterogeneity_activity_slopes.csv", index=False)
    lg = H.adjusted_logit(df); lg.to_csv(TAB / "adjusted_logit_odds_ratios.csv", index=False)
    lad = H.predictive_ladder(df); lad.to_csv(TAB / "auc_ladder.csv", index=False)
    fig_dose_response(dr); fig_forest(main_t); fig_ses(df); fig_logit(lg); fig_auc(lad)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(main_t.drop(columns=["hypothesis", "outcome"]).round(4)); print(eq.round(4)); print(dr.round(3))
    print(dr.attrs); print(het.round(4)); print(lg.round(3)); print(lg.attrs); print(lad.round(4))
    print(H.data_quality(df))


if __name__ == "__main__":
    main()
