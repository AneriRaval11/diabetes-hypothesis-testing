# Data

`raw/diabetes_dataset.csv` — 100,000 rows x 31 columns, no missing values.

**Source and license:** [Diabetes Health Indicators Dataset on Kaggle](https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset), listed as **CC0: Public Domain** on the dataset page. The dataset describes its records as synthetic.

The data appear to be **synthetic or simulated** (60% diagnosed prevalence, no socioeconomic gradient). See the main README.

## Data dictionary

| Group | Variables |
|---|---|
| Demographics / SES | `age`, `gender`, `ethnicity`, `education_level`, `income_level`, `employment_status` |
| Behaviour | `smoking_status`, `alcohol_consumption_per_week`, `physical_activity_minutes_per_week`, `diet_score` (0-10), `sleep_hours_per_day`, `screen_time_hours_per_day` |
| History | `family_history_diabetes`, `hypertension_history`, `cardiovascular_history` (0/1) |
| Body / vitals | `bmi`, `waist_to_hip_ratio`, `systolic_bp`, `diastolic_bp`, `heart_rate` |
| Lipids | `cholesterol_total`, `hdl_cholesterol`, `ldl_cholesterol`, `triglycerides` |
| Glycaemic markers | `glucose_fasting`, `glucose_postprandial`, `insulin_level`, `hba1c` (%) |
| Outcomes | `diabetes_risk_score`, `diabetes_stage` (No Diabetes / Pre-Diabetes / Type 2 / Type 1 / Gestational), `diagnosed_diabetes` (0/1) |

`diabetes_risk_score` is not used as a predictor: its construction is undocumented and it may encode the outcome.
