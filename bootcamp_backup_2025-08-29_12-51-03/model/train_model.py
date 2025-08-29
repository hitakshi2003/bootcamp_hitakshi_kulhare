# 1) Load raw categorical dataset
df_raw = pd.read_csv("data/german.data", sep=r"\s+", header=None, engine="python")

# 2) Official column names (20 features + target)
cols = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment_since", "installment_rate_pct", "personal_status_sex",
    "other_debtors", "residence_since", "property_magnitude", "age",
    "other_installment_plans", "housing", "number_existing_credits", "job",
    "people_liable_maintenance", "telephone", "foreign_worker", "target"
]
df_raw.columns = cols

# 3) Target to binary: 1=good -> 0, 2=bad -> 1  (bad = default)
df_raw["target"] = df_raw["target"].map({1: 0, 2: 1}).astype(int)

# 4) Optional: map A-codes to readable labels (you can skip this; one-hot on codes also works)
map_dicts = {
    "checking_status": {"A11":"<0", "A12":"0<= <200", "A13":">=200", "A14":"no_checking"},
    "credit_history": {"A30":"no_credits/all_paid", "A31":"all_paid",
                       "A32":"existing_paid", "A33":"delayed",
                       "A34":"critical/other"},
    "purpose": {"A40":"car_new","A41":"car_used","A42":"furniture","A43":"radio_tv","A44":"appliances",
                "A45":"repairs","A46":"education","A47":"vacation","A48":"retraining",
                "A49":"business","A410":"other"},
    "savings_status": {"A61":"<100","A62":"100-500","A63":"500-1000","A64":">=1000","A65":"unknown"},
    "employment_since": {"A71":"unemployed","A72":"<1","A73":"1-4","A74":"4-7","A75":">=7"},
    "personal_status_sex": {"A91":"male_single","A92":"male_div_sep","A93":"male_mar_wid","A94":"female_any"},
    "other_debtors": {"A101":"none","A102":"co-applicant","A103":"guarantor"},
    "property_magnitude": {"A121":"real_estate","A122":"life_insurance","A123":"car_other","A124":"none/unknown"},
    "other_installment_plans": {"A141":"bank","A142":"stores","A143":"none"},
    "housing": {"A151":"rent","A152":"own","A153":"free"},
    "job": {"A171":"unemployed_unskilled_nonres","A172":"unskilled_resident",
            "A173":"skilled","A174":"management/self_emp/high_qual"},
    "telephone": {"A191":"none","A192":"yes_registered"},
    "foreign_worker": {"A201":"yes","A202":"no"},
}

for c, m in map_dicts.items():
    if c in df_raw.columns:
        df_raw[c] = df_raw[c].map(m).astype("category")

# 5) Ensure 'age' is numeric years
df_raw["age"] = pd.to_numeric(df_raw["age"], errors="coerce")

print("Shape:", df_raw.shape)
print(df_raw[["age","duration","credit_amount","target"]].describe())
print("\nAge unique sample:", sorted(df_raw["age"].dropna().unique())[:10], " ...")
print("\nCategorical example values:")
print(df_raw[["checking_status","purpose","savings_status"]].head())

# project/src/train_model.py
from pathlib import Path
import os, json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib
from dotenv import load_dotenv

# Load env vars
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent  # go up from src/ to project root
DATA_PATH = Path(os.getenv("DATA_PATH", BASE_DIR / "data" / "raw" / "german.data-numeric"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "model"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODEL_DIR / "model.joblib"))
FEATURES_PATH = Path(os.getenv("FEATURES_PATH", MODEL_DIR / "feature_names.json"))

MODEL_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"DATA_PATH not found: {DATA_PATH}\\nSet DATA_PATH in .env or put the file there.")

# Load dataset (25 columns: 24 features + target)
df = pd.read_csv(DATA_PATH, sep=r"\\s+", header=None, engine="python")
if df.shape[1] != 25:
    raise ValueError(f"Expected 25 columns, got {df.shape[1]}. Check dataset format.")

# Define column names (update if you used different names in SME_creditrisk notebook)
df.columns = [
    "checking_status", "duration_months", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment_since", "installment_rate_pct", "personal_status_sex", "other_debtors",
    "residence_since", "property_magnitude", "age_years", "other_installment_plans", "housing",
    "number_credits", "job", "people_liable", "telephone", "foreign_worker",
    "existing_credit", "dependents", "own_telephone", "foreign_worker_flag", "target"
]

X = df.drop(columns=["target"])
y = df["target"]

# Convert target from {1,2} → {0,1}
if set(np.unique(y)) == {1, 2}:
    y = (y == 2).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression pipeline
pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),  # safe for numeric + encoded categorical
    ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
])

pipe.fit(X_train, y_train)

# Evaluate
proba = pipe.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)
print(f"Validation AUC: {auc:.4f}")

# Save model + features
joblib.dump(pipe, MODEL_PATH)
FEATURES_PATH.write_text(json.dumps(list(X.columns), indent=2))
print(f"Saved model → {MODEL_PATH}")
print(f"Saved feature names → {FEATURES_PATH}")