"""
Core logic for the Job Offer Acceptance app.

Two layers work together:

1. ML layer  - the same pipeline as the Colab notebook (cleaning -> imputation/encoding ->
               Logistic Regression / Random Forest / Gradient Boosting). It only knows the
               fields that exist in the dataset (age, education, experience, salaries, ...).

2. Rule layer - transparent, adjustable scoring for real-world factors that are NOT in the
               dataset (offered salary, location / relocation, work mode, notice period, ...).

The final probability is a weighted blend of the two, plus caps for deal-breakers (very low offer, pay cut, impossible relocation).
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "offer_model.joblib"

# Same exclusions as the notebook: ID, target, and fields only known after the decision.
NUMERIC_FEATURES = ["Age", "Years of Experience", "Current Salary", "Expected Salary", "Companies Changed"]
CATEGORICAL_FEATURES = ["Gender", "Education", "Skills", "Job Sector", "Has Another Offer"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Offer Accepted"

# Verdict thresholds (probability of accepting)
LIKELY_ACCEPT = 0.65
LIKELY_DECLINE = 0.45


# ----------------------------------------------------------------------------------------
# Data cleaning (mirrors notebook section 5)
# ----------------------------------------------------------------------------------------
def _canonicalize(series: pd.Series) -> pd.Series:
    """Trim/collapse whitespace and unify capitalisation variants ('it', 'IT ', 'It' -> most common form)."""
    s = series.astype("object").map(lambda v: re.sub(r"\s+", " ", v).strip() if isinstance(v, str) else v)
    s = s.replace("", np.nan)
    keys = s.map(lambda v: v.lower() if isinstance(v, str) else v)
    canonical = {}
    for key, group in pd.DataFrame({"k": keys, "v": s}).dropna().groupby("k")["v"]:
        canonical[key] = group.value_counts().idxmax()
    return keys.map(lambda k: canonical.get(k, np.nan) if isinstance(k, str) else np.nan)


def clean_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s): {missing}. Found: {df.columns.tolist()}"
        )

    df = df.drop_duplicates().copy()

    target = (
        df[TARGET].astype("string").str.strip().str.lower()
        .replace({"accepted": "yes", "accept": "yes", "true": "yes", "1": "yes",
                  "rejected": "no", "declined": "no", "false": "no", "0": "no"})
    )
    df[TARGET] = target.map({"yes": "Yes", "no": "No"})

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan  # negative age/salary/experience is invalid

    for col in CATEGORICAL_FEATURES:
        df[col] = _canonicalize(df[col])

    # Normalise Yes/No style field
    df["Has Another Offer"] = (
        df["Has Another Offer"].astype("string").str.lower()
        .map({"yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes", "no": "No", "n": "No", "false": "No", "0": "No"})
    )

    if "Candidate ID" in df.columns:
        df = df[df["Candidate ID"].notna()].copy()
        df["Candidate ID"] = df["Candidate ID"].astype(str).str.strip()
        df = df.drop_duplicates(subset=["Candidate ID"], keep="first").copy()

    return df


# ----------------------------------------------------------------------------------------
# Training (mirrors notebook sections 12-13)
# ----------------------------------------------------------------------------------------
def _build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def _candidate_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([
            ("preprocessing", _build_preprocessor()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessing", _build_preprocessor()),
            ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                             class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessing", _build_preprocessor()),
            ("model", GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2,
                                                 min_samples_leaf=5, random_state=RANDOM_STATE)),
        ]),
    }


def train_and_save(df_raw: pd.DataFrame, path: Path = MODEL_PATH) -> dict[str, Any]:
    """Clean -> split -> train 3 models -> keep the best by ROC-AUC -> save a bundle."""
    df = clean_dataset(df_raw)
    df = df[df[TARGET].isin(["Yes", "No"])].copy()
    if df[TARGET].nunique() != 2 or df[TARGET].value_counts().min() < 5:
        raise ValueError("Need at least 5 'Yes' and 5 'No' labelled rows to train.")

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    y_test_bin = (y_test == "Yes").astype(int)
    majority_acc = float(max(y_test_bin.mean(), 1 - y_test_bin.mean()))

    rows, fitted = [], {}
    for name, pipe in _candidate_models().items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, list(pipe.classes_).index("Yes")]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Balanced Accuracy": balanced_accuracy_score(y_test, pred),
            "F1 (Yes)": f1_score(y_test, pred, pos_label="Yes", zero_division=0),
            "ROC-AUC": roc_auc_score(y_test_bin, proba),
        })
        fitted[name] = pipe

    # Choose by ROC-AUC (not F1/recall of 'Yes', which rewards always predicting 'Yes').
    best = max(rows, key=lambda r: r["ROC-AUC"])["Model"]

    # Refit the winner on ALL labelled rows for the deployed model.
    final = _candidate_models()[best]
    final.fit(X, y)

    choices = {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL_FEATURES}
    ranges = {c: (float(df[c].min()), float(df[c].max())) for c in NUMERIC_FEATURES}

    bundle = {
        "pipeline": final,
        "model_name": best,
        "metrics": rows,
        "majority_baseline_accuracy": majority_acc,
        "choices": choices,
        "ranges": ranges,
        "n_rows": int(len(df)),
        "yes_rate": float((y == "Yes").mean()),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return bundle


def load_bundle(path: Path = MODEL_PATH) -> dict[str, Any] | None:
    if not Path(path).exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def best_metrics(bundle: dict[str, Any]) -> dict[str, float]:
    return next(r for r in bundle["metrics"] if r["Model"] == bundle["model_name"])


def default_ml_weight(bundle: dict[str, Any] | None) -> float:
    """Weak models get little say: AUC 0.5 -> 0%, AUC 0.8+ -> 40% (rounded to 5%)."""
    if not bundle:
        return 0.0
    auc = best_metrics(bundle)["ROC-AUC"]
    w = 0.40 * min(max((auc - 0.5) / 0.3, 0.0), 1.0)
    return round(w / 0.05) * 0.05


def predict_ml(bundle: dict[str, Any], inputs: dict[str, Any]) -> float:
    row = pd.DataFrame([{c: inputs.get(c, np.nan) for c in FEATURES}])
    pipe = bundle["pipeline"]
    return float(pipe.predict_proba(row)[0, list(pipe.classes_).index("Yes")])


# ----------------------------------------------------------------------------------------
# Cities (rough lat/lon + cost-of-living tier). Tier is an approximation, edit freely.
# ----------------------------------------------------------------------------------------
REMOTE = "Remote / no fixed city"
OTHER = "Other city (enter details)"

CITIES: dict[str, tuple[float, float, int]] = {
    "Mumbai": (19.076, 72.878, 1), "Delhi NCR": (28.614, 77.209, 1), "Bengaluru": (12.972, 77.595, 1),
    "Hyderabad": (17.385, 78.487, 1), "Chennai": (13.083, 80.271, 1), "Pune": (18.520, 73.857, 1),
    "Kolkata": (22.573, 88.364, 1), "Gurugram": (28.459, 77.027, 1), "Noida": (28.535, 77.391, 1),
    "Ahmedabad": (23.023, 72.572, 1),
    "Chandigarh / Mohali": (30.733, 76.779, 2), "Jaipur": (26.912, 75.787, 2), "Lucknow": (26.847, 80.947, 2),
    "Indore": (22.720, 75.858, 2), "Kochi": (9.931, 76.267, 2), "Coimbatore": (11.017, 76.956, 2),
    "Nagpur": (21.146, 79.088, 2), "Bhubaneswar": (20.296, 85.825, 2), "Vadodara": (22.307, 73.181, 2),
    "Surat": (21.170, 72.831, 2), "Visakhapatnam": (17.687, 83.219, 2), "Thiruvananthapuram": (8.524, 76.936, 2),
    "Mysuru": (12.296, 76.639, 2), "Bhopal": (23.259, 77.413, 2), "Dehradun": (30.316, 78.032, 2),
    "Ludhiana": (30.901, 75.857, 3), "Amritsar": (31.634, 74.872, 3), "Jalandhar": (31.326, 75.576, 3),
    "Patna": (25.594, 85.138, 3), "Ranchi": (23.344, 85.310, 3), "Guwahati": (26.144, 91.736, 3),
    "Raipur": (21.251, 81.630, 3), "Kanpur": (26.449, 80.332, 3), "Varanasi": (25.317, 82.973, 3),
    "Jammu": (32.727, 74.857, 3), "Shimla": (31.105, 77.173, 3), "Nashik": (19.998, 73.790, 2),
    "Mangaluru": (12.914, 74.856, 3),
}
COST_INDEX = {1: 1.00, 2: 0.85, 3: 0.75}  # rough relative cost of living by tier


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


# ----------------------------------------------------------------------------------------
# Rule layer
# ----------------------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "Offer vs your expectation": 0.22,
    "Real pay rise (cost-of-living adjusted)": 0.14,
    "Location / relocation": 0.16,
    "Work mode fit": 0.08,
    "Career growth": 0.10,
    "Company brand": 0.05,
    "Benefits": 0.05,
    "Notice period vs joining date": 0.06,
    "Competing offer": 0.10,
    "Employment type": 0.04,
}

WORK_MODE_FIT = {  # (preferred, offered) -> score
    ("Remote", "Remote"): 100, ("Remote", "Hybrid"): 60, ("Remote", "On-site"): 20,
    ("Hybrid", "Remote"): 80, ("Hybrid", "Hybrid"): 100, ("Hybrid", "On-site"): 55,
    ("On-site", "Remote"): 55, ("On-site", "Hybrid"): 80, ("On-site", "On-site"): 100,
}
EMPLOYMENT_TYPE_SCORE = {"Full-time": 100, "Contract": 55, "Part-time": 45, "Internship": 35}
LEVEL_SCORE = {"Promotion / bigger role": 90, "Same level": 55, "Lower level / step back": 15}
BRAND_SCORE = {"Stronger / better known": 85, "Similar": 55, "Less known": 30}
BENEFITS_SCORE = {"Better than current": 85, "About the same": 55, "Worse than current": 25}
RELOCATE_BASE = {"Yes, happy to relocate": 80, "Only if the package is good": 50, "No, I cannot relocate": 5}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class Factor:
    name: str
    score: float          # 0-100, higher = more likely to accept
    weight: float
    detail: str
    applicable: bool = True


@dataclass
class Result:
    probability: float
    verdict: str
    ml_probability: float | None
    rule_probability: float
    ml_weight: float
    factors: list[Factor]
    caps: list[tuple[float, str]] = field(default_factory=list)  # (max probability, reason)


def resolve_distance(inp: dict[str, Any]) -> tuple[float | None, str]:
    """Distance in km between current city and job city, and whether a relocation is involved."""
    cur, new = inp["current_city"], inp["new_city"]
    if new == REMOTE:
        return 0.0, "remote"
    if cur in CITIES and new in CITIES:
        if cur == new:
            return 0.0, "same"
        return haversine_km(CITIES[cur][:2], CITIES[new][:2]), "calc"
    d = float(inp.get("manual_distance_km") or 0)
    if cur == new and cur != OTHER:
        return 0.0, "same"
    return d, "manual"


def _tier(city: str, manual_tier: int) -> int:
    return CITIES[city][2] if city in CITIES else int(manual_tier)


def score_factors(inp: dict[str, Any], weights: dict[str, float]) -> tuple[list[Factor], list[tuple[float, str]]]:
    f: list[Factor] = []
    breakers: list[tuple[float, str]] = []
    offered, expected, current = float(inp["offered_salary"]), float(inp["expected_salary"]), float(inp["current_salary"])
    mode = inp["offered_mode"]
    dist, kind = resolve_distance(inp)
    dist = dist or 0.0

    # 1. Offer vs expectation
    if expected > 0:
        ratio = offered / expected
        s = 100 * _sigmoid((ratio - 0.90) / 0.05)
        f.append(Factor("Offer vs your expectation", s, weights["Offer vs your expectation"],
                        f"Offer is {ratio:.0%} of your expected salary"))
        if ratio < 0.60:
            breakers.append((0.12, f"Offer is only {ratio:.0%} of the expected salary (below 60%)."))
        elif ratio < 0.75:
            breakers.append((0.35, f"Offer is only {ratio:.0%} of the expected salary (below 75%)."))
        elif ratio < 0.85:
            breakers.append((0.55, f"Offer is {ratio:.0%} of the expected salary (below 85%)."))
    else:
        f.append(Factor("Offer vs your expectation", 0, 0, "No expected salary entered", applicable=False))

    # 2. Real pay rise after cost of living
    cur_tier = _tier(inp["current_city"], inp.get("current_manual_tier", 2))
    new_tier = cur_tier if mode == "Remote" or inp["new_city"] == REMOTE else _tier(inp["new_city"], inp.get("new_manual_tier", 2))
    if current > 0:
        adj = offered * COST_INDEX[cur_tier] / COST_INDEX[new_tier]
        real_hike = adj / current - 1
        nominal = offered / current - 1
        if nominal < 0:
            breakers.append((0.40, f"Offer is a pay cut ({nominal:+.0%}) versus the current salary."))
        s = 100 * _sigmoid((real_hike - 0.15) / 0.08)
        f.append(Factor("Real pay rise (cost-of-living adjusted)", s,
                        weights["Real pay rise (cost-of-living adjusted)"],
                        f"Nominal rise {nominal:+.0%}; after cost of living ≈ {real_hike:+.0%}"))
    else:
        f.append(Factor("Real pay rise (cost-of-living adjusted)", 0, 0, "No current salary (fresher)", applicable=False))

    # 3. Location / relocation
    if mode == "Remote" or inp["new_city"] == REMOTE:
        loc_score, loc_detail = 100.0, "Remote role - no move needed"
    elif dist <= 25:
        loc_score, loc_detail = 95.0, "Job is in your current city"
    elif dist <= 60:
        loc_score, loc_detail = 80.0, f"About {dist:.0f} km away - commutable"
    else:
        will = inp["relocation_willingness"]
        loc_score = float(RELOCATE_BASE[will])
        if inp.get("relocation_assistance"):
            loc_score = min(100.0, loc_score + 15)
        if dist > 1000 and will != "No, I cannot relocate":
            loc_score -= 10
        loc_detail = f"About {dist:.0f} km away - relocation needed ({will.lower()})" + \
                     ("; relocation support offered" if inp.get("relocation_assistance") else "")
        if will == "No, I cannot relocate":
            breakers.append((0.12, "The job needs a relocation (>60 km) and the candidate cannot relocate."))
    f.append(Factor("Location / relocation", max(loc_score, 0), weights["Location / relocation"], loc_detail))

    # 4. Work mode
    pref = inp["preferred_mode"]
    s = 80.0 if pref == "No preference" else float(WORK_MODE_FIT[(pref, mode)])
    f.append(Factor("Work mode fit", s, weights["Work mode fit"], f"You prefer: {pref}; offer is: {mode}"))

    # 5-7. Qualitative comparisons
    f.append(Factor("Career growth", LEVEL_SCORE[inp["level_change"]], weights["Career growth"], inp["level_change"]))
    f.append(Factor("Company brand", BRAND_SCORE[inp["brand"]], weights["Company brand"], inp["brand"]))
    f.append(Factor("Benefits", BENEFITS_SCORE[inp["benefits"]], weights["Benefits"], inp["benefits"]))

    # 8. Notice period vs joining date
    notice, joining = int(inp["notice_days"]), int(inp["joining_days"])
    if notice <= joining:
        s, d = 100.0, f"Your notice ({notice} d) fits the joining window ({joining} d)"
    else:
        excess = notice - joining
        s, d = max(10.0, 100 - excess * 1.2), f"Notice is {excess} days longer than the joining window"
    f.append(Factor("Notice period vs joining date", s, weights["Notice period vs joining date"], d))

    # 9. Competing offer
    if inp["has_other_offer"] == "Yes":
        other = float(inp.get("other_offer_salary") or 0)
        if other > 0:
            s = min(85.0, 100 * _sigmoid((offered / other - 1.0) / 0.06))
            d = f"Other offer pays {other:,.0f}; this one is {offered / other:.0%} of it"
        else:
            s, d = 35.0, "You hold another offer (amount unknown)"
    else:
        s, d = 100.0, "No competing offer"
    f.append(Factor("Competing offer", s, weights["Competing offer"], d))

    # 10. Employment type
    f.append(Factor("Employment type", EMPLOYMENT_TYPE_SCORE[inp["employment_type"]],
                    weights["Employment type"], inp["employment_type"]))

    return f, breakers


def _rule_probability(factors: list[Factor]) -> float:
    active = [x for x in factors if x.applicable and x.weight > 0]
    total = sum(x.weight for x in active)
    if total <= 0:
        return 0.5
    return sum(x.score * x.weight for x in active) / total / 100.0


def verdict_for(p: float) -> str:
    if p >= LIKELY_ACCEPT:
        return "Likely to ACCEPT"
    if p < LIKELY_DECLINE:
        return "Likely to DECLINE"
    return "UNCERTAIN - could go either way"


def evaluate(
    inp: dict[str, Any],
    bundle: dict[str, Any] | None,
    ml_weight: float,
    weights: dict[str, float] | None = None,
) -> Result:
    weights = weights or DEFAULT_WEIGHTS
    factors, caps = score_factors(inp, weights)
    rule_p = _rule_probability(factors)

    ml_p = None
    if bundle is not None and ml_weight > 0:
        ml_p = predict_ml(bundle, {
            "Age": inp["age"], "Years of Experience": inp["experience"],
            "Current Salary": inp["current_salary"], "Expected Salary": inp["expected_salary"],
            "Companies Changed": inp["companies_changed"], "Gender": inp["gender"],
            "Education": inp["education"], "Skills": inp["skills"], "Job Sector": inp["sector"],
            "Has Another Offer": inp["has_other_offer"],
        })
        p = ml_weight * ml_p + (1 - ml_weight) * rule_p
    else:
        ml_weight, p = 0.0, rule_p

    # Caps: a weak salary offer / pay cut / impossible relocation cannot be averaged away by other factors
    if caps:
        p = min(p, min(c for c, _ in caps))

    return Result(p, verdict_for(p), ml_p, rule_p, ml_weight, factors, caps)


def minimum_offer_for_accept(inp: dict[str, Any], bundle, ml_weight: float, weights=None) -> float | None:
    """Smallest offered salary (searching up to 1.6x expectation) that reaches 'Likely to ACCEPT'."""
    base = float(inp["expected_salary"]) or float(inp["offered_salary"]) or 1.0
    start = float(inp["offered_salary"])
    steps = np.arange(start, base * 1.6 + 1, base * 0.01)
    for v in steps:
        trial = dict(inp, offered_salary=float(v))
        if evaluate(trial, bundle, ml_weight, weights).probability >= LIKELY_ACCEPT:
            return float(v)
    return None


# ----------------------------------------------------------------------------------------
# MongoDB (optional)  -  same database / collection names as the project report
# ----------------------------------------------------------------------------------------
DB_NAME = "job_offer_prediction"
CANDIDATES = "candidates"
PREDICTIONS = "predictions"


def get_mongo_uri(streamlit_secrets=None) -> str | None:
    if streamlit_secrets is not None:
        try:
            uri = streamlit_secrets.get("MONGODB_URI")
            if uri:
                return str(uri)
        except Exception:
            pass
    return os.environ.get("MONGODB_URI")


def _client(uri: str):
    from pymongo import MongoClient  # imported lazily so the app runs without pymongo installed
    c = MongoClient(uri, serverSelectionTimeoutMS=8000, connectTimeoutMS=8000)
    c.admin.command("ping")
    return c


def load_from_mongo(uri: str) -> pd.DataFrame:
    docs = list(_client(uri)[DB_NAME][CANDIDATES].find({}, {"_id": 0}))
    if not docs:
        raise ValueError(f"Collection {DB_NAME}.{CANDIDATES} is empty.")
    return pd.DataFrame(docs)


def save_prediction(uri: str, inp: dict[str, Any], res: Result) -> None:
    doc = {k: (v.item() if hasattr(v, "item") else v) for k, v in inp.items()}
    doc.update({
        "probability_accept": round(res.probability, 4),
        "verdict": res.verdict,
        "ml_probability": None if res.ml_probability is None else round(res.ml_probability, 4),
        "rule_probability": round(res.rule_probability, 4),
        "created_at": datetime.utcnow(),
    })
    _client(uri)[DB_NAME][PREDICTIONS].insert_one(doc)
