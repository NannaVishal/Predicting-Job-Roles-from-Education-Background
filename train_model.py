# train_model.py
import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

OUT_PATH = os.path.join("users", "model_payload.pkl")
DATA_CANDIDATES = ["users/JobRole.xlsx", "JobRole.xlsx", "users/JobRole.csv", "JobRole.csv"]
LABEL_COL = "Job Role"

def find_dataset():
    for p in DATA_CANDIDATES:
        if os.path.exists(p):
            return p
    return None

def load_dataset(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xls", ".xlsx"):
        return pd.read_excel(path, engine="openpyxl")
    return pd.read_csv(path)

def main():
    path = find_dataset()
    if not path:
        print("Dataset not found. Tried:", DATA_CANDIDATES)
        sys.exit(2)

    df = load_dataset(path)

    if LABEL_COL not in df.columns:
        print(f"ERROR: expected label column '{LABEL_COL}' not found. Columns:", list(df.columns))
        sys.exit(3)

    # Required features (use those present)
    feature_cols = [
        "Degree", "Major", "Specialization", "CGPA", "Skills",
        "Certification", "Years of Experience", "Preferred Industry"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()

    # numeric conversion
    if "CGPA" in X.columns:
        X["CGPA"] = pd.to_numeric(X["CGPA"], errors="coerce")
    if "Years of Experience" in X.columns:
        X["Years of Experience"] = pd.to_numeric(X["Years of Experience"], errors="coerce")

    y_raw = df[LABEL_COL].astype(str).str.strip().fillna("")
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # split numeric vs categorical; Skills handled separately
    numeric_cols = [c for c in X.columns if X[c].dtype.kind in "biufc" and c != "Skills"]
    cat_cols = [c for c in X.columns if c not in numeric_cols and c != "Skills"]

    transformers = []

    if numeric_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipe, numeric_cols))

    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    if "Skills" in X.columns:
        skills_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("squeeze", FunctionTransformer(np.ravel, validate=False)),
            ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1,2), sublinear_tf=True))
        ])
        transformers.append(("skills", skills_pipe, ["Skills"]))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)

    clf = Pipeline([
        ("pre", preprocessor),
        ("rf", RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1))
    ])

    stratify_arg = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42, stratify=stratify_arg)

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100.0
    print(f"Accurarcy: {acc:.2f}%")

    payload = {
        "model": clf,
        "label_encoder": le,
        "feature_columns": feature_cols,
        "cat_cols": cat_cols,
        "label_col": LABEL_COL,
    }

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    joblib.dump(payload, OUT_PATH)

if __name__ == "__main__":
    main()
