# users/ml_helper.py
import os
import logging
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model_payload.pkl")
ALT = os.path.join(BASE_DIR, "users", "model_payload.pkl")
if os.path.exists(ALT) and not os.path.exists(MODEL_PATH):
    MODEL_PATH = ALT

_cached = None

def _load(path=None):
    p = path or MODEL_PATH
    if not os.path.exists(p):
        raise FileNotFoundError(f"Model payload not found at {p}")
    payload = joblib.load(p)
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError("Invalid model payload")
    return payload

def get_model():
    global _cached
    if _cached is None:
        _cached = _load()
    return _cached

def reload_model(path=None):
    global _cached
    _cached = _load(path)
    logger.info("Reloaded model payload from %s", path or MODEL_PATH)
    return _cached

def _coerce_input_df(sample_dict, feature_columns, cat_cols=None):
    df = pd.DataFrame([sample_dict])
    for col in feature_columns:
        if col not in df.columns:
            df[col] = None
    df = df[feature_columns].copy()
    if cat_cols is None:
        cat_cols = []
    for col in feature_columns:
        if col in cat_cols or col == "Skills":
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df

def _extract_classifier(pipeline):
    if pipeline is None:
        return None
    if hasattr(pipeline, "predict_proba") and hasattr(pipeline, "classes_"):
        return pipeline
    if hasattr(pipeline, "named_steps"):
        for name, step in reversed(list(pipeline.named_steps.items())):
            if hasattr(step, "predict_proba"):
                return step
    try:
        last = pipeline.steps[-1][1]
        if hasattr(last, "predict_proba"):
            return last
    except Exception:
        pass
    return None

def predict_proba(sample_dict):
    payload = get_model()
    model = payload["model"]
    le = payload.get("label_encoder")
    feature_cols = payload.get("feature_columns", [])
    cat_cols = payload.get("cat_cols", []) or []

    df = _coerce_input_df(sample_dict, feature_cols, cat_cols)

    try:
        probs = model.predict_proba(df)[0]
    except Exception as e:
        logger.debug("pipeline.predict_proba failed: %s", e)
        classifier = _extract_classifier(model)
        if classifier is None:
            raise RuntimeError("No classifier with predict_proba found")
        if hasattr(model, "named_steps") and "pre" in model.named_steps:
            try:
                X_trans = model.named_steps["pre"].transform(df)
            except Exception:
                X_trans = model.named_steps["pre"].transform(df.values)
        else:
            X_trans = df.values
        X_trans = np.nan_to_num(np.asarray(X_trans, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        probs = classifier.predict_proba(X_trans)[0]

    # determine classes
    try:
        if hasattr(model, "classes_"):
            classes = model.classes_
        else:
            classifier = _extract_classifier(model)
            classes = getattr(classifier, "classes_", None)
    except Exception:
        classes = None

    if classes is None or len(classes) != len(probs):
        return {str(i): float(p) for i, p in enumerate(probs)}

    # if label encoder present, classes may be integer indexes - try inverse transform
    if le is not None:
        try:
            # if classes are ints, map through label encoder
            classes_arr = np.array(classes)
            try:
                labels = le.inverse_transform(classes_arr.astype(int))
            except Exception:
                labels = [str(c) for c in classes_arr]
        except Exception:
            labels = [str(c) for c in classes]
    else:
        labels = [str(c) for c in classes]

    return {str(lbl): float(p) for lbl, p in zip(labels, probs)}

def predict_proba_percent(sample_dict):
    probs = predict_proba(sample_dict)
    return {lbl: round(float(p) * 100.0, 1) for lbl, p in probs.items()}

def predict_topk_percent(sample_dict, k=5):
    probs = predict_proba_percent(sample_dict)
    items = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(lbl, float(score)) for lbl, score in items]

def predict_from_dict(sample_dict):
    probs = predict_proba(sample_dict)
    if not probs:
        raise RuntimeError("No probabilities returned")
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_label, top_prob = sorted_items[0]
    return {
        "role": str(top_label),
        "confidence": round(float(top_prob) * 100.0, 1),
        "breakdown": {str(k): float(v) for k, v in probs.items()}
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("ml_helper loaded. MODEL_PATH:", MODEL_PATH)
