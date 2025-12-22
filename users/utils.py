# users/utils.py
import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'jobrole_model.joblib')  # update path if needed

_model = None
def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_from_fields(degree, major, specialization, cgpa, skills, certification, experience, industry):
    """
    Build a single-row DataFrame with same column names your model expects,
    then call model.predict / predict_proba.
    Adjust column names/order to match your saved pipeline.
    """
    model = load_model()
    # normalize skills to a single string (if model pipeline expects text)
    skills_str = skills if isinstance(skills, str) else ','.join(skills)

    # Build DataFrame - update keys to match your training columns exactly
    row = {
        'degree': degree,
        'major': major,
        'specialization': specialization or '',
        'cgpa': float(cgpa) if cgpa is not None else 0.0,
        'skills': skills_str,
        'certification': certification or '',
        'experience': float(experience) if experience is not None else 0.0,
        'industry': industry or '',
    }
    X = pd.DataFrame([row])

    # If your model pipeline expects different preprocessing, adapt here.
    pred = model.predict(X)
    confidence = None
    try:
        proba = model.predict_proba(X)
        confidence = float(max(proba[0]))
    except Exception:
        confidence = None

    return str(pred[0]), confidence
