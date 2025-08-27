from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

# -----------------------------
# Load artifacts
# -----------------------------
MODEL_PATH = "standard_nfl_broadcast_model.txt"
ENCODER_PATH = "onehot_encoder.pkl"
FEATURE_INFO_PATH = "feature_info.pkl"

booster = lgb.Booster(model_file=MODEL_PATH)
onehot = joblib.load(ENCODER_PATH)
feature_info = joblib.load(FEATURE_INFO_PATH)

# Defaults if feature_info is missing keys
CAT_COLS = feature_info.get("cat_cols", ["network", "kickoff_bucket"])
NUM_COLS = feature_info.get(
    "num_cols",
    ["week", "home_win_pct_pre", "away_win_pct_pre", "home_wins_pre", "away_wins_pre"],
)

DEFAULTS = {
    "network": "UNKNOWN",
    "kickoff_bucket": "UNKNOWN",
}

# -----------------------------
# API schema
# -----------------------------
class Game(BaseModel):
    game_id: str
    season: Optional[int] = None
    week: Optional[int] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None

    # minimal numeric/binary payload
    home_win_pct_pre: Optional[float] = None
    away_win_pct_pre: Optional[float] = None
    home_wins_pre: Optional[float] = None
    away_wins_pre: Optional[float] = None

    # categoricals are optional
    network: Optional[str] = None
    kickoff_bucket: Optional[str] = None

class PredictRequest(BaseModel):
    games: List[Game]

class PredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]

class PredictTopResponse(BaseModel):
    top_game_id: str
    top_probability: float
    probabilities: List[Dict[str, Any]]
    skipped: List[Dict[str, Any]] = []

# -----------------------------
# Utils
# -----------------------------
def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    exps = np.exp(x)
    denom = np.sum(exps)
    return exps / denom if denom > 0 else np.full_like(exps, 1.0 / len(exps))

def _prepare_dataframe(games: List[Game]):
    """Build DataFrame, fill defaults, validate, and return df + skipped list."""
    raw = [g.dict() for g in games]
    df = pd.DataFrame(raw)

    # Ensure columns exist
    for col in CAT_COLS:
        if col not in df.columns:
            df[col] = DEFAULTS.get(col, "UNKNOWN")
    for col in NUM_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # Fill categoricals defaults if missing
    for col in CAT_COLS:
        df[col] = df[col].fillna(DEFAULTS.get(col, "UNKNOWN")).astype(str)

    # Coerce numerics
    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Skip candidates with missing essential numerics
    essential = ["home_win_pct_pre", "away_win_pct_pre", "home_wins_pre", "away_wins_pre"]
    mask_bad = df[essential].isna().any(axis=1)
    skipped = []
    if mask_bad.any():
        for _, r in df.loc[mask_bad].iterrows():
            skipped.append({"game_id": r.get("game_id", "<unknown>"), "reason": "missing essential numeric fields"})
        df = df.loc[~mask_bad].copy()

    return df, skipped

def _encode_and_predict(df: pd.DataFrame):
    """One-hot encode categoricals, stack with numerics, predict scores."""
    # OneHotEncoder may return sparse matrix OR ndarray depending on version/params
    enc = onehot.transform(df[CAT_COLS])
    if hasattr(enc, "toarray"):
        enc = enc.toarray()

    X_num = df[NUM_COLS].astype(float).values
    X = np.hstack([X_num, enc])

    scores = booster.predict(X, num_iteration=booster.best_iteration)
    if isinstance(scores, list):
        scores = np.asarray(scores, dtype=float)

    probs = _softmax(scores)
    return scores, probs

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="NFL Broadcast Predictor", version="1.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

# Keeps your existing /predict but now returns softmax probabilities and handles dense/sparse
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.games:
        raise HTTPException(status_code=400, detail="No games provided")

    df, skipped = _prepare_dataframe(req.games)
    if len(df) == 0:
        raise HTTPException(status_code=400, detail={"message": "All candidates invalid", "skipped": skipped})

    scores, probs = _encode_and_predict(df)

    df_out = df.copy()
    df_out["score"] = scores
    df_out["probability"] = probs

    ranked = df_out.sort_values("probability", ascending=False).to_dict(orient="records")
    return {"predictions": ranked}

# New endpoint that returns the top pick + all probabilities + list of skipped
@app.post("/predict_top", response_model=PredictTopResponse)
def predict_top(req: PredictRequest):
    if not req.games:
        raise HTTPException(status_code=400, detail="No games provided")

    df, skipped = _prepare_dataframe(req.games)
    if len(df) == 0:
        raise HTTPException(status_code=400, detail={"message": "All candidates invalid", "skipped": skipped})

    scores, probs = _encode_and_predict(df)

    df_out = df.copy()
    df_out["score"] = scores
    df_out["probability"] = probs
    ranked = df_out.sort_values("probability", ascending=False)

    top = ranked.iloc[0]
    return {
        "top_game_id": str(top["game_id"]),
        "top_probability": float(top["probability"]),
        "probabilities": ranked.to_dict(orient="records"),
        "skipped": skipped
    }

# --------------- local dev ---------------
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
