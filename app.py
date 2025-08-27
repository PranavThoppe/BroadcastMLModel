from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import joblib
import lightgbm as lgb
import pandas as pd

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
NUM_COLS = feature_info.get("num_cols", ["week","home_win_pct_pre","away_win_pct_pre","home_wins_pre","away_wins_pre"])

# -----------------------------
# API schema
# -----------------------------
class Game(BaseModel):
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    network: Optional[str] = None
    kickoff_bucket: Optional[str] = None
    home_win_pct_pre: Optional[float] = None
    away_win_pct_pre: Optional[float] = None
    home_wins_pre: Optional[int] = None
    away_wins_pre: Optional[int] = None

class PredictRequest(BaseModel):
    games: List[Game]

class PredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="NFL Broadcast Predictor")

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.games:
        raise HTTPException(status_code=400, detail="No games provided")

    df = pd.DataFrame([g.dict() for g in req.games])

    # Fill missing optional fields
    for col in NUM_COLS:
        if col not in df:
            df[col] = 0
    for col in CAT_COLS:
        if col not in df:
            df[col] = "UNKNOWN"

    # Encode categorical
    df_enc = onehot.transform(df[CAT_COLS])
    df_num = df[NUM_COLS]
    X = pd.concat([df_num.reset_index(drop=True),
                   pd.DataFrame(df_enc.toarray())], axis=1)

    # Predictions
    scores = booster.predict(X)
    df["probability"] = scores

    # Sort by probability
    ranked = df.sort_values("probability", ascending=False).to_dict(orient="records")

    return {"predictions": ranked}
