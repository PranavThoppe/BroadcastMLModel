from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

# -----------------------------
# Artifact paths
# -----------------------------
MODEL_PATH = "standard_nfl_broadcast_model.txt"
ENCODER_PATH = "onehot_encoder.pkl"
FEATURE_INFO_PATH = "feature_info.pkl"

# -----------------------------
# Load artifacts
# -----------------------------
booster = lgb.Booster(model_file=MODEL_PATH)
onehot = joblib.load(ENCODER_PATH)
feature_info: Dict[str, Any] = joblib.load(FEATURE_INFO_PATH)

# Columns exactly as saved during training
CAT_COLS = feature_info.get("cat_cols", ["network", "kickoff_bucket"])
NUM_COLS = feature_info.get(
    "num_cols",
    ["home_win_pct_pre","away_win_pct_pre","home_wins_pre","away_wins_pre","week",
     "total_team_wins","win_pct_diff","avg_win_pct","max_win_pct","min_win_pct",
     "strength_product"]
)
BIN_COLS = feature_info.get(
    "bin_cols",
    ["is_local_team","is_in_state_team","divisional_matchup",
     "competitive_game","high_quality_matchup","playoff_implications",
     "prime_matchup","market_available","both_teams_good","either_team_bad"]
)

DEFAULTS = {
    "network": "UNKNOWN",
    "kickoff_bucket": "UNKNOWN",
    "week": 10,  # safe mid-season default if week omitted
}

# -----------------------------
# API models
# -----------------------------
class Game(BaseModel):
    game_id: str

    # minimal numeric/binary payload you said you'd send
    is_local_team: Optional[int] = None
    is_in_state_team: Optional[int] = None
    divisional_matchup: Optional[int] = None
    home_win_pct_pre: Optional[float] = None
    away_win_pct_pre: Optional[float] = None
    home_wins_pre: Optional[float] = None
    away_wins_pre: Optional[float] = None

    # optional categoricals / week
    network: Optional[str] = None
    kickoff_bucket: Optional[str] = None
    week: Optional[int] = None

class PredictRequest(BaseModel):
    games: List[Game]

# -----------------------------
# Feature engineering (matches training)
# -----------------------------
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure booleans exist (fill missing with 0)
    for col in ["is_local_team", "is_in_state_team", "divisional_matchup"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Ensure numeric inputs exist
    for col in ["home_win_pct_pre", "away_win_pct_pre", "home_wins_pre", "away_wins_pre"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # week & cats with defaults
    if "week" not in df.columns:
        df["week"] = DEFAULTS["week"]
    df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(DEFAULTS["week"])

    if "network" not in df.columns:
        df["network"] = DEFAULTS["network"]
    df["network"] = df["network"].fillna(DEFAULTS["network"]).astype(str)

    if "kickoff_bucket" not in df.columns:
        df["kickoff_bucket"] = DEFAULTS["kickoff_bucket"]
    df["kickoff_bucket"] = df["kickoff_bucket"].fillna(DEFAULTS["kickoff_bucket"]).astype(str)

    # Derived numerics
    df["total_team_wins"] = df["home_wins_pre"] + df["away_wins_pre"]
    df["win_pct_diff"] = (df["home_win_pct_pre"] - df["away_win_pct_pre"]).abs()
    df["avg_win_pct"] = (df["home_win_pct_pre"] + df["away_win_pct_pre"]) / 2
    df["max_win_pct"] = df[["home_win_pct_pre", "away_win_pct_pre"]].max(axis=1)
    df["min_win_pct"] = df[["home_win_pct_pre", "away_win_pct_pre"]].min(axis=1)

    # Heuristics (bins)
    df["competitive_game"] = (df["win_pct_diff"] < 0.30).astype(int)
    df["high_quality_matchup"] = (df["avg_win_pct"] > 0.60).astype(int)
    df["playoff_implications"] = ((df["week"] >= 15) & (df["avg_win_pct"] > 0.40)).astype(int)
    df["prime_matchup"] = ((df["kickoff_bucket"] == "LATE") & (df["avg_win_pct"] > 0.50)).astype(int)

    # market_available
    if "market_code" in df.columns:
        df["market_available"] = (~df["market_code"].isna()).astype(int)
    else:
        df["market_available"] = 0

    # Team strength interactions
    df["strength_product"] = df["home_win_pct_pre"] * df["away_win_pct_pre"]
    df["both_teams_good"] = ((df["home_win_pct_pre"] > 0.5) & (df["away_win_pct_pre"] > 0.5)).astype(int)
    df["either_team_bad"] = ((df["home_win_pct_pre"] < 0.3) | (df["away_win_pct_pre"] < 0.3)).astype(int)

    return df

def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    exps = np.exp(x)
    s = exps.sum()
    return exps / s if s > 0 else np.full_like(exps, 1.0 / len(exps))

def _encode_and_stack(df: pd.DataFrame) -> np.ndarray:
    # OHE for categoricals
    enc = onehot.transform(df[CAT_COLS])
    if hasattr(enc, "toarray"):
        enc = enc.toarray()

    # Stack numeric + binary in the SAME order used in training
    X_num = df[NUM_COLS].astype(float).values
    X_bin = df[BIN_COLS].astype(float).values
    return np.hstack([X_num, X_bin, enc])

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="NFL Broadcast Predictor", version="1.2.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict_top")
def predict_top(req: PredictRequest):
    if not req.games:
        raise HTTPException(status_code=400, detail="No games provided")

    # Raw -> DataFrame
    df = pd.DataFrame([g.dict() for g in req.games])

    # Build features (exactly like training)
    feats = create_features(df)

    # Validate essentials
    essentials = ["home_win_pct_pre", "away_win_pct_pre", "home_wins_pre", "away_wins_pre"]
    mask_bad = feats[essentials].isna().any(axis=1)
    skipped = []
    if mask_bad.any():
        for _, r in feats.loc[mask_bad, ["game_id"] + essentials].iterrows():
            skipped.append({"game_id": r["game_id"], "reason": "missing essential numeric fields"})
        feats = feats.loc[~mask_bad].copy()

    if len(feats) == 0:
        raise HTTPException(status_code=400, detail={"message": "All candidates invalid", "skipped": skipped})

    # Ensure all expected cols exist (if feature_info had extra cols)
    for c in CAT_COLS:
        if c not in feats.columns:
            feats[c] = DEFAULTS.get(c, "UNKNOWN")
    for c in NUM_COLS:
        if c not in feats.columns:
            feats[c] = 0.0
    for c in BIN_COLS:
        if c not in feats.columns:
            feats[c] = 0

    # Build matrix in training order and predict
    X = _encode_and_stack(feats)
    scores = booster.predict(X, num_iteration=booster.best_iteration)
    scores = np.asarray(scores, dtype=float)
    probs = _softmax(scores)

    out = feats[["game_id"]].copy()
    out["score"] = scores
    out["probability"] = probs
    out = out.sort_values("probability", ascending=False)

    top = out.iloc[0]
    return {
        "top_game_id": str(top["game_id"]),
        "top_probability": float(top["probability"]),
        "probabilities": out.to_dict(orient="records"),
        "skipped": skipped
    }
