# --- add flag in request model ---
class PredictRequest(BaseModel):
    games: List[Game]
    enforce_local_priority: bool = True  # NEW: default ON for guaranteed local pick

@app.post("/predict_top")
def predict_top(req: PredictRequest):
    if not req.games:
        raise HTTPException(status_code=400, detail="No games provided")

    df = pd.DataFrame([g.dict() for g in req.games])
    feats = create_features(df)

    essentials = ["home_win_pct_pre", "away_win_pct_pre", "home_wins_pre", "away_wins_pre"]
    mask_bad = feats[essentials].isna().any(axis=1)
    skipped = []
    if mask_bad.any():
        skipped += [{"game_id": r["game_id"], "reason": "missing essential numeric fields"}
                    for _, r in feats.loc[mask_bad, ["game_id"] + essentials].iterrows()]
        feats = feats.loc[~mask_bad].copy()

    if len(feats) == 0:
        raise HTTPException(status_code=400, detail={"message": "All candidates invalid", "skipped": skipped})

    # Ensure columns exist
    for c in CAT_COLS:
        if c not in feats.columns:
            feats[c] = DEFAULTS.get(c, "UNKNOWN")
    for c in NUM_COLS:
        if c not in feats.columns:
            feats[c] = 0.0
    for c in BIN_COLS:
        if c not in feats.columns:
            feats[c] = 0

    # OPTIONAL: enforce local priority (exactly like your training eval)
    subset = feats
    if req.enforce_local_priority and "is_local_team" in feats.columns:
        local_mask = feats["is_local_team"].astype(int) == 1
        if local_mask.any():
            subset = feats.loc[local_mask].copy()

    # Predict on the chosen subset (either locals only or all)
    X = _encode_and_stack(subset)
    scores = booster.predict(X, num_iteration=booster.best_iteration)
    scores = np.asarray(scores, dtype=float)
    probs = _softmax(scores)

    ranked_subset = subset[["game_id"]].copy()
    ranked_subset["score"] = scores
    ranked_subset["probability"] = probs
    ranked_subset = ranked_subset.sort_values("probability", ascending=False)

    # rebuild full list probs: local subset gets its softmax; non-selected rows get 0 prob (for transparency)
    out = feats[["game_id"]].copy()
    out["score"] = 0.0
    out["probability"] = 0.0
    out = out.merge(ranked_subset, on="game_id", how="left", suffixes=("_all", ""))
    out["score"] = out["score"].fillna(out["score_all"]).fillna(0.0)
    out["probability"] = out["probability"].fillna(0.0)
    out = out.drop(columns=["score_all"]).sort_values("probability", ascending=False)

    top = out.iloc[0]
    return {
        "top_game_id": str(top["game_id"]),
        "top_probability": float(top["probability"]),
        "probabilities": out.to_dict(orient="records"),
        "skipped": skipped,
        "local_enforced": req.enforce_local_priority
    }
