from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from features.point_in_time import FEATURE_COLS
from ranking.features import RANKING_COLS


def _feature_cols(frame: pd.DataFrame, cols: list[str] | None) -> list[str]:
    if cols is not None:
        return list(cols)
    if "retrieval_score" in frame.columns:
        return list(RANKING_COLS)
    return list(FEATURE_COLS)


def train_binary(
    frame: pd.DataFrame,
    params: dict[str, Any] | None = None,
    feature_cols: list[str] | None = None,
) -> lgb.LGBMClassifier:
    cfg = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        cfg.update(params)
    if "verbose" in cfg:
        cfg["verbosity"] = cfg.pop("verbose")
    cols = _feature_cols(frame, feature_cols)
    model = lgb.LGBMClassifier(**cfg)
    model.fit(frame[cols], frame["label"].astype(int))
    return model


def predict_proba(
    model: lgb.LGBMClassifier,
    frame: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    cols = _feature_cols(frame, feature_cols)
    return model.predict_proba(frame[cols])[:, 1]


def classification_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
    }


def ndcg_at_k_grouped(frame: pd.DataFrame, scores: np.ndarray, k: int = 10) -> float:
    """NDCG@k per user, then mean. Candidate set is whatever rows are in `frame`."""
    tmp = frame[["user_id", "label"]].copy()
    tmp["score"] = scores
    vals = []
    log2 = np.log2
    for _, grp in tmp.groupby("user_id", sort=False):
        if grp["label"].sum() <= 0:
            continue
        order = np.argsort(-grp["score"].to_numpy())
        labels = grp["label"].to_numpy()[order][:k]
        dcg = 0.0
        for i, rel in enumerate(labels):
            if rel:
                dcg += 1.0 / log2(i + 2)
        n_rel = int(min(k, grp["label"].sum()))
        idcg = sum(1.0 / log2(i + 2) for i in range(n_rel))
        if idcg > 0:
            vals.append(dcg / idcg)
    return float(np.mean(vals)) if vals else float("nan")


def save_model(model: lgb.LGBMClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(path))


def load_booster(path: Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))


def predict_booster(
    booster: lgb.Booster,
    frame: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    cols = _feature_cols(frame, feature_cols)
    return np.asarray(booster.predict(frame[cols]), dtype=np.float64)


def feature_importance(model: lgb.LGBMClassifier) -> pd.DataFrame:
    names = list(model.booster_.feature_name())
    gain = model.booster_.feature_importance(importance_type="gain")
    return (
        pd.DataFrame({"feature": names, "gain": gain})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
