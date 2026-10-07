"""Leakage-aware feature and model evaluation for offline GP research."""
from __future__ import annotations

from tqdm.auto import tqdm
import numpy as np
import pandas as pd


def _matrix(X):
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float), list(X.columns)
    array = np.asarray(X, dtype=float)
    return array, [f"x{i}" for i in range(array.shape[1])]


def _impute_scale(train, test):
    med = np.nanmedian(np.where(np.isfinite(train), train, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    train = np.where(np.isfinite(train), train, med)
    test = np.where(np.isfinite(test), test, med)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def canonical_model_name(model_type):
    key = "".join(ch for ch in str(model_type).casefold() if ch.isalnum())
    aliases = {"ridge": "ridge", "l2": "ridge", "ols": "ols", "linear": "ols",
               "ordinaryleastsquares": "ols", "tree": "tree", "decisiontree": "tree",
               "neuralnetwork": "neural_network", "neural": "neural_network", "nn": "neural_network", "mlp": "neural_network"}
    if key not in aliases:
        raise ValueError(f"unknown predictive model {model_type!r}; choose Ridge, OLS, Tree, or Neural Network")
    return aliases[key]


def validate_evaluation_models(model_types):
    if not model_types:
        raise ValueError("evaluation_models must contain at least one model")
    return tuple(dict.fromkeys(canonical_model_name(name) for name in model_types))


def fit_predictive_model(model_type, X_train, y_train, X_test, model_params=None):
    """Fit one interchangeable model and return predictions; uses numpy only."""
    params = model_params or {}
    train, _ = _matrix(X_train)
    test, _ = _matrix(X_test)
    y = np.asarray(y_train, dtype=float).reshape(-1)
    valid = np.isfinite(y) & np.isfinite(train).all(axis=1)
    train, y = train[valid], y[valid]
    if not len(y):
        raise ValueError("no finite training samples")
    train, test = _impute_scale(train, test)
    center = y.mean()
    yc = y - center
    name = canonical_model_name(model_type)
    if name == "ridge":
        alpha = float(params.get("alpha", 1.0))
        coef = np.linalg.solve(train.T @ train + alpha * np.eye(train.shape[1]), train.T @ yc)
        return test @ coef + center
    if name == "ols":
        coef = np.linalg.lstsq(np.column_stack([np.ones(len(train)), train]), y, rcond=None)[0]
        return np.column_stack([np.ones(len(test)), test]) @ coef
    if name == "tree":
        # Small deterministic regression tree, recursively splitting by squared error.
        def grow(tx, ty, vx, depth):
            if depth <= 0 or len(ty) < int(params.get("min_samples_leaf", 8)) * 2:
                return np.full(len(vx), float(ty.mean()))
            best = None
            for col in range(tx.shape[1]):
                threshold = np.median(tx[:, col])
                left = tx[:, col] <= threshold
                if left.sum() < 2 or (~left).sum() < 2:
                    continue
                loss = ((ty[left] - ty[left].mean()) ** 2).sum() + ((ty[~left] - ty[~left].mean()) ** 2).sum()
                if best is None or loss < best[0]:
                    best = (loss, col, threshold, left)
            if best is None:
                return np.full(len(vx), float(ty.mean()))
            _, col, threshold, left = best
            out = np.empty(len(vx))
            mask = vx[:, col] <= threshold
            out[mask] = grow(tx[left], ty[left], vx[mask], depth - 1)
            out[~mask] = grow(tx[~left], ty[~left], vx[~mask], depth - 1)
            return out
        return grow(train, y, test, int(params.get("max_depth", 3)))
    if name == "neural_network":
        rng = np.random.default_rng(int(params.get("seed", 0)))
        hidden = int(params.get("hidden_units", 16))
        w1 = rng.normal(0, 1 / np.sqrt(train.shape[1]), (train.shape[1], hidden))
        b1 = np.zeros(hidden)
        w2 = np.zeros(hidden)
        b2 = 0.0
        lr = float(params.get("learning_rate", 0.01))
        for _ in range(int(params.get("epochs", 120))):
            h = np.tanh(train @ w1 + b1)
            pred = h @ w2 + b2
            err = (pred - yc) / len(y)
            gw2 = h.T @ err
            gb2 = err.sum()
            gh = (err[:, None] * w2) * (1 - h * h)
            w1 -= lr * train.T @ gh
            b1 -= lr * gh.sum(axis=0)
            w2 -= lr * gw2
            b2 -= lr * gb2
        return np.tanh(test @ w1 + b1) @ w2 + b2 + center
    raise AssertionError("canonical_model_name returned an unsupported model")


def prefilter_features(X, y, *, min_abs_correlation=0.0, max_features=None):
    """Cheap univariate Pearson filter; callers must pass training rows only."""
    matrix, names = _matrix(X)
    target = np.asarray(y, dtype=float)
    scores = []
    for j, name in enumerate(names):
        valid = np.isfinite(matrix[:, j]) & np.isfinite(target)
        corr = np.corrcoef(matrix[valid, j], target[valid])[0, 1] if valid.sum() >= 3 else np.nan
        if np.isfinite(corr) and abs(corr) >= min_abs_correlation:
            scores.append((name, abs(float(corr)), j))
    scores.sort(key=lambda row: row[1], reverse=True)
    if max_features is not None:
        scores = scores[:max(0, int(max_features))]
    return [row[0] for row in scores]


def ridge_select_features(X, y, *, alpha=1.0, max_features=None, min_weight=1e-8):
    """Rank features using training-only Ridge coefficients, then remove collinear extras."""
    ranked = [(name, weight) for name, weight in _ridge_importance(X, np.asarray(y, float), alpha)
              if weight >= min_weight]
    matrix, names = _matrix(X)
    selected = []
    for name, _ in ranked:
        idx = names.index(name)
        if selected:
            prior = [names.index(item) for item in selected]
            candidate = matrix[:, idx]
            redundant = False
            for col in prior:
                a, b = matrix[:, col], candidate
                valid = np.isfinite(a) & np.isfinite(b)
                if valid.sum() >= 3 and np.std(a[valid]) > 1e-12 and np.std(b[valid]) > 1e-12:
                    if abs(np.corrcoef(a[valid], b[valid])[0, 1]) >= 0.98:
                        redundant = True
                        break
            if redundant:
                continue
        selected.append(name)
        if max_features is not None and len(selected) >= max_features:
            break
    return selected


def _ridge_importance(X, y, alpha):
    x, names = _matrix(X)
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x = x[valid]
    target = np.asarray(y, float)[valid]
    if not len(target):
        return []
    x, _ = _impute_scale(x, x)
    yy = target - target.mean()
    coef = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ yy)
    return sorted(zip(names, np.abs(coef)), key=lambda item: item[1], reverse=True)


def test_feature(X, y, *, model_type="ridge", model_params=None, purge_rows=0, min_samples=30, **kwargs):
    return run_purged_walk_forward(X, y, model_type=model_type, model_params=model_params,
                                   purge_rows=purge_rows, min_samples=min_samples, **kwargs)


def test_feature_set(X, y, *, model_type="ridge", model_params=None, purge_rows=0, min_samples=30, **kwargs):
    return run_purged_walk_forward(X, y, model_type=model_type, model_params=model_params,
                                   purge_rows=purge_rows, min_samples=min_samples, **kwargs)


def run_walk_forward_test(X, y, *, model_type="ridge", model_params=None, folds=4, min_samples=30):
    return run_purged_walk_forward(X, y, model_type=model_type, model_params=model_params,
                                   folds=folds, purge_rows=0, min_samples=min_samples)


def run_purged_walk_forward(X, y, *, model_type="ridge", model_params=None, folds=4, purge_rows=0,
                            min_samples=30, regime_labels=None, regime_id=None, episode_ids=None):
    """Expanding chronological folds, with embargo, optionally scored on one latent only."""
    canonical_model = canonical_model_name(model_type)
    matrix, names = _matrix(X)
    target = np.asarray(y, dtype=float)
    if len(matrix) != len(target):
        raise ValueError("X and y row counts must match")
    eligible = np.isfinite(target) & np.isfinite(matrix).any(axis=1)
    if regime_labels is not None:
        labels = np.asarray(regime_labels, dtype=object)
        normalized = np.asarray([str(x).casefold() if pd.notna(x) else "null" for x in labels], dtype=object)
        if regime_id is not None:
            eligible &= normalized == str(regime_id).casefold()
        else:
            eligible &= pd.notna(labels) & (normalized != "null")
        if episode_ids is None:
            episode_ids = np.cumsum(np.r_[True, normalized[1:] != normalized[:-1]])
    if episode_ids is not None:
        episode_ids = np.asarray(episode_ids)
        if len(episode_ids) != len(target):
            raise ValueError("episode_ids must align with X and y")
    positions = np.flatnonzero(eligible)
    if len(positions) < min_samples:
        return {"model": canonical_model, "folds": [], "n_samples": len(positions), "prediction_error": np.nan,
                "directional_accuracy": np.nan, "expectancy": np.nan, "stability": np.nan}
    if episode_ids is not None:
        episode_order = list(dict.fromkeys(episode_ids[positions].tolist()))
        # Hold out individual recurring episodes. Use the latest folds while
        # every earlier episode remains available to the expanding train set.
        test_episodes = episode_order[1:][-max(1, int(folds)):]
        episode_chunks = [np.asarray([episode], dtype=object) for episode in test_episodes]
        splits = []
        episode_ordinal = {episode: i for i, episode in enumerate(episode_order)}
        for chunk in episode_chunks:
            if len(chunk):
                first = episode_ordinal[chunk[0]]
                train_pos = positions[np.asarray([episode_ordinal[e] < first for e in episode_ids[positions]])]
                test_pos = positions[np.isin(episode_ids[positions], chunk)]
                splits.append((train_pos, test_pos, chunk[0]))
    else:
        chunks = np.array_split(positions, max(2, int(folds) + 1))[1:]
        splits = []
        for chunk in chunks:
            if len(chunk):
                start = int(chunk[0])
                train_pos = positions[positions < max(0, start - int(purge_rows))]
                splits.append((train_pos, chunk, None))
    fold_metrics = []
    for train_pos, test_pos, episode_id in tqdm(splits, desc="Evaluating Folds", leave=False):
        train_pos = train_pos[train_pos < max(0, int(test_pos[0]) - int(purge_rows))]
        if len(train_pos) < min_samples or len(test_pos) == 0:
            continue
        pred = fit_predictive_model(canonical_model, matrix[train_pos], target[train_pos], matrix[test_pos], model_params)
        actual = target[test_pos]
        error = float(np.sqrt(np.mean((actual - pred) ** 2)))
        train_mean = float(np.mean(target[train_pos]))
        fold_target_scale = float(np.std(target[train_pos]))
        signal = np.sign(pred - train_mean)
        centered_actual = actual - train_mean
        direction = float(np.mean(signal == np.sign(centered_actual)))
        expectancy = float(np.mean(signal * centered_actual))
        fold_metrics.append({"n_train": len(train_pos), "n_test": len(test_pos), "episode_id": episode_id,
                             "prediction_error": error, "directional_accuracy": direction,
                             "expectancy": expectancy, "target_scale": fold_target_scale})
    if not fold_metrics:
        return {"model": canonical_model, "folds": [], "n_samples": len(positions), "prediction_error": np.nan,
                "directional_accuracy": np.nan, "expectancy": np.nan, "stability": np.nan}
    errors = np.array([f["prediction_error"] for f in fold_metrics])
    acc = np.array([f["directional_accuracy"] for f in fold_metrics])
    exp = np.array([f["expectancy"] for f in fold_metrics])
    target_scale = float(np.nanmean([f["target_scale"] for f in fold_metrics]))
    stability = float(np.clip(1.0 - np.std(exp) / max(target_scale, 1e-12), 0.0, 1.0))
    signs = np.sign(exp[np.abs(exp) > 1e-12])
    recurrence_consistency = float(max(np.mean(signs > 0), np.mean(signs < 0))) if signs.size else np.nan
    return {"model": canonical_model, "folds": fold_metrics, "n_samples": len(positions),
            "prediction_error": float(np.mean(errors)), "directional_accuracy": float(np.mean(acc)),
            "expectancy": float(np.mean(exp)), "stability": stability,
            "expectancy_std": float(np.std(exp)), "target_scale": target_scale,
            "episode_expectancy_std": float(np.std(exp)), "recurrence_consistency": recurrence_consistency,
            "episode_count": len({str(f["episode_id"]) for f in fold_metrics if f["episode_id"] is not None}),
            "feature_names": names}


def calculate_feature_stability(fold_scores, target_scale=None):
    values = np.asarray(fold_scores, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size or target_scale is None or not np.isfinite(target_scale):
        return np.nan
    return float(np.clip(1.0 - np.std(values) / max(float(target_scale), 1e-12), 0.0, 1.0))


def calculate_model_stability(results):
    scales = np.asarray([row.get("target_scale", np.nan) for row in results], dtype=float)
    scale = float(np.nanmean(scales)) if np.isfinite(scales).any() else None
    episode_scores = [fold.get("expectancy", np.nan) for row in results for fold in row.get("folds", ())]
    scores = episode_scores or [row.get("expectancy", np.nan) for row in results]
    return calculate_feature_stability(scores, scale)


def calculate_prediction_error(y_true, y_pred):
    return float(np.sqrt(np.nanmean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


def calculate_directional_accuracy(y_true, y_pred, train_mean=None):
    baseline = float(np.nanmean(y_true)) if train_mean is None else float(train_mean)
    return float(np.nanmean(np.sign(np.asarray(y_pred) - baseline) == np.sign(np.asarray(y_true) - baseline)))


def calculate_expectancy(y_true, y_pred, train_mean=None):
    actual = np.asarray(y_true, float)
    baseline = float(np.nanmean(actual)) if train_mean is None else float(train_mean)
    return float(np.nanmean(np.sign(np.asarray(y_pred) - baseline) * (actual - baseline)))


def calculate_complexity_penalty(complexity, coefficient=0.0005):
    return float(coefficient * np.log1p(max(0, complexity)))


def calculate_fitness(result, *, complexity=1, complexity_penalty=0.0005):
    expectancy = result.get("expectancy", np.nan)
    stability = result.get("stability", np.nan)
    error = result.get("prediction_error", np.nan)
    if not np.isfinite(expectancy) or not np.isfinite(stability) or not np.isfinite(error):
        return -np.inf
    scale = result.get("target_scale", np.nan)
    if not np.isfinite(scale) or scale <= 0:
        return -np.inf
    return float((expectancy / scale) * stability - 0.05 * (error / scale)
                 - calculate_complexity_penalty(complexity, complexity_penalty))


test_feature.__test__ = False
test_feature_set.__test__ = False
