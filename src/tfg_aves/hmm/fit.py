"""Entrenamiento de HMM con k-means init + EM y múltiples restarts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans


def build_sequences(
    df_features: pd.DataFrame,
    bird_ids: list[str],
    feature_cols: list[str],
) -> tuple[np.ndarray, list[int]]:
    """Concatena tramos consecutivos válidos por ave en (X, lengths) para hmmlearn."""
    X_parts: list[np.ndarray] = []
    lengths: list[int] = []
    for bid in bird_ids:
        sub = df_features[
            (df_features["bird_id"] == bid) & df_features["is_observation_valid"]
        ].sort_values("date_utc")
        if len(sub) == 0:
            continue
        # Detectar tramos consecutivos por fecha.
        dates = pd.to_datetime(sub["date_utc"]).reset_index(drop=True)
        gap = (dates.diff() != pd.Timedelta(days=1)).cumsum()
        for _, segment in sub.groupby(gap.values):
            if len(segment) > 0:
                X_parts.append(segment[feature_cols].to_numpy(dtype=np.float64))
                lengths.append(len(segment))
    if not X_parts:
        return np.zeros((0, len(feature_cols)), dtype=np.float64), []
    return np.vstack(X_parts), lengths


def _initialize_hmm_with_kmeans(
    X: np.ndarray, n_components: int, random_state: int
) -> GaussianHMM:
    """Inicializa GaussianHMM con medias y varianzas de k-means."""
    kmeans = KMeans(n_clusters=n_components, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(X)
    means_init = kmeans.cluster_centers_
    covars_init = np.zeros((n_components, X.shape[1]))
    for k in range(n_components):
        mask = labels == k
        if mask.sum() < 2:
            covars_init[k] = np.ones(X.shape[1])
        else:
            covars_init[k] = np.var(X[mask], axis=0) + 1e-6

    hmm = GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        init_params="",
        n_iter=200,
        tol=1e-4,
        random_state=random_state,
    )
    hmm.startprob_ = np.full(n_components, 1.0 / n_components)
    transmat = np.full((n_components, n_components), 0.1 / (n_components - 1))
    np.fill_diagonal(transmat, 0.9)
    hmm.transmat_ = transmat
    hmm.means_ = means_init
    hmm.covars_ = covars_init
    return hmm


def fit_hmm_with_restarts(
    X_train: np.ndarray,
    lengths_train: list[int],
    n_components: int = 2,
    n_restarts: int = 10,
    random_state: int = 0,
) -> tuple[GaussianHMM, float, list[float]]:
    """Ejecuta k-means+EM n_restarts veces sobre X_train sin estandarización.

    Devuelve (mejor modelo, mejor LL, lista de todas las LL).
    La bimodalidad en km crudos (estacionario ~pocos km, migración ~cientos km)
    hace innecesaria la estandarización y evita diluir la separación entre estados.
    """
    best_model: GaussianHMM | None = None
    best_ll = -np.inf
    all_lls: list[float] = []

    for restart in range(n_restarts):
        seed = random_state + restart
        model = _initialize_hmm_with_kmeans(X_train, n_components, seed)
        model.fit(X_train, lengths_train)
        ll = float(model.score(X_train, lengths_train))
        all_lls.append(ll)
        if ll > best_ll:
            best_ll = ll
            best_model = model

    assert best_model is not None
    return best_model, best_ll, all_lls


