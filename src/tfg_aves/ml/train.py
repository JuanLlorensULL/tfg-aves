"""Entrenamiento de los tres modelos supervisados con configuración fija (§8.6)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


class _CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Transformer que aplica LabelEncoding a las columnas categóricas indicadas.

    Se ajusta sobre todos los valores vistos (train + val) para evitar
    valores desconocidos en predicción.
    """

    def __init__(self, categorical_cols: list[str]) -> None:
        self.categorical_cols = categorical_cols

    def fit(self, X: pd.DataFrame, y: object = None) -> _CategoricalEncoder:
        self.encoders_: dict[str, LabelEncoder] = {}
        for col in self.categorical_cols:
            if col not in X.columns:
                continue
            enc = LabelEncoder()
            enc.fit(X[col].astype(str))
            self.encoders_[col] = enc
        return self

    def transform(self, X: pd.DataFrame, y: object = None) -> pd.DataFrame:
        X_out = X.copy()
        for col, enc in self.encoders_.items():
            # Valores desconocidos se asignan a la clase 0 de forma segura
            known = set(enc.classes_)
            X_out[col] = X_out[col].astype(str).map(
                lambda v, k=known, e=enc: e.transform([v])[0] if v in k else 0
            )
        return X_out


class _XGBoostWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper fino sobre XGBClassifier para manejar label encoding de categóricas.

    Aplica _CategoricalEncoder internamente en fit/predict para que la interfaz
    externa acepte DataFrames crudos con columnas de tipo str.
    """

    def __init__(self, categorical_cols: list[str], seed: int = 0) -> None:
        self.categorical_cols = categorical_cols
        self.seed = seed

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> _XGBoostWrapper:
        all_X = pd.concat([X, X_val], ignore_index=True) if X_val is not None else X
        self._enc = _CategoricalEncoder(self.categorical_cols)
        self._enc.fit(all_X)

        X_enc = self._enc.transform(X)
        self._xgb = xgb.XGBClassifier(
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=1000,
            objective="multi:softprob",
            tree_method="hist",
            random_state=self.seed,
            n_jobs=-1,
            early_stopping_rounds=50,
            eval_metric="mlogloss",
        )
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_enc = self._enc.transform(X_val)
            eval_set = [(X_val_enc, y_val)]
        self._xgb.fit(X_enc, y, eval_set=eval_set, verbose=False)
        self.classes_ = self._xgb.classes_
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._xgb.predict_proba(self._enc.transform(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._xgb.predict(self._enc.transform(X))

    @property
    def best_iteration(self) -> int | None:
        """Iteración óptima detectada por early stopping (None si no aplica)."""
        return getattr(self._xgb, "best_iteration", None)


class _LightGBMWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper fino sobre LGBMClassifier para manejar categóricas nativas.

    Convierte las columnas categóricas a dtype 'category' antes de fit/predict,
    garantizando consistencia de categorías entre train y predicción.
    """

    def __init__(
        self,
        categorical_cols: list[str],
        seed: int = 0,
    ) -> None:
        self.categorical_cols = categorical_cols
        self.seed = seed
        self._lgbm = lgb.LGBMClassifier(
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=1000,
            objective="multiclass",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

    def _to_categorical(
        self, X: pd.DataFrame, categories: dict[str, list] | None = None
    ) -> tuple[pd.DataFrame, dict[str, list]]:
        """Convierte columnas a dtype category; retorna el mapeo de categorías."""
        X_out = X.copy()
        cats: dict[str, list] = {}
        for col in self.categorical_cols:
            if col not in X_out.columns:
                continue
            if categories is None:
                X_out[col] = X_out[col].astype("category")
            else:
                X_out[col] = pd.Categorical(X_out[col], categories=categories[col])
            cats[col] = list(X_out[col].cat.categories)
        return X_out, cats

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> _LightGBMWrapper:
        X_train_lgb, self._train_categories = self._to_categorical(X)
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_lgb, _ = self._to_categorical(X_val, self._train_categories)
            eval_set = [(X_val_lgb, y_val)]
        cat_feature = [c for c in self.categorical_cols if c in X.columns]
        self._lgbm.fit(
            X_train_lgb, y,
            eval_set=eval_set,
            eval_metric="multi_logloss",
            categorical_feature=cat_feature,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        self.classes_ = self._lgbm.classes_
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_lgb, _ = self._to_categorical(X, self._train_categories)
        return self._lgbm.predict_proba(X_lgb)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_lgb, _ = self._to_categorical(X, self._train_categories)
        return self._lgbm.predict(X_lgb)


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    categorical_cols: list[str],
    seed: int = 0,
) -> ClassifierMixin:
    """Entrena RF con la configuración de §8.6: n_estimators=300, max_depth=12,
    min_samples_leaf=10. Aplica label encoding a categóricas.

    Devuelve un Pipeline (encoder + RF) que acepta DataFrames crudos en
    predict / predict_proba.
    """
    # Ajustar encoder sobre train (sin val: RF no usa val)
    enc = _CategoricalEncoder(categorical_cols)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        random_state=seed,
        n_jobs=-1,
    )
    pipeline = Pipeline([("encoder", enc), ("model", rf)])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    categorical_cols: list[str],
    seed: int = 0,
) -> ClassifierMixin:
    """Entrena XGBoost con la configuración de §8.6 + early stopping=50.

    Devuelve un _XGBoostWrapper que acepta DataFrames crudos en
    predict / predict_proba.
    """
    wrapper = _XGBoostWrapper(categorical_cols=categorical_cols, seed=seed)
    wrapper.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    return wrapper


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    categorical_cols: list[str],
    seed: int = 0,
) -> ClassifierMixin:
    """Entrena LightGBM con categóricas nativas + early stopping=50.

    Devuelve un wrapper que acepta DataFrames crudos en predict / predict_proba.
    """
    wrapper = _LightGBMWrapper(categorical_cols=categorical_cols, seed=seed)
    wrapper.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    return wrapper
