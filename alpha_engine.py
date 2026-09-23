import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from utils.validation import PurgedKFoldWithEmbargo

class AlphaEngine:
    """
    LAYER 1: ML Alpha Extraction con Purging ed Embargo.
    Unisce la logica anti-overfitting di Marcos Lopez de Prado con i driver macro.
    """
    def __init__(self, max_depth=3, n_estimators=100, learning_rate=0.03):
        self.model = xgb.XGBRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            reg_alpha=5.0,     # Penalizzazione L1 (Lasso)
            reg_lambda=10.0,   # Penalizzazione L2 (Ridge)
            random_state=42,
            n_jobs=-1
        )
        self.feature_names = ['Asset_Mom', 'VIX_Ret', 'GOLD_Ret', 'TNX_Diff']
        self.explainer = None
        self.is_fitted = False

    def prepare_features(self, data, is_training=True):
        """
        Crea le feature macro-economiche.
        In modalita' inferenza, NON elimina le ultime 24 ore per garantire la reattivita' real-time.
        """
        df = data.copy()

        # 1. Feature Engineering (Momentum asset + Macro Drivers)
        df['Asset_Mom'] = df['Close'].pct_change(5)       # Momentum a 5 ore
        df['VIX_Ret']   = df['VIX'].pct_change()          # Shock di volatilita'
        df['GOLD_Ret']  = df['GOLD'].pct_change()         # Flusso Safe-Haven
        df['TNX_Diff']  = df['TNX'].diff()                # Variazione tassi Treasury

        if is_training:
            # TARGET: Rendimento CUMULATO a 24 ore (non 1 singola ora futura)
            df['target'] = df['Close'].pct_change(24).shift(-24)
            # In training eliminiamo i NaN sia delle feature sia del target
            return df.dropna(subset=self.feature_names + ['target'])
        else:
            # In inferenza preserviamo l'ultima barra corrente
            return df.dropna(subset=self.feature_names)

    def train_with_anti_overfitting(self, data, n_splits=5, embargo_pct=0.01):
        """
        Addestramento con Purged and Embargoed K-Fold Cross-Validation.
        Elimina il leakage temporale dovuto all'orizzonte di 24 ore del target.
        """
        df = self.prepare_features(data, is_training=True)
        if len(df) < 100:
            raise ValueError(f"Dati insufficienti per il training ({len(df)} righe)")

        X = df[self.feature_names].reset_index(drop=True)
        y = df['target'].reset_index(drop=True)

        cv = PurgedKFoldWithEmbargo(n_splits=n_splits, label_horizon=24, pct_embargo=embargo_pct)
        
        # Addestramento sul fold finale validato
        for train_idx, val_idx in cv.split(X, y):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_val = X.iloc[val_idx], y.iloc[val_idx]
            self.model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_val)],
                verbose=False
            )

        self.is_fitted = True
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            self.explainer = None
            
        print(">>> [L1 AlphaEngine] Training completato con Purged K-Fold ed Embargo.")

    def get_alpha_signal(self, current_data):
        """
        Calcola la predizione di alpha atteso sull'ultima barra disponibile.
        Zero ritardo temporale (no dropna delle ultime 24 ore).
        """
        if not self.is_fitted:
            return 0.0
            
        df = self.prepare_features(current_data, is_training=False)
        if df.empty:
            return 0.0

        last_feature_row = df[self.feature_names].iloc[-1:]
        alpha_pred = self.model.predict(last_feature_row)[0]
        return float(alpha_pred)

    def get_shap_stability(self, data):
        """
        Ritorna l'importanza relativa dei driver macro sul campione recente.
        """
        if not self.is_fitted or self.explainer is None:
            return {f: 0.25 for f in self.feature_names}

        df = self.prepare_features(data, is_training=False)
        if len(df) < 10:
            return {f: 0.25 for f in self.feature_names}

        sample_X = df[self.feature_names].iloc[-50:]
        try:
            shap_vals = self.explainer.shap_values(sample_X)
            mean_importance = np.abs(shap_vals).mean(axis=0)
            total = np.sum(mean_importance)
            if total > 0:
                normalized = mean_importance / total
            else:
                normalized = np.ones(len(self.feature_names)) / len(self.feature_names)
            return dict(zip(self.feature_names, [float(v) for v in normalized]))
        except Exception:
            return {f: 0.25 for f in self.feature_names}
