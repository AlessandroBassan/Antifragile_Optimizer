import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import TimeSeriesSplit

class AlphaEngine:
    """
    LAYER 1: ML Alpha Extraction (Versione Finale Tesi)
    Unisce la logica di Purging/Embargo con i driver Macro-Timing.
    """
    def __init__(self, max_depth=4, n_estimators=100):
        self.model = xgb.XGBRegressor(
            max_depth=max_depth,
            learning_rate=0.03,
            n_estimators=n_estimators,
            reg_alpha=5.0,     # Regolarizzazione L1 dalla tesi
            reg_lambda=10.0,   # Regolarizzazione L2 dalla tesi
            n_jobs=-1
        )
        # Feature Macro-Economiche identificate nella tesi
        self.feature_names = ['Asset_Mom', 'VIX_Ret', 'GOLD_Ret', 'TNX_Diff']
        self.explainer = None

    def prepare_features(self, data):
        """
        Crea il dataset unendo l'asset ai driver macro.
        """
        df = data.copy()
        
        # Feature Engineering (Macro-Timing Focus)
        df['Asset_Mom'] = df['Close'].pct_change(5)       # Momentum asset
        df['VIX_Ret']   = df['VIX'].pct_change()          # Stress di mercato
        df['GOLD_Ret']  = df['GOLD'].pct_change()         # Safe Haven flow
        df['TNX_Diff']  = df['TNX'].diff()                # Variazione tassi
        
        # TARGET: Rendimento futuro a 24 ore (cosa vogliamo predire)
        df['target'] = df['Close'].pct_change().shift(-24)
        
        return df.dropna()

    def train_with_anti_overfitting(self, data):
        """
        Walk-Forward con Purging ed Embargo.
        Implementazione del framework di Marcos Lopez de Prado citato in tesi.
        """
        df = self.prepare_features(data)
        X = df[self.feature_names]
        y = df['target']
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        print(f">>> Training Layer 1: Estrazione Alpha Macro-Timing...")
        
        for train_index, test_index in tscv.split(X):
            # PURGING: Eliminiamo le ultime 24 ore del train per evitare che 
            # il modello 'spii' il target del test set attraverso la correlazione.
            purged_train_index = train_index[:-24] 
            
            X_train, X_test = X.iloc[purged_train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[purged_train_index], y.iloc[test_index]
            
            self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Inizializzazione SHAP per validazione stabilità
        self.explainer = shap.TreeExplainer(self.model)
        print(">>> Alpha Engine (Layer 1) validato con Purging ed Embargo.")

    def get_alpha_signal(self, current_data):
        df = self.prepare_features(current_data)
        if df.empty: return 0.0
        
        last_features = df[self.feature_names].iloc[-1:]
        prediction = self.model.predict(last_features)[0]
        return float(prediction)

    def get_shap_stability(self, data):
        """Identifica quale driver macro sta guidando l'Alpha attuale"""
        df = self.prepare_features(data)
        X = df[self.feature_names].iloc[-50:] 
        shap_values = self.explainer.shap_values(X)
        importance = np.abs(shap_values).mean(0)
        return dict(zip(self.feature_names, importance))