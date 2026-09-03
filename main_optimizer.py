import sys
import os
import traceback
import pandas as pd
import numpy as np

# --- CONFIGURAZIONE PERCORSI (Per trovare core/ e utils/) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

try:
    from utils.data_loader import get_full_alpha_data
    from core.alpha_engine import AlphaEngine          # Layer 1: ML Alpha (XGBoost + Purging)
    from core.physics_engine import PhysicsEngine      # Layer 2: Econofisica (Hurst, LZC, Shiller Z-Score)
    from core.macro_sentiment import SentimentEngine   # Layer 2: LLM (FinBERT)
    from core.stochastic_opt import StochasticOptimizer # Layer 3: Programmazione Stocastica & Barbell

    def run_main_optimizer():
        print(f"\n>>> Sistema attivo su Python {sys.version_info.major}.{sys.version_info.minor}")
        print("\n" + "="*65)
        print("   ANTIFRAGILE OPTIMIZER (L1 + L2 + L3 FULL INTEGRATION)")
        print("="*65)

        # --- 1. DATA INGESTION (Asset + Macro Drivers) ---
        ticker = "AAPL"
        # Scarichiamo Asset + VIX + GOLD + TNX (10Y Yield) per il Macro-Timing
        raw_data = get_full_alpha_data(ticker=ticker, period="6mo", interval="1h")
        
        # --- 2. LAYER 1: ALPHA EXTRACTION (XGBoost + Purging & Embargo) ---
        print("\n[L1] Training XGBoost con Purging ed Embargo (Anti-Overfitting)...")
        alpha_model = AlphaEngine()
        alpha_model.train_with_anti_overfitting(raw_data)
        
        expected_alpha = alpha_model.get_alpha_signal(raw_data)
        shap_values = alpha_model.get_shap_stability(raw_data)
        top_macro_driver = max(shap_values, key=shap_values.get) if shap_values else "N/A"

        # --- 3. LAYER 2: REGIME MAPPING (Econofisica + Z-Score Shiller + NLP) ---
        print("\n[L2] Analisi Fisica (Hurst, LZC, Shiller Z-Score) e Sentiment (FinBERT)...")
        p_engine = PhysicsEngine(window_size=200)
        state = p_engine.compute_state(raw_data[['Close']])
        
        if not state:
            print("[!] Errore: Dati insufficienti per il calcolo dei regimi fisici (min. 200 ore).")
            return

        # Analisi Sentiment LLM
        llm = SentimentEngine()
        headlines = [
            f"{ticker} displays strong resilience in current macro environment",
            "Federal Reserve policy uncertainty remains a key market driver",
            "Tech institutional demand offset by rising 10Y Treasury yields"
        ]
        sentiment_score = llm.analyze_news(headlines)
        context = llm.get_contextual_bias(sentiment_score)

        # --- 4. LAYER 3: STOCHASTIC OPTIMIZATION (Barbell & Lindy/Shiller) ---
        print("\n[L3] Generazione Scenari Frattali e Ottimizzazione Barbell...")
        optimizer = StochasticOptimizer(n_scenarios=5000)
        decision = optimizer.solve_barbell(state, raw_data[['Close']])

        # --- 5. LOGICA DI FUSIONE E SOSTENIBILITÀ (Alpha vs Theta) ---
        theta_cost = decision['theta_decay']
        is_sustainable = expected_alpha > theta_cost
        
        final_safe = decision['safe_weight']
        final_convex = decision['convex_weight']
        
        if not is_sustainable:
            final_safe, final_convex = 1.0, 0.0
            status_alert = "DE-RISKING (Alpha < Theta Cost)"
        elif context == "BEARISH" and state['hurst'] < 0.55:
            final_safe, final_convex = 1.0, 0.0
            status_alert = "DE-RISKING (Sentiment Mismatch)"
        else:
            status_alert = "SUSTAINABLE (All Layers Confirmed)"

        # --- 6. OUTPUT: QUANT TRADER FINAL DASHBOARD ---
        print("\n" + "#"*65)
        print("             QUANT TRADER FINAL DASHBOARD")
        print("#"*65)
        print(f"ASSET ANALIZZATO:      {ticker}")
        print("-" * 65)
        print(f"[L1] PREDICTED ALPHA:  {expected_alpha*100:.4f}%")
        print(f"[L1] TOP MACRO DRIVER: {top_macro_driver}")
        print("-" * 65)
        print(f"[L2] HURST EXPONENT:   {state['hurst']:.4f} ({state['regime']})")
        print(f"[L2] SHILLER Z-SCORE:  {state['z_score_shiller']:.4f}")
        print(f"[L2] SENTIMENT SCORE:  {sentiment_score:.2f} ({context})")
        print("-" * 65)
        print(f"[L3] REGOLA ATTIVA:    {decision['rule_applied']}")
        print(f"[L3] DAILY THETA COST: {theta_cost*100:.4f}%")
        print("-" * 65)
        print(f"ALLOCAZIONE SAFE:      {final_safe*100:.1f}%")
        print(f"ALLOCAZIONE CONVEX:    {final_convex*100:.1f}%")
        print("-" * 65)
        print(f"SYSTEM STATUS:         {status_alert}")
        print("#"*65)

    if __name__ == "__main__":
        run_main_optimizer()

except Exception:
    print("\n" + "!"*65)
    print("ERRORE CRITICO RILEVATO:")
    traceback.print_exc()
    print("!"*65)

finally:
    print("\n" + "-"*65)
    input("PROCESSO TERMINATO. Premi INVIO per chiudere questa finestra...")