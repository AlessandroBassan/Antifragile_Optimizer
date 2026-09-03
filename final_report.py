import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import traceback
import numpy as np

# --- CONFIGURAZIONE PERCORSI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

try:
    from utils.data_loader import get_full_alpha_data
    from utils.metrics import calculate_dsr, calculate_standard_metrics
    from core.alpha_engine import AlphaEngine
    from core.physics_engine import PhysicsEngine
    from core.stochastic_opt import StochasticOptimizer
    from core.backtester import BacktestEngine
    from core.macro_sentiment import SentimentEngine
    from core.structural_overlay import StructuralOverlay
except ImportError as e:
    print(f"Errore Importazione Moduli: {e}")
    input("Premi INVIO per uscire...")
    sys.exit()

def run_integrated_validation():
    # --- CONFIGURAZIONE TARGET ---
    ticker = "GC=F" # Cambia in SPY, BTC-USD, GC=F
    
    print(f"\n>>> SISTEMA ATTIVO SU PYTHON {sys.version_info.major}.{sys.version_info.minor}")
    print("="*65)
    print(f"   REPORT INTEGRATO TESI: {ticker} (L1 + L2 + L3 + L4)")
    print("="*65)
    
    try:
        # 1. DATA INGESTION
        print(f">>> Scaricamento Dataset Macro Completo...")
        data = get_full_alpha_data(ticker=ticker, period="1y", interval="1h")
        
        # 2. LAYER 1: ALPHA ML
        print("\n[L1] Training XGBoost con Purging ed Embargo...")
        alpha_e = AlphaEngine()
        alpha_e.train_with_anti_overfitting(data)
        expected_alpha = alpha_e.get_alpha_signal(data)
        shap_values = alpha_e.get_shap_stability(data)
        top_macro = max(shap_values, key=shap_values.get) if shap_values else "N/A"
        
        # 3. LAYER 2 & 3: FISICA E OTTIMIZZAZIONE
        physics_e = PhysicsEngine()
        llm_e = SentimentEngine()
        opt_e = StochasticOptimizer()
        
        state = physics_e.compute_state(data[['Close']])
        headlines = [f"{ticker} displays resilience", "Macroeconomic headwinds", "Fed policy outlook"]
        sentiment_score = llm_e.analyze_news(headlines)
        context = llm_e.get_contextual_bias(sentiment_score)
        
        decision = opt_e.solve_barbell(state, data[['Close']])

        # 4. LAYER 4: ACTIVE STRUCTURAL OVERLAY
        current_tnx_yield = data['TNX'].iloc[-1] 
        overlay_e = StructuralOverlay(commission_rate=0.001)
        l4_results = overlay_e.apply_active_overlay(decision, current_tnx_yield, expected_alpha)

        # --- OUTPUT DASHBOARD SNAPSHOT (L1-L2-L3) ---
        print("\n" + "#"*65)
        print("             CURRENT MARKET SNAPSHOT (DASHBOARD)")
        print("#"*65)
        print(f"L1 PREDICTED ALPHA:    {expected_alpha*100:.4f}%")
        print(f"L1 TOP MACRO DRIVER:   {top_macro}")
        print(f"L2 HURST EXPONENT:     {state['hurst']:.4f} ({state['regime']})")
        print(f"L2 SHILLER Z-SCORE:    {state['z_score_shiller']:.4f}")
        print(f"L2 SENTIMENT SCORE:    {sentiment_score:.2f} ({context})")
        print(f"L3 REGOLA ATTIVA:      {decision['rule_applied']}")
        print(f"L3 DAILY THETA COST:   {decision['theta_decay']*100:.4f}%")

        # --- OUTPUT DASHBOARD LAYER 4 ---
        print("#"*65)
        print("             LAYER 4: ACTIVE WEALTH CONTROLLER")
        print("#"*65)
        print(f"MARKET ^TNX YIELD:     {current_tnx_yield:.2f}%")
        print(f"BUDGET STATUS:         {l4_results['budget_status']}")
        print(f"EXECUTION STATUS:      {l4_results['execution_status']}")
        print("-" * 65)
        print(f"FINAL P1 (Ultra-Safe): {l4_results['p1_safe']*100:.1f}%")
        print(f"FINAL P2 (Mid-Risk):   {l4_results['p2_mid']*100:.1f}%")
        print(f"FINAL P3 (Ultra-Risk): {l4_results['p3_ultra']*100:.1f}%")
        print("-" * 65)
        print(f"CARRY-TO-THETA RATIO:  {l4_results['funding_ratio']:.2f}x")
        print(f"EST. NET CARRY (Real): {l4_results['net_carry_annual']*100:.2f}% p.a.")
        print("#"*65)

        # 5. BACKTEST
        print("\n>>> Avvio Simulazione Storica Walk-Forward...")
        bt = BacktestEngine(initial_capital=100000)
        results = bt.run(data, alpha_e, physics_e, opt_e)
        
        # 6. METRICHE
        rets = results['daily_ret'].dropna()
        sr_grezzo, dsr_value = calculate_dsr(rets, n_trials=100)
        metrics = calculate_standard_metrics(rets)
        
        print("\n" + "="*55)
        print("             REPORT PERFORMANCE STORICA")
        print("="*55)
        print(f"RENDIMENTO TOTALE:   {metrics.get('Total Return', 0)*100:.2f}%")
        print(f"SHARPE RATIO:        {sr_grezzo:.2f}")
        print(f"DSR (CONFIDENZA):    {dsr_value*100:.1f}%")
        print(f"MAX DRAWDOWN:        {metrics.get('Max Drawdown', 0)*100:.2f}%")
        print("-" * 55)
        status = "LODE (Significativo)" if dsr_value > 0.95 else "NON SIGNIFICATIVO"
        print(f"STATUS VALIDAZIONE:  {status}")
        print("="*55)

        # 7. GRAFICI
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        ax1.plot(results['date'], results['equity'], label='Triple-Layer Strategy', color='#0047AB', lw=2)
        ax1.plot(results['date'], results['benchmark'], label='Buy & Hold', color='gray', ls='--', alpha=0.6)
        ax1.set_title(f"Equity Curve: Antifragile vs Market ({ticker})")
        ax1.legend(); ax1.grid(alpha=0.2)
        ax2.plot(results['date'], results['hurst'], color='#8E44AD', label='Hurst Exponent (Filtered)')
        ax2.axhline(0.5, color='black', ls='--')
        ax2.set_title("Fractal Regime Dynamics")
        ax2.legend(); ax2.grid(alpha=0.2)
        ax3.fill_between(results['date'], results['convex_w']*100, color='#E67E22', alpha=0.4, label='Convex Exposure %')
        ax3.set_title("Dynamic Barbell Allocation")
        ax3.legend(); ax3.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig('tesi_final_dashboard.png', dpi=300)
        print("\n>>> Grafico salvato come: tesi_final_dashboard.png")
        plt.show()

    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    run_integrated_validation()
    input("\nProcesso terminato. Premi INVIO per chiudere...")