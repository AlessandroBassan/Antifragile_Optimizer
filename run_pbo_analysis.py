import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from utils.data_loader import get_full_alpha_data
from utils.validation import calculate_pbo, generate_cscv_combinations
from utils.metrics import calculate_dsr
from core.physics_engine import PhysicsEngine
from core.alpha_engine import AlphaEngine
from core.stochastic_opt import StochasticOptimizer
from core.structural_overlay import StructuralOverlay
from core.backtester import BacktestEngine

def run_pbo_grid_analysis(ticker="AAPL"):
    print("=" * 70)
    print(f"   ANALISI PBO & CSCV (PROBABILITY OF BACKTEST OVERFITTING)")
    print(f"   Metodologia di Bailey, Borwein, Lopez de Prado & Zhu (2017)")
    print("=" * 70)

    # 1. Caricamento Dati
    data = get_full_alpha_data(ticker=ticker, period="1y", interval="1h")
    if len(data) < 260:
        print("[!] Dati insufficienti.")
        return

    # Addestramento L1 una tantum con Purged K-Fold
    alpha_e = AlphaEngine()
    alpha_e.train_with_anti_overfitting(data, n_splits=5)
    physics_e = PhysicsEngine(window_size=200, trading_hours_per_day=6.5)

    # 2. Definizione Griglia Parametrica di Strategie Alternative (12 varianti)
    # Variamo: soglia Hurst per il trend Lindy [0.55, 0.58, 0.62]
    # e peso convesso target [0.15, 0.25, 0.35, 0.45]
    hurst_thresholds = [0.55, 0.58, 0.62]
    convex_targets = [0.15, 0.25, 0.35, 0.45]

    strategy_returns = {}
    print("\n>>> Generazione dei rendimenti delle varianti di strategia...")

    for h_thresh in hurst_thresholds:
        for c_target in convex_targets:
            strat_name = f"H_{h_thresh:.2f}_Cw_{c_target:.2f}"
            
            # Istanza ottimizzatore con parametri specifici
            opt_e = StochasticOptimizer(n_scenarios=1000, horizon=24, trading_hours=6.5)
            # Personalizziamo la regola
            class CustomOpt(StochasticOptimizer):
                def solve_barbell(self, state, price_data):
                    res = super().solve_barbell(state, price_data)
                    if state['hurst'] > h_thresh and state['lzc'] < 0.45:
                        res['convex_weight'] = c_target
                        res['safe_weight'] = 1.0 - c_target
                    return res

            c_opt = CustomOpt(n_scenarios=1000, horizon=24, trading_hours=6.5)
            bt = BacktestEngine(initial_capital=100000.0, base_tc=0.0010)
            sim_res = bt.run(data, alpha_e, physics_e, c_opt)
            
            strategy_returns[strat_name] = sim_res['daily_ret'].values

    # Creazione Matrice T x N
    df_matrix = pd.DataFrame(strategy_returns)
    print(f"\n>>> Matrice di Strategie creata: {df_matrix.shape[0]} periodi x {df_matrix.shape[1]} configurazioni.")

    # 3. Calcolo PBO con CSCV (10 partizioni = 252 combinazioni)
    pbo_results = calculate_pbo(df_matrix, n_partitions=10)

    print("\n" + "=" * 70)
    print("                     RISULTATI DIAGNOSTICI PBO")
    print("=" * 70)
    print(f"NUMERO DI PARTIZIONI (S):       10 (Partizioni Combinatorie: {pbo_results['n_combinations']})")
    print(f"PBO (PROBABILITA' OVERFITTING): {pbo_results['pbo'] * 100:.2f}%")
    print(f"MEDIANA RANGO OOS:              {pbo_results['median_oos_rank']:.3f} (Soglia Neutra = 0.500)")
    print(f"MEDIA LOG-ODDS:                 {pbo_results['mean_log_odds']:.3f}")
    print(f"CORRELAZIONE IS vs OOS:         {pbo_results['degradation_corr']:.3f}")
    print(f"SLOPE DETERIORAMENTO (IS->OOS): {pbo_results['degradation_slope']:.3f}")
    status = "NON OVERFITTATO (Generalizzabile)" if not pbo_results['is_overfitted'] else "ATTENZIONE: OVERFITTATO"
    print(f"VERDETTO STATISTICO:            {status}")
    print("=" * 70)

    # 4. Generazione Grafico PBO (Distribuzione Log-Odds e Ranghi OOS)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Istogramma Ranghi OOS
    ax1.hist(pbo_results['rank_distribution'], bins=15, color='#3498DB', edgecolor='black', alpha=0.7)
    ax1.axvline(0.5, color='red', ls='--', lw=2, label='Soglia Mediana (PBO = area a sinistra)')
    ax1.axvline(pbo_results['median_oos_rank'], color='green', ls='-', lw=2, label=f"Mediana Reale ({pbo_results['median_oos_rank']:.2f})")
    ax1.set_title(f"Distribuzione dei Ranghi OOS (PBO = {pbo_results['pbo']*100:.1f}%)", fontweight='bold')
    ax1.set_xlabel("Rango Percentile OOS (1.0 = Migliore)")
    ax1.set_ylabel("Frequenza")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Distribuzione Log-Odds
    ax2.hist(pbo_results['log_odds_distribution'], bins=15, color='#9B59B6', edgecolor='black', alpha=0.7)
    ax2.axvline(0.0, color='red', ls='--', lw=2, label='Log-Odds = 0 (Pari)')
    ax2.axvline(pbo_results['mean_log_odds'], color='gold', ls='-', lw=2, label=f"Media Log-Odds ({pbo_results['mean_log_odds']:.2f})")
    ax2.set_title("Distribuzione Log-Odds di Sovraperformance OOS", fontweight='bold')
    ax2.set_xlabel("Log-Odds Lambda_c")
    ax2.set_ylabel("Frequenza")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(BASE_DIR, "pbo_analysis_report.png")
    plt.savefig(chart_path, dpi=300)
    print(f"\n>>> Grafico salvato come: {chart_path}")

if __name__ == "__main__":
    run_pbo_grid_analysis("AAPL")
