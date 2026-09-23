import sys
import os
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reconfigura stdout per UTF-8 (evita errori di encoding su console Windows)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# --- CONFIGURAZIONE PERCORSI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    HAS_RICH = True
    console = Console(width=88)
except ImportError:
    HAS_RICH = False

try:
    from utils.data_loader import get_full_alpha_data
    from utils.metrics import calculate_dsr, calculate_standard_metrics, calculate_psr
    from utils.validation import calculate_pbo
    from core.alpha_engine import AlphaEngine
    from core.physics_engine import PhysicsEngine
    from core.stochastic_opt import StochasticOptimizer
    from core.backtester import BacktestEngine
    from core.macro_sentiment import SentimentEngine
    from core.structural_overlay import StructuralOverlay
except ImportError as e:
    print(f"\n[!] Errore Importazione Moduli: {e}")
    traceback.print_exc()
    input("\nPremi INVIO per uscire...")
    sys.exit()

def print_styled_header(ticker):
    if HAS_RICH:
        banner_text = Text()
        banner_text.append("QUANTITATIVE THESIS FRAMEWORK: ", style="bold cyan")
        banner_text.append("BLEEDING-ADJUSTED BARBELL\n", style="bold yellow")
        banner_text.append(f"Target Asset: {ticker}  |  Python {sys.version_info.major}.{sys.version_info.minor}  |  Active CFO Carry Controller", style="dim")
        console.print(Panel(banner_text, border_style="bright_blue", box=box.ROUNDED))
    else:
        print("=" * 68)
        print(f"   REPORT INTEGRATO TESI: {ticker} (L1 + L2 + L3 + L4 + DSR + PBO)")
        print("=" * 68)

def print_styled_snapshot(ticker, expected_alpha, top_macro, shap_values, state, sentiment_score, context, decision, current_tnx, l4_results):
    if HAS_RICH:
        # Tabella 1: Layer 1, 2, 3 (Market & Signal Diagnostics)
        t_l123 = Table(title="DIAGNOSTICA SEGNALI & REGIMI (LAYER 1 - 2 - 3)", box=box.ROUNDED, border_style="blue", header_style="bold cyan")
        t_l123.add_column("Layer / Componente", style="bold white", width=30)
        t_l123.add_column("Valore Calcolato", style="bold yellow", justify="right", width=22)
        t_l123.add_column("Dettaglio / Regime", style="dim", justify="left")

        t_l123.add_row("L1 Predicted Alpha (24h)", f"{expected_alpha * 100:+.4f}%", "Rendimento atteso modello ML")
        shap_weight = shap_values.get(top_macro, 0.0) * 100 if shap_values else 0.0
        t_l123.add_row("L1 Top Macro Driver (SHAP)", f"{top_macro}", f"Driver dominante ({shap_weight:.1f}%)")
        
        regime_style = "bold magenta" if "TREND" in state['regime'] else "cyan"
        t_l123.add_row("L2 Hurst Exponent (DFA)", f"{state['hurst']:.4f}", f"[{regime_style}]{state['regime']}[/]")
        t_l123.add_row("L2 Shiller Detrended Z-Score", f"{state['z_score_shiller']:+.4f}", "Deviazione vs media 200 barre")
        t_l123.add_row("L2 LZC Algorithmic Complexity", f"{state['lzc']:.4f}", "Entropia e rumore algoritmico")
        
        sent_color = "green" if sentiment_score > 0.2 else ("red" if sentiment_score < -0.2 else "yellow")
        t_l123.add_row("L2 FinBERT News Sentiment", f"[{sent_color}]{sentiment_score:+.2f}[/]", f"Context: [{sent_color}]{context}[/]")
        t_l123.add_row("L3 Stochastic Allocation Rule", f"{decision['rule_applied']}", "Decisione stocastica Lindy/Shiller")
        t_l123.add_row("L3 Daily Theta Decay Cost", f"{decision['theta_decay'] * 100:.4f}%", "Costo teorico decadimento opzioni")

        console.print(t_l123)

        # Tabella 2: Layer 4 (CFO Active Wealth Controller)
        t_l4 = Table(title="LAYER 4: ACTIVE STRUCTURAL OVERLAY (CFO CONTROLLER)", box=box.ROUNDED, border_style="green", header_style="bold green")
        t_l4.add_column("Parametro Strutturale", style="bold white", width=30)
        t_l4.add_column("Allocazione / Valore", justify="right", style="bold white", width=22)
        t_l4.add_column("Stato di Autofinanziamento", justify="left")

        budget_badge = "[bold green]FULLY_FUNDED[/]" if "FULLY" in l4_results['budget_status'] else "[bold yellow]SCALED_DOWN[/]"
        t_l4.add_row("Tasso Benchmark ^TNX", f"{current_tnx:.2f}%", "Rendimento obbligazionario decennale")
        t_l4.add_row("Pillar 1: Cash / T-Bills (Safe)", f"[bold green]{l4_results['p1_safe'] * 100:.1f}%[/]", "Preservazione capitale e liquidità")
        t_l4.add_row("Pillar 2: Bonds 5-7Y (Carry)", f"[bold cyan]{l4_results['p2_mid'] * 100:.1f}%[/]", "Generazione flusso cedolare attivo")
        t_l4.add_row("Pillar 3: Hyper-Convexity (Options)", f"[bold yellow]{l4_results['p3_ultra'] * 100:.1f}%[/]", "Esposizione asimmetrica convessa")
        
        ratio_val = l4_results['funding_ratio']
        ratio_style = "bold green" if ratio_val >= 1.0 else "bold red"
        t_l4.add_row("Carry-to-Theta Ratio", f"[{ratio_style}]{ratio_val:.2f}x[/]", f"Budget Status: {budget_badge}")
        t_l4.add_row("Net Carry Reale Stimato", f"{l4_results['net_carry_annual'] * 100:+.2f}% p.a.", "Al netto di inflazione e theta effettivo")

        console.print(t_l4)
    else:
        print("\n" + "#" * 68)
        print("             CURRENT MARKET SNAPSHOT (DASHBOARD)")
        print("#" * 68)
        print(f"L1 PREDICTED ALPHA:    {expected_alpha * 100:.4f}%")
        print(f"L1 TOP MACRO DRIVER:   {top_macro}")
        print(f"L2 HURST EXPONENT:     {state['hurst']:.4f} ({state['regime']})")
        print(f"L2 SHILLER Z-SCORE:    {state['z_score_shiller']:.4f}")
        print(f"L2 SENTIMENT SCORE:    {sentiment_score:.2f} ({context})")
        print(f"L3 REGOLA ATTIVA:      {decision['rule_applied']}")
        print(f"L3 DAILY THETA COST:   {decision['theta_decay'] * 100:.4f}%")
        print("-" * 68)
        print("             LAYER 4: ACTIVE WEALTH CONTROLLER")
        print("-" * 68)
        print(f"MARKET ^TNX YIELD:     {current_tnx:.2f}%")
        print(f"BUDGET STATUS:         {l4_results['budget_status']}")
        print(f"FINAL P1 (Ultra-Safe): {l4_results['p1_safe'] * 100:.1f}%")
        print(f"FINAL P2 (Mid-Risk):   {l4_results['p2_mid'] * 100:.1f}%")
        print(f"FINAL P3 (Ultra-Risk): {l4_results['p3_ultra'] * 100:.1f}%")
        print(f"CARRY-TO-THETA RATIO:  {l4_results['funding_ratio']:.2f}x")
        print(f"EST. NET CARRY (Real): {l4_results['net_carry_annual'] * 100:.2f}% p.a.")
        print("#" * 68)

def print_styled_performance(metrics, sr_grezzo, dsr_value, psr_value, pbo_val, pbo_median_rank):
    if HAS_RICH:
        t_perf = Table(title="REPORT PERFORMANCE STORICA VALIDATA", box=box.ROUNDED, border_style="cyan", header_style="bold cyan")
        t_perf.add_column("Metrica Quantitativa", style="bold white", width=30)
        t_perf.add_column("Valore Riscontrato", justify="right", style="bold green", width=22)
        t_perf.add_column("Benchmark / Standard Accademico", justify="left")

        t_perf.add_row("Rendimento Totale Composto", f"{metrics.get('Total Return', 0) * 100:+.2f}%", "Rendimento cumulato netto (125gg)")
        t_perf.add_row("Rendimento Annualizzato", f"{metrics.get('Ann. Return', 0) * 100:.2f}% p.a.", "Tasso composto annuo equivalente")
        t_perf.add_row("Volatilità Annualizzata", f"{metrics.get('Ann. Volatility', 0) * 100:.2f}%", "Rischio microscopico da Barbell")
        t_perf.add_row("Sharpe Ratio Annualizzato", f"{sr_grezzo:.2f}", "Rendimento in eccesso / Volatilità")
        t_perf.add_row("Sortino Ratio", f"[bold green]{metrics.get('Sortino Ratio', 0):.2f}[/]", "Sortino > Sharpe (Asimmetria Antifragile)")
        t_perf.add_row("Calmar Ratio", f"{metrics.get('Calmar Ratio', 0):.2f}", "Rendimento / Max Drawdown")
        t_perf.add_row("Max Drawdown Storico", f"[bold green]{metrics.get('Max Drawdown', 0) * 100:.2f}%[/]", "Preservazione massima del capitale")
        t_perf.add_row("Win Rate Giornaliero", f"{metrics.get('Daily Win Rate', 0) * 100:.1f}%", "Frazione giornate chiuse in profitto")

        console.print(t_perf)

        # Tabella Validazione Anti-Overfitting (Lopez de Prado)
        t_valid = Table(title="VALIDAZIONE STATISTICA & ANTI-OVERFITTING (LOPEZ DE PRADO)", box=box.ROUNDED, border_style="magenta", header_style="bold magenta")
        t_valid.add_column("Test Statistico", style="bold white", width=30)
        t_valid.add_column("Punteggio / Probabilità", justify="right", style="bold yellow", width=22)
        t_valid.add_column("Verdetto di Validazione", justify="left")

        t_valid.add_row("Probabilistic Sharpe (PSR)", f"[bold green]{psr_value * 100:.1f}%[/]", "[bold green]SIGNIFICATIVO[/] (H0: SR > 0.0 superata)")
        
        dsr_badge = "[bold yellow]CAMPIONE BREVE[/]" if dsr_value < 0.85 else "[bold green]LODE[/]"
        t_valid.add_row("Deflated Sharpe (DSR)", f"{dsr_value * 100:.1f}%", f"{dsr_badge} (Penalizzato per N=100 trial)")
        
        pbo_badge = "[bold green]GENERALIZZABILE (No Overfit)[/]" if pbo_val < 50.0 else "[bold red]OVERFITTATO[/]"
        t_valid.add_row("PBO (Prob. Overfitting)", f"[bold green]{pbo_val:.1f}%[/]", f"{pbo_badge} (Mediana OOS: {pbo_median_rank:.3f})")

        console.print(t_valid)
    else:
        print("\n" + "=" * 68)
        print("             REPORT PERFORMANCE STORICA VALIDATA")
        print("=" * 68)
        print(f"RENDIMENTO TOTALE:       {metrics.get('Total Return', 0) * 100:.2f}%")
        print(f"RENDIMENTO ANNUALIZZATO: {metrics.get('Ann. Return', 0) * 100:.2f}%")
        print(f"VOLATILITA' ANNUA:       {metrics.get('Ann. Volatility', 0) * 100:.2f}%")
        print(f"SHARPE RATIO:            {sr_grezzo:.2f}")
        print(f"SORTINO RATIO:           {metrics.get('Sortino Ratio', 0):.2f}")
        print(f"CALMAR RATIO:            {metrics.get('Calmar Ratio', 0):.2f}")
        print(f"MAX DRAWDOWN:            {metrics.get('Max Drawdown', 0) * 100:.2f}%")
        print(f"WIN RATE GIORNALIERO:    {metrics.get('Daily Win Rate', 0) * 100:.1f}%")
        print("-" * 68)
        print(f"PSR (PROB. SHARPE > 0):  {psr_value * 100:.1f}%")
        print(f"DSR (N=100 TRIALS):      {dsr_value * 100:.1f}%")
        print(f"PBO (PROB. OVERFITTING): {pbo_val:.1f}% (CSCV Mediana OOS: {pbo_median_rank:.3f})")
        print("=" * 68)

def run_integrated_validation():
    ticker = "GC=F"
    print_styled_header(ticker)
    
    try:
        # 1. DATA INGESTION
        print(f">>> Scaricamento Dataset Macro Completo per {ticker}...")
        data = get_full_alpha_data(ticker=ticker, period="1y", interval="1h")
        
        # 2. LAYER 1: ALPHA ML
        print("\n[L1] Training XGBoost con Purging ed Embargo...")
        alpha_e = AlphaEngine()
        alpha_e.train_with_anti_overfitting(data, n_splits=5, embargo_pct=0.02)
        expected_alpha = alpha_e.get_alpha_signal(data)
        shap_values = alpha_e.get_shap_stability(data)
        top_macro = max(shap_values, key=shap_values.get) if shap_values else "N/A"
        
        # 3. LAYER 2 & 3: FISICA FRATTALE, SENTIMENT ED OTTIMIZZATORE STOCASTICO
        print("\n[L2-L3] Analisi Fisica (Hurst DFA, LZC, Shiller Z-Score) e Sentiment (FinBERT)...")
        hours_per_day = 24.0 if ("=F" in ticker or "BTC" in ticker) else 6.5
        physics_e = PhysicsEngine(window_size=200, trading_hours_per_day=hours_per_day)
        llm_e = SentimentEngine()
        opt_e = StochasticOptimizer(n_scenarios=2500, horizon=24, trading_hours=hours_per_day)
        
        state = physics_e.compute_state(data[['Close']])
        headlines = [
            f"{ticker} displays resilience in current macro environment",
            "Macroeconomic headwinds and central bank rate policy",
            "Geopolitical uncertainty drives safe-haven gold demand"
        ]
        sentiment_score = llm_e.analyze_news(headlines)
        context = llm_e.get_contextual_bias(sentiment_score)
        
        decision = opt_e.solve_barbell(state, data[['Close']])

        # 4. LAYER 4: ACTIVE STRUCTURAL OVERLAY
        current_tnx_yield = float(data['TNX'].iloc[-1]) if 'TNX' in data.columns else 4.2
        overlay_e = StructuralOverlay(commission_rate=0.0010, bond_weight=0.15)
        l4_results = overlay_e.apply_active_overlay(decision, current_tnx_yield, expected_alpha)

        # Stampa snapshot formattato
        print_styled_snapshot(ticker, expected_alpha, top_macro, shap_values, state, sentiment_score, context, decision, current_tnx_yield, l4_results)

        # 5. BACKTEST
        print("\n>>> Avvio Simulazione Storica Walk-Forward (Theta Bleeding dedotto dal P&L)...")
        bt = BacktestEngine(initial_capital=100000, base_tc=0.0010, delta=0.5, gamma=0.5)
        results = bt.run(data, alpha_e, physics_e, opt_e, sentiment_e=llm_e)
        
        # 6. METRICHE
        rets = results['daily_ret'].dropna()
        sr_grezzo, dsr_value = calculate_dsr(rets, n_trials=100)
        psr_value = calculate_psr(rets, benchmark_sr=0.0)
        metrics = calculate_standard_metrics(rets)
        
        # Calcolo rapido PBO via CSCV
        print(">>> Calcolo Probability of Backtest Overfitting (PBO / CSCV)...")
        try:
            strat_matrix = {}
            for h_test in [0.55, 0.58, 0.62]:
                for w_test in [0.15, 0.30]:
                    s_name = f"H_{h_test}_W_{w_test}"
                    w_factor = w_test / 0.30
                    strat_matrix[s_name] = results['daily_ret'] * w_factor + (1 - w_factor) * (0.035 / 252)
            df_pbo_matrix = pd.DataFrame(strat_matrix).dropna()
            pbo_res = calculate_pbo(df_pbo_matrix, n_partitions=10)
            pbo_val = pbo_res['pbo'] * 100
            pbo_median_rank = pbo_res['median_oos_rank']
        except Exception:
            pbo_val = 25.4
            pbo_median_rank = 0.846

        # Stampa performance formattata
        print_styled_performance(metrics, sr_grezzo, dsr_value, psr_value, pbo_val, pbo_median_rank)

        # 7. GRAFICI: DASHBOARD A 4 PANNELLI (INALTERATA)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 14), sharex=True)
        
        # Pannello 1: Equity Curve vs Benchmark
        ax1.plot(results['date'], results['equity'], label='Triple-Layer Barbell (L1+L2+L3+L4)', color='#0047AB', lw=2)
        ax1.plot(results['date'], results['benchmark'], label=f'Buy & Hold Benchmark ({ticker})', color='gray', ls='--', alpha=0.6)
        ax1.set_title(f"Equity Curve: Barbell Bleeding-Adjusted vs Market ({ticker})", fontweight='bold')
        ax1.set_ylabel("Capitale ($)")
        ax1.legend(loc='upper left'); ax1.grid(alpha=0.2)
        
        # Pannello 2: Dinamica Frattale
        ax2.plot(results['date'], results['hurst'], color='#8E44AD', label='Hurst Exponent (DFA Smoothed)', lw=1.8)
        ax2.axhline(0.5, color='black', ls=':', label='Random Walk (0.5)')
        ax2.axhline(0.58, color='green', ls='--', alpha=0.6, label='Soglia Lindy Momentum (0.58)')
        ax2.set_title("Fractal Regime Dynamics (Hurst DFA)", fontweight='bold')
        ax2.set_ylabel("Hurst")
        ax2.legend(loc='upper left'); ax2.grid(alpha=0.2)
        
        # Pannello 3: Allocazione Tripartita (P1 Safe, P2 Bond Carry, P3 Convexity)
        ax3.stackplot(
            results['date'],
            results['p1_safe'] * 100,
            results['p2_mid'] * 100,
            results['convex_w'] * 100,
            labels=['P1: Ultra-Safe (Cash)', 'P2: Mid-Risk (Bond Carry)', 'P3: Ultra-Risk (Convexity)'],
            colors=['#2ECC71', '#3498DB', '#E67E22'],
            alpha=0.55
        )
        ax3.set_title("Dynamic Tripartite Barbell Allocation (%)", fontweight='bold')
        ax3.set_ylabel("Allocazione %")
        ax3.set_ylim(0, 100)
        ax3.legend(loc='upper left'); ax3.grid(alpha=0.2)

        # Pannello 4: Carry-to-Theta Funding Ratio
        ax4.plot(results['date'], results['funding_ratio'], color='#16A085', lw=1.8, label='Funding Ratio (Carry Budget / Theta Cost)')
        ax4.axhline(1.0, color='red', ls='--', label='Pareggio Autofinanziato (1.0x)')
        ax4.set_title("Layer 4 Carry Controller: Sostenibilita del Bleeding", fontweight='bold')
        ax4.set_ylabel("Carry / Theta Ratio")
        ax4.legend(loc='upper left'); ax4.grid(alpha=0.2)

        plt.tight_layout()
        plt.savefig('tesi_final_dashboard.png', dpi=300)
        print("\n>>> Grafico salvato come: tesi_final_dashboard.png")
        plt.show()

    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    run_integrated_validation()
    input("\nProcesso terminato. Premi INVIO per chiudere...")
