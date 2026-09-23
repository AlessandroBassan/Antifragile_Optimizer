import numpy as np
import pandas as pd
from core.structural_overlay import StructuralOverlay

class BacktestEngine:
    """
    ENGINE DI BACKTESTING QUANTITATIVO INTEGRATO (L1 + L2 + L3 + L4).
    Simula la crescita patrimoniale con:
    1. Reale detrazione del Theta Bleeding dal P&L convesso.
    2. Modellazione convessa asimmetrica (Delta-Gamma) per Pillar 3.
    3. Flusso cedolare attivo del Pillar 2 e liquidita' Pillar 1.
    4. Attriti di turnover proporzionali alla complessita' entropica (LZC).
    """
    def __init__(self, initial_capital=100000.0, base_tc=0.0010, delta=0.5, gamma=0.5):
        self.initial_capital = initial_capital
        self.base_tc = base_tc
        self.delta = delta
        self.gamma = gamma
        self.overlay_manager = StructuralOverlay(commission_rate=base_tc)

    def run(self, data, alpha_e, physics_e, opt_e, sentiment_e=None):
        """
        Esegue il walk-forward a step di 24 ore (ribilanciamento giornaliero).
        """
        print(f"\n>>> Avvio Walk-Forward Backtest (Bleeding Dedotto + Delta-Gamma Payoff)...")
        physics_e.reset()
        self.overlay_manager.reset()

        current_cap = self.initial_capital
        bench_cap = self.initial_capital
        history = []

        step_hours = 24
        window_size = 200

        for i in range(window_size + 50, len(data) - step_hours, step_hours):
            window = data.iloc[i - window_size : i]
            next_day = data.iloc[i : i + step_hours]

            # 1. LAYER 2: STATO FISICO (Hurst, LZC, Shiller Z-score, Volatilita')
            state = physics_e.compute_state(window[['Close']])
            if not state:
                continue

            # 2. LAYER 1: ALPHA ML (XGBoost con Purging/Embargo)
            alpha_signal = alpha_e.get_alpha_signal(window)

            # 3. LAYER 3: OTTIMIZZATORE STOCASTICO (Barbell Allocation & Theta Cost)
            l3_decision = opt_e.solve_barbell(state, window[['Close']])

            # Filtro opzionale sentiment
            if sentiment_e is not None:
                headline_proxy = [f"Market volatility at {state['volatility']:.2f}", "Macro policy outlook"]
                score = sentiment_e.analyze_news(headline_proxy)
                bias = sentiment_e.get_contextual_bias(score)
                if bias == "BEARISH" and state['hurst'] < 0.52:
                    l3_decision['convex_weight'] = 0.0
                    l3_decision['safe_weight'] = 1.0
                    l3_decision['status'] = "BEARISH_SENTIMENT_OVERRIDE"

            # 4. LAYER 4: ACTIVE STRUCTURAL OVERLAY (CFO Feedback Loop)
            tnx_now = float(data['TNX'].iloc[i]) if 'TNX' in data.columns else 4.0
            l4_decision = self.overlay_manager.apply_active_overlay(l3_decision, tnx_now, alpha_signal)

            w_p1 = l4_decision['p1_safe']
            w_p2 = l4_decision['p2_mid']
            w_p3 = l4_decision['p3_ultra']

            # 5. RENDIMENTI REALI DEI 3 PILASTRI
            # Pillar 1 (Cash / T-Bills 3M): tasso risk-free privo di duration (approx 3.5% annuo)
            r_p1 = (0.035 / 252.0) + np.random.normal(0.0, 0.00002)

            # Pillar 2 (Bonds 5-7Y): cedola giornaliera dal tasso TNX decennale
            r_p2 = (tnx_now / 100.0) / 252.0

            # Pillar 3 (Hyper-Convexity): Payoff convesso asimmetrico (Delta-Gamma)
            asset_ret = float((next_day['Close'].iloc[-1] / next_day['Close'].iloc[0]) - 1.0)
            convex_payoff = (self.delta * asset_ret) + (0.5 * self.gamma * (asset_ret ** 2))
            
            # DETRAZIONE REALE DEL THETA BLEEDING:
            # Il costo del theta calcolato in L3 viene effettivamente detratto dal rendimento di P3
            actual_theta = l3_decision['theta_decay'] * (w_p3 / (l3_decision['convex_weight'] if l3_decision['convex_weight'] > 0 else 1.0))
            r_p3_net = convex_payoff - actual_theta

            # 6. ATTRITI E TURNOVER PENALIZZATI DA LZC (Complessita' algoritmica di mercato)
            lzc_friction = 1.0 + state['lzc']
            turnover_cost = abs(w_p3 - self.overlay_manager.last_convex_w) * (self.base_tc * lzc_friction)

            # Rendimento Complessivo del Portafoglio Netto
            port_ret = (w_p1 * r_p1) + (w_p2 * r_p2) + (w_p3 * r_p3_net) - turnover_cost

            current_cap *= (1.0 + port_ret)
            bench_cap *= (1.0 + asset_ret)

            history.append({
                'date': data.index[i],
                'equity': current_cap,
                'benchmark': bench_cap,
                'hurst': state['hurst'],
                'lzc': state['lzc'],
                'volatility': state['volatility'],
                'p1_safe': w_p1,
                'p2_mid': w_p2,
                'convex_w': w_p3,
                'daily_ret': port_ret,
                'asset_ret': asset_ret,
                'r_p3_net': r_p3_net,
                'theta_decay': actual_theta,
                'funding_ratio': l4_decision['funding_ratio'],
                'net_carry_annual': l4_decision['net_carry_annual'],
                'budget_status': l4_decision['budget_status'],
                'execution_status': l4_decision['execution_status']
            })

        df_history = pd.DataFrame(history)
        print(f">>> Backtest concluso: {len(df_history)} giornate simulate.")
        return df_history
