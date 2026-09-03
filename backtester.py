import pandas as pd
import numpy as np
from core.structural_overlay import StructuralOverlay

class BacktestEngine:
    """
    ENGINE DI BACKTESTING INTEGRATO (L1+L2+L3+L4).
    Simula la crescita del capitale basata sulla tripartizione strutturale.
    """
    def __init__(self, initial_capital=100000, base_tc=0.0010):
        self.initial_capital = initial_capital
        self.base_tc = base_tc
        self.overlay_manager = StructuralOverlay(commission_rate=base_tc)
        self.history = []

    def run(self, data, alpha_e, physics_e, opt_e):
        print(f"\n>>> Avvio Backtest Integrato (Friction + L4 CFO Logic)...")
        current_cap = self.initial_capital
        bench_cap = self.initial_capital
        
        # Step di 24 ore (Ribilanciamento giornaliero)
        for i in range(250, len(data) - 24, 24):
            window = data.iloc[i-200 : i]
            next_day = data.iloc[i : i+24]
            
            # 1. SEGNALI LAYER 1-2-3
            state = physics_e.compute_state(window[['Close']])
            if not state: continue
            
            alpha_signal = alpha_e.get_alpha_signal(window)
            l3_decision = opt_e.solve_barbell(state, window[['Close']])
            
            # 2. INTEGRAZIONE LAYER 4 (Budgeting & Feedback)
            # Recuperiamo il tasso TNX del momento
            tnx_now = data['TNX'].iloc[i]
            l4_decision = self.overlay_manager.apply_active_overlay(l3_decision, tnx_now, alpha_signal)
            
            # Pesi finali dai 3 pilastri
            w_p1 = l4_decision['p1_safe']
            w_p2 = l4_decision['p2_mid']
            w_p3 = l4_decision['p3_ultra']

            # 3. RENDIMENTI REALI DEI PILASTRI
            # Pillar 1 (Cash): 0.01% daily + micro-volatilità
            r_p1 = 0.00005 + np.random.normal(0, 0.0001)
            
            # Pillar 2 (Bonds 5-7y): Cedola reale dal tasso decennale
            r_p2 = (tnx_now / 100) / 365
            
            # Pillar 3 (Convexity): Rendimento dell'asset target
            asset_ret = (next_day['Close'].iloc[-1] / next_day['Close'].iloc[0]) - 1
            r_p3 = asset_ret
            
            # 4. CALCOLO ATTRITI (Slippage dinamico su Pillar 3)
            penalty = 1 + state['lzc']
            turnover_cost = abs(w_p3 - self.overlay_manager.last_convex_w) * (self.base_tc * penalty)
            
            # Rendimento Totale Netto
            port_ret = (w_p1 * r_p1) + (w_p2 * r_p2) + (w_p3 * r_p3) - turnover_cost
            
            current_cap *= (1 + port_ret)
            bench_cap *= (1 + asset_ret)
            
            self.history.append({
                'date': data.index[i],
                'equity': current_cap,
                'benchmark': bench_cap,
                'hurst': state['hurst'],
                'convex_w': w_p3,
                'daily_ret': port_ret,
                'funding_ratio': l4_decision['funding_ratio']
            })
            
        return pd.DataFrame(self.history)