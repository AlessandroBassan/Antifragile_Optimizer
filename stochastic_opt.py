import numpy as np
from numba import njit

@njit(fastmath=True)
def _simulate_fractal_paths_fast(current_price, persistence, hourly_vol, n_scenarios, horizon):
    """
    Simulazione Monte Carlo ad alta velocita' con Numba.
    Genera n_scenarios cammini futuri di lunghezza horizon ore.
    """
    paths = np.empty((n_scenarios, horizon))
    for i in range(n_scenarios):
        price = current_price
        p0 = current_price
        for t in range(horizon):
            drift = persistence * (price - p0) / (p0 if p0 != 0 else 1.0)
            shock = np.random.normal(0.0, hourly_vol)
            price = price * (1.0 + drift + shock)
            paths[i, t] = price
    return paths

class StochasticOptimizer:
    """
    LAYER 3: Antifragile Allocation & Stochastic Programming.
    Risolve il trade-off tra Lindy Momentum, Shiller Mean-Reversion e Theta Decay.
    """
    def __init__(self, n_scenarios=2500, horizon=24, trading_hours=6.5):
        self.n_scenarios = n_scenarios
        self.horizon = horizon
        self.trading_hours = trading_hours

    def calculate_theta_decay(self, vol, weight_convex):
        """
        Calcola il costo giornaliero (Theta decay) di una posizione convessa protetta.
        Approssimazione no-arbitrage Black-Scholes ATM: Theta approx 0.5 * sigma^2 * w / 365.
        """
        if weight_convex <= 0 or vol <= 0:
            return 0.0
        return float(0.5 * (vol ** 2) * weight_convex / 365.0)

    def generate_fractal_scenarios(self, current_price, h_exp, vol, lzc):
        """
        Genera scenari con deriva persistente/antipersistente proporzionale a (H - 0.5).
        """
        persistence = float((h_exp - 0.5) * 0.1)
        # Volatilita' oraria
        hourly_vol = float(vol / np.sqrt(252.0 * self.trading_hours)) if vol > 0 else 0.001
        return _simulate_fractal_paths_fast(current_price, persistence, hourly_vol, self.n_scenarios, self.horizon)

    def solve_barbell(self, state, price_data):
        """
        Determina l'allocazione ottima (Safe vs Convex) integrando i segnali fisici.
        """
        h = state['hurst']
        lzc = state['lzc']
        vol = state['volatility']
        z_shiller = state['z_score_shiller']
        curr_price = float(price_data.iloc[-1].values[0])

        # 1. Generazione Scenari Frattali
        scenarios = self.generate_fractal_scenarios(curr_price, h, vol, lzc)
        expected_alpha = float(np.mean(scenarios[:, -1]) / curr_price - 1.0)

        # 2. Regole Decisionali
        # CASO 1: SHILLER PROTECTION (Prezzo statisticamente deviato oltre 2 deviazioni standard)
        if abs(z_shiller) > 2.0:
            safe_w, convex_w = 1.0, 0.0
            rule = "SHILLER_PROTECTION (Active)"

        # CASO 2: LINDY MOMENTUM (Forte persistenza H > 0.58 con bassa entropia LZC)
        elif h > 0.58 and lzc < 0.45 and expected_alpha > 0.0002:
            safe_w, convex_w = 0.70, 0.30
            rule = "LINDY_MOMENTUM (Active)"

        # CASO 3: REGIME NEUTRALE / RUMORE GAUSSIANO
        else:
            safe_w, convex_w = 0.95, 0.05
            rule = "NEUTRAL (Standby)"

        # 3. GATING DI SOSTENIBILITA': ALPHA vs THETA BLEEDING
        theta_cost = self.calculate_theta_decay(vol, convex_w)
        status = "SUSTAINABLE"
        if expected_alpha < theta_cost and convex_w > 0.05:
            safe_w, convex_w = 1.0, 0.0
            status = "THETA_DE-RISKING"

        return {
            "safe_weight": safe_w,
            "convex_weight": convex_w,
            "rule_applied": rule,
            "status": status,
            "theta_decay": theta_cost,
            "expected_alpha": expected_alpha
        }
