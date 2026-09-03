import numpy as np

class StochasticOptimizer:
    """
    LAYER 3: Antifragile Allocation & Stochastic Programming.
    Risolve il conflitto Lindy vs Shiller e gestisce il Theta Risk.
    """
    def __init__(self, n_scenarios=5000, horizon=24):
        self.n_scenarios = n_scenarios
        self.horizon = horizon

    def calculate_theta_decay(self, vol, weight_convex):
        """Calcola il costo giornaliero (Theta) della protezione convessa."""
        return 0.5 * (vol**2) * weight_convex / 365

    def generate_fractal_scenarios(self, current_price, h_exp, vol, lzc):
        """Genera cammini futuri Hurst-biased per stimare l'Alpha attesa."""
        persistence = (h_exp - 0.5) * 0.1 
        paths = np.zeros((self.n_scenarios, self.horizon))
        # Volatilità oraria corretta
        hourly_vol = vol / np.sqrt(252 * 6.5) 

        for i in range(self.n_scenarios):
            shocks = np.random.normal(0, hourly_vol, self.horizon)
            path = [current_price]
            for s in shocks:
                drift = persistence * (path[-1] - path[0]) / path[0]
                next_val = path[-1] * (1 + drift + s)
                path.append(next_val)
            paths[i, :] = path[1:]
        return paths

    def solve_barbell(self, state, price_data):
        """Determina l'allocazione ottima integrando i segnali dei Layer."""
        h, lzc, vol = state['hurst'], state['lzc'], state['volatility']
        z_shiller = state['z_score_shiller']
        curr_price = float(price_data.iloc[-1].values[0])

        # 1. Generazione Scenari per Alpha attesa
        scenarios = self.generate_fractal_scenarios(curr_price, h, vol, lzc)
        expected_alpha = np.mean(scenarios[:, -1]) / curr_price - 1
        
        # --- LOGICA DECISIONALE IBRIDA (Lindy vs Shiller) ---
        
        # CASO 1: SHILLER PROTECTION (Prezzo statisticamente troppo alto)
        if abs(z_shiller) > 2.0:
            safe_w, convex_w = 1.0, 0.0
            rule = "SHILLER_PROTECTION (Active)"
            
        # CASO 2: LINDY MOMENTUM (Trend persistente rilevato)
        elif h > 0.58 and lzc < 0.45 and expected_alpha > 0.0003:
            safe_w, convex_w = 0.70, 0.30 
            rule = "LINDY_MOMENTUM (Active)"
            
        # CASO 3: NEUTRAL (Gaussian Noise / Incertezza)
        else:
            # Manteniamo un 5% per mostrare attività nel grafico
            safe_w, convex_w = 0.95, 0.05
            rule = "NEUTRAL (Lindy/Shiller Standby)"

        # --- GESTIONE THETA RISK ---
        theta_cost = self.calculate_theta_decay(vol, convex_w)
        status = "SUSTAINABLE"
        if expected_alpha < theta_cost and convex_w > 0.05:
            safe_w, convex_w = 1.0, 0.0
            status = "THETA_DE-RISKING"

        return {
            "safe_weight": safe_w, "convex_weight": convex_w,
            "rule_applied": rule, "status": status,
            "theta_decay": theta_cost, "expected_alpha": expected_alpha
        }