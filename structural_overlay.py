import numpy as np

class StructuralOverlay:
    """
    LAYER 4: Active Wealth Controller (CFO Layer).
    Implementa il Feedback Loop tra rendimento obbligazionario e costo della protezione.
    """
    def __init__(self, inflation_rate=0.025, commission_rate=0.001):
        self.inflation_rate = inflation_rate
        self.commission_rate = commission_rate
        self.last_convex_w = 0.0

    def apply_active_overlay(self, l3_decision, tnx_yield, expected_alpha):
        """
        Applica il budget rigido: il costo del Theta non può superare il Carry prodotto.
        """
        target_convex_w = l3_decision['convex_weight']
        theta_cost = l3_decision['theta_decay']
        
        # 1. CALCOLO BUDGET (Pillar 2: 15% del capitale in Bond 5-7y)
        # Usiamo il rendimento reale ^TNX come proxy del carry
        mid_risk_w = 0.15
        daily_carry_budget = (mid_risk_w * (tnx_yield / 100)) / 365
        
        # 2. FEEDBACK LOOP (Hard Budgeting)
        # Se il costo giornaliero delle opzioni > budget cedole, scaliamo la size
        final_convex_w = target_convex_w
        budget_status = "FULLY_FUNDED"
        
        if theta_cost > daily_carry_budget and target_convex_w > 0:
            scaling_factor = daily_carry_budget / theta_cost
            final_convex_w = target_convex_w * scaling_factor
            budget_status = f"SCALED_DOWN ({scaling_factor:.2f}x)"

        # 3. PROFITABILITY HURDLE (Hysteresis)
        # Evitiamo di cambiare allocazione se le commissioni mangiano troppo profitto atteso
        w_diff = abs(final_convex_w - self.last_convex_w)
        fee_cost = w_diff * self.commission_rate
        
        if expected_alpha < (fee_cost * 2) and self.last_convex_w > 0:
            final_convex_w = self.last_convex_w
            execution_status = "STAY_PUT (High Fees)"
        else:
            execution_status = "REBALANCED"
            self.last_convex_w = final_convex_w

        # 4. COMPOSIZIONE FINALE DEL BILANCIERE
        # Pillar 1 + Pillar 2 + Pillar 3 = 100%
        p2_mid = 0.15
        p3_ultra = final_convex_w
        p1_safe = 1.0 - p2_mid - p3_ultra

        # Calcolo Net Carry Annuo (Rendimento atteso stando fermi)
        annual_theta = (theta_cost * (final_convex_w / (target_convex_w if target_convex_w > 0 else 1e-9))) * 365
        net_carry = (p2_mid * (tnx_yield/100)) - annual_theta - self.inflation_rate

        return {
            "p1_safe": p1_safe,
            "p2_mid": p2_mid,
            "p3_ultra": p3_ultra,
            "funding_ratio": daily_carry_budget / (theta_cost if theta_cost > 0 else 1e-9),
            "net_carry_annual": net_carry,
            "budget_status": budget_status,
            "execution_status": execution_status
        }