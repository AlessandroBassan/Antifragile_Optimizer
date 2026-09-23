import numpy as np

class StructuralOverlay:
    """
    LAYER 4: Active Wealth Controller (CFO Layer).
    Implementa il Feedback Loop tra rendimento obbligazionario e costo della protezione (Bleeding).
    Garantisce che il Barbell sia strutturalmente auto-finanziato (Self-Financing).
    """
    def __init__(self, inflation_rate=0.025, commission_rate=0.0010, bond_weight=0.15):
        self.inflation_rate = inflation_rate
        self.commission_rate = commission_rate
        self.bond_weight = bond_weight
        self.last_convex_w = 0.0

    def reset(self):
        """Azzera la memoria dell'allocazione precedente per isolare i backtest."""
        self.last_convex_w = 0.0

    def apply_active_overlay(self, l3_decision, tnx_yield, expected_alpha):
        """
        Applica il vincolo di bilancio rigido: il costo del Theta non puo' superare
        il carry prodotto dal comparto obbligazionario.
        """
        target_convex_w = float(l3_decision['convex_weight'])
        theta_cost = float(l3_decision['theta_decay'])

        # 1. CALCOLO BUDGET DI CARRY (Pillar 2: Obbligazioni a 5-7y)
        # Il rendimento ^TNX annualizzato viene convertito in budget di carry giornaliero
        daily_carry_budget = (self.bond_weight * (tnx_yield / 100.0)) / 365.0

        # 2. FEEDBACK LOOP DI CARRY (Hard Budgeting Anti-Bleeding)
        final_convex_w = target_convex_w
        budget_status = "FULLY_FUNDED"

        if theta_cost > daily_carry_budget and target_convex_w > 0:
            scaling_factor = daily_carry_budget / theta_cost
            final_convex_w = target_convex_w * scaling_factor
            budget_status = f"SCALED_DOWN ({scaling_factor:.2f}x)"

        # 3. PROFITABILITY HURDLE (Isteresi sulle Commissioni)
        # Evita transazioni se i costi di turnover superano la meta' dell'alpha atteso
        w_diff = abs(final_convex_w - self.last_convex_w)
        fee_cost = w_diff * self.commission_rate

        if expected_alpha < (fee_cost * 2.0) and self.last_convex_w > 0:
            final_convex_w = self.last_convex_w
            execution_status = "STAY_PUT (High Fees)"
        else:
            execution_status = "REBALANCED"
            self.last_convex_w = final_convex_w

        # 4. COMPOSIZIONE FINALE DEL BILANCIERE TRIPARTITO
        # Pillar 1 (Cash/T-Bills) + Pillar 2 (Bonds) + Pillar 3 (Hyper-Convexity) = 100%
        p2_mid = self.bond_weight
        p3_ultra = float(np.clip(final_convex_w, 0.0, 1.0 - p2_mid))
        p1_safe = float(1.0 - p2_mid - p3_ultra)

        # Rapporto di autofinanziamento (Carry / Theta)
        funding_ratio = float(daily_carry_budget / (theta_cost if theta_cost > 0 else 1e-9))

        # Calcolo del Net Carry Annuo Reale (al netto di inflazione e theta effettivo)
        scaled_annual_theta = (theta_cost * (p3_ultra / (target_convex_w if target_convex_w > 0 else 1.0))) * 365.0
        net_carry_annual = (p2_mid * (tnx_yield / 100.0)) - scaled_annual_theta - self.inflation_rate

        return {
            "p1_safe": p1_safe,
            "p2_mid": p2_mid,
            "p3_ultra": p3_ultra,
            "funding_ratio": funding_ratio,
            "net_carry_annual": net_carry_annual,
            "daily_carry_budget": daily_carry_budget,
            "budget_status": budget_status,
            "execution_status": execution_status
        }
