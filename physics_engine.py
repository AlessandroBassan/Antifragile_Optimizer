import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import linregress

@njit(fastmath=True)
def _calc_lzc(binary_seq):
    """
    Calcolo Lempel-Ziv Complexity 76 con compilazione JIT.
    """
    n = len(binary_seq)
    if n <= 1:
        return 0.0
    c, l_val, i, k = 1, 1, 0, 1
    while l_val + k <= n:
        if binary_seq[i + k - 1] == binary_seq[l_val + k - 1]:
            k += 1
        else:
            c += 1
            i = 0
            l_val += k
            k = 1
    return (c * np.log2(n)) / n

class PhysicsEngine:
    """
    LAYER 2: Econofisica Quantitativa (Hurst DFA, LZC, Shiller Proxy Z-Score).
    Mappa lo stato frattale e di entropia del mercato.
    """
    def __init__(self, window_size=200, trading_hours_per_day=6.5):
        self.window_size = window_size
        self.trading_hours_per_day = trading_hours_per_day
        self.h_buffer = []

    def reset(self):
        """Azzera il buffer di smoothing per evitare data leakage tra backtest."""
        self.h_buffer.clear()

    def get_hurst_dfa(self, returns):
        """
        Detrended Fluctuation Analysis (DFA) per l'esponente di Hurst H.
        Immune da trend locali e derive di breve termine rispetto a R/S.
        """
        if len(returns) < 30:
            return 0.5
        
        y = np.cumsum(returns - np.mean(returns))
        n = len(y)
        max_scale = max(10, n // 4)
        scales = np.unique(np.logspace(np.log10(8), np.log10(max_scale), 12).astype(np.int32))
        scales = scales[scales >= 4]
        
        if len(scales) < 3:
            return 0.5
            
        fluctuations = np.zeros(len(scales))
        for i, scale in enumerate(scales):
            n_segments = n // scale
            if n_segments == 0:
                continue
            rms = 0.0
            x = np.arange(scale)
            for j in range(n_segments):
                seg = y[j * scale : (j + 1) * scale]
                coeff = np.polyfit(x, seg, 1)
                trend = np.polyval(coeff, x)
                rms += np.sum((seg - trend) ** 2)
            fluctuations[i] = np.sqrt(rms / (n_segments * scale))
            
        # Filtro valori validi > 0 per la regressione log-log
        valid = fluctuations > 0
        if np.sum(valid) < 3:
            return 0.5
            
        h, _, _, _, _ = linregress(np.log10(scales[valid]), np.log10(fluctuations[valid]))
        # Clamp statistico coerente [0.05, 0.95]
        return float(np.clip(h, 0.05, 0.95))

    def get_complexity(self, returns):
        """
        Calcola l'entropia algoritmica di Lempel-Ziv sui rendimenti.
        """
        binary_seq = (returns > 0).astype(np.int8)
        return float(_calc_lzc(binary_seq))

    def compute_state(self, price_series):
        """
        Calcola il vettore di stato fisico completo:
        - Hurst smoothed
        - Lempel-Ziv Complexity
        - Shiller-style Detrended Price Z-Score
        - Volatilità oraria annualizzata
        - Regime di mercato
        """
        prices = price_series.values.flatten()
        if len(prices) < self.window_size:
            return None
            
        returns = np.diff(np.log(prices))
        recent_returns = returns[-(self.window_size - 1):]
        
        # 1. Z-Score di Shiller (Prezzo vs Media e Deviazione Standard storica a 200 periodi)
        ma_200 = np.mean(prices[-self.window_size:])
        std_200 = np.std(prices[-self.window_size:])
        z_score = float((prices[-1] - ma_200) / std_200) if std_200 > 0 else 0.0
        
        # 2. Esponente di Hurst via DFA con smoothing
        h_raw = self.get_hurst_dfa(recent_returns)
        self.h_buffer.append(h_raw)
        if len(self.h_buffer) > 10:
            self.h_buffer.pop(0)
        h_smooth = float(np.mean(self.h_buffer))
        
        # 3. Complessità LZC ed Entropia
        lzc = self.get_complexity(recent_returns)
        
        # 4. Volatilità Annualizzata con ore effettive di scambio
        annualization_factor = np.sqrt(252 * self.trading_hours_per_day)
        vol = float(np.std(recent_returns) * annualization_factor)
        
        return {
            "hurst": h_smooth,
            "hurst_raw": h_raw,
            "lzc": lzc,
            "z_score_shiller": z_score,
            "volatility": vol,
            "regime": self._classify_regime(h_smooth, lzc, z_score)
        }

    def _classify_regime(self, h, lzc, z_score):
        """
        Classifica lo stato del mercato in quattro regimi fisici.
        """
        if abs(z_score) > 2.2:
            return "SHILLER_EXTREME"
        if h > 0.60:
            return "PERSISTENT_TREND" if lzc < 0.45 else "FRAGILE_MOMENTUM"
        if h < 0.42:
            return "MEAN_REVERTING"
        return "GAUSSIAN_NOISE"
