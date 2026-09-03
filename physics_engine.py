import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import linregress

@njit(fastmath=True)
def _calc_lzc(binary_seq):
    n = len(binary_seq)
    if n == 0: return 0.0
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
    def __init__(self, window_size=200):
        self.window_size = window_size
        self.h_buffer = [] # Buffer per lo smoothing di Hurst

    def get_hurst_dfa(self, returns):
        y = np.cumsum(returns - np.mean(returns))
        n = len(y)
        scales = np.unique(np.logspace(np.log10(10), np.log10(n//4), 15).astype(np.int32))
        fluctuations = np.zeros(len(scales))
        for i, scale in enumerate(scales):
            n_segments = n // scale
            rms = 0.0
            for j in range(n_segments):
                seg = y[j*scale : (j+1)*scale]
                x = np.arange(scale)
                coeff = np.polyfit(x, seg, 1)
                trend = np.polyval(coeff, x)
                rms += np.sum((seg - trend)**2)
            fluctuations[i] = np.sqrt(rms / (n_segments * scale))
        h, _, _, _, _ = linregress(np.log10(scales), np.log10(fluctuations))
        return h

    def get_complexity(self, returns):
        binary_seq = (returns > 0).astype(np.int8)
        return _calc_lzc(binary_seq)

    def compute_state(self, price_series):
        prices = price_series.values.flatten()
        returns = np.diff(np.log(prices))
        if len(prices) < 200: return None
        
        # Z-Score Shiller
        ma_200 = np.mean(prices[-200:])
        std_200 = np.std(prices[-200:])
        z_score = (prices[-1] - ma_200) / std_200 if std_200 != 0 else 0
        
        # Hurst con Smoothing (Filtro Passa-Basso)
        recent_returns = returns[-(self.window_size-1):]
        h_raw = self.get_hurst_dfa(recent_returns)
        
        self.h_buffer.append(h_raw)
        if len(self.h_buffer) > 10: self.h_buffer.pop(0) # Media mobile a 10 ore
        h_smooth = np.mean(self.h_buffer)
        
        lzc = self.get_complexity(recent_returns)
        vol = np.std(recent_returns) * np.sqrt(252 * 24)
        
        return {
            "hurst": h_smooth, 
            "lzc": lzc, 
            "z_score_shiller": z_score, 
            "volatility": vol,
            "regime": self._classify_regime(h_smooth, lzc, z_score)
        }

    def _classify_regime(self, h, lzc, z_score):
        if abs(z_score) > 2.2: return "SHILLER_EXTREME"
        if h > 0.65: return "PERSISTENT_TREND" if lzc < 0.40 else "FRAGILE_MOMENTUM"
        return "GAUSSIAN_NOISE"