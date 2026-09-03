import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_standard_metrics(returns):
    """
    Calcola le metriche di performance istituzionali su base giornaliera.
    Necessaria per il report finale.
    """
    if len(returns) < 2: 
        return {
            "Total Return": 0.0, 
            "Sharpe Ratio": 0.0, 
            "Max Drawdown": 0.0, 
            "Ann. Volatility": 0.0
        }
    
    # 1. Rendimento Totale
    total_ret = (returns + 1).prod() - 1
    
    # 2. Annualizzazione (Fattore 252 perché il backtest è a step di 24h)
    ann_factor = 252
    mean_ret = returns.mean()
    std_ret = returns.std()
    
    ann_vol = std_ret * np.sqrt(ann_factor)
    
    # 3. Sharpe Ratio Standard
    sharpe = (mean_ret / std_ret) * np.sqrt(ann_factor) if std_ret > 0 else 0
    
    # 4. Max Drawdown
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    return {
        "Total Return": total_ret,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
        "Ann. Volatility": ann_vol
    }

def calculate_dsr(returns, n_trials=100):
    """
    DEFLATED SHARPE RATIO (DSR) - Implementazione Marcos Lopez de Prado.
    Corregge lo Sharpe per il Multiple Testing Bias e la Non-Normalità.
    Include la protezione per valori di varianza negativi/nulli.
    """
    if len(returns) < 30 or returns.std() == 0: 
        return 0.0, 0.0
    
    # 1. Metriche di base
    T = len(returns)
    sk = returns.skew()
    ku = returns.kurtosis()
    sr = (returns.mean() / returns.std()) * np.sqrt(252)
    
    # 2. Calcolo dello Sharpe Massimo Atteso (per puro caso)
    em_constant = 0.5772156649
    expected_max_sr = np.sqrt(2 * np.log(n_trials)) - \
                     ((em_constant + np.log(np.log(n_trials))) / np.sqrt(2 * np.log(n_trials)))
    
    # 3. Calcolo della deviazione standard dello Sharpe (Lo, 2002)
    # Protezione aggiunta per evitare il RuntimeWarning sulla radice quadrata
    variance_sr = (1 - sk * sr + ((ku - 1) / 4) * sr**2) / (T - 1)
    sigma_sr = np.sqrt(max(variance_sr, 1e-10)) 
    
    # 4. Probabilistic Sharpe Ratio (PSR) rispetto allo Sharpe atteso per caso
    psr_stat = (sr - expected_max_sr) / sigma_sr
    dsr_value = norm.cdf(psr_stat)
        
    return sr, dsr_value

def calculate_information_coefficient(predictions, actuals):
    """Calcola la correlazione tra predizioni L1 e risultati reali."""
    if len(predictions) != len(actuals) or len(predictions) < 2: 
        return 0.0
    return np.corrcoef(predictions, actuals)[0, 1]