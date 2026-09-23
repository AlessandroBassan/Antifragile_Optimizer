import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_standard_metrics(returns, trading_days=252):
    """
    Calcola le metriche di performance quantitative e istituzionali.
    """
    if len(returns) < 2 or returns.std() == 0:
        return {
            "Total Return": 0.0,
            "Ann. Return": 0.0,
            "Ann. Volatility": 0.0,
            "Sharpe Ratio": 0.0,
            "Sortino Ratio": 0.0,
            "Max Drawdown": 0.0,
            "Calmar Ratio": 0.0,
            "Daily Win Rate": 0.0
        }

    # 1. Rendimento Totale Composto
    total_ret = float((returns + 1.0).prod() - 1.0)

    # 2. Rendimento e Volatilità Annualizzati
    mean_ret = float(returns.mean())
    std_ret = float(returns.std())
    ann_ret = (1.0 + mean_ret) ** trading_days - 1.0
    ann_vol = std_ret * np.sqrt(trading_days)

    # 3. Sharpe Ratio Annualizzato
    sharpe = (mean_ret / std_ret) * np.sqrt(trading_days) if std_ret > 0 else 0.0

    # 4. Sortino Ratio (Downside deviation con target 0)
    downside_returns = returns[returns < 0]
    downside_std = float(downside_returns.std()) if len(downside_returns) > 1 else std_ret
    sortino = (mean_ret / downside_std) * np.sqrt(trading_days) if downside_std > 0 else 0.0

    # 5. Drawdown Series & Max Drawdown
    cumulative = (1.0 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = float(drawdown.min())

    # 6. Calmar Ratio
    calmar = abs(ann_ret / max_drawdown) if max_drawdown < 0 else 0.0

    # 7. Win Rate
    win_rate = float((returns > 0).mean())

    return {
        "Total Return": total_ret,
        "Ann. Return": ann_ret,
        "Ann. Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_drawdown,
        "Calmar Ratio": calmar,
        "Daily Win Rate": win_rate
    }

def calculate_dsr(returns, n_trials=100, sr_benchmark=0.0):
    """
    DEFLATED SHARPE RATIO (DSR) - Bailey & Lopez de Prado (2014).
    
    Corregge lo Sharpe Ratio per:
    1. Multiple testing (data mining / selezione del miglior modello tra N trial).
    2. Non-normalità dei rendimenti (skewness e fat tails/kurtosis).
    3. Lunghezza del campione T.
    
    Risolve il warning di varianza negativa convertendo la Fisher Excess Kurtosis
    di Pandas in Pearson Kurtosis K = kurt_fisher + 3.
    """
    if len(returns) < 30 or returns.std() == 0:
        return 0.0, 0.0

    T = len(returns)
    sk = float(returns.skew())
    # In Pandas .kurtosis() restituisce la curtosi in eccesso (normale = 0).
    # La formula originale di Lo/Mertens usa la curtosi di Pearson (normale = 3).
    ku_excess = float(returns.kurtosis())
    
    # 1. Sharpe Ratio NON annualizzato (su periodo base)
    sr_period = float(returns.mean() / returns.std())
    sr_annual = sr_period * np.sqrt(252)

    # 2. Varianza asintotica corretta di Lo (2002) / Mertens (2002)
    # Formula con curtosi in eccesso: (ku_excess + 2) / 4 invece di (ku - 1)/4
    variance_sr = (1.0 - sk * sr_period + ((ku_excess + 2.0) / 4.0) * (sr_period ** 2)) / (T - 1.0)
    sigma_sr = float(np.sqrt(max(variance_sr, 1e-12)))

    # 3. Expected Maximum Sharpe Ratio sotto H0 (selezione tra N trial indipendenti con SR=0)
    # Derivata dalla distribuzione dei valori estremi di Gumbel per N variabili normali standard
    em_constant = 0.5772156649  # Costante di Eulero-Mascheroni
    if n_trials > 1:
        log_n = np.log(n_trials)
        sqrt_2log_n = np.sqrt(2.0 * log_n)
        z_max = sqrt_2log_n - ((em_constant + np.log(log_n)) / sqrt_2log_n)
    else:
        z_max = 0.0

    # Lo Sharpe atteso massimo per puro caso (su scala non annualizzata del periodo)
    expected_max_sr = max(z_max * sigma_sr, sr_benchmark / np.sqrt(252))

    # 4. Calcolo statistica DSR e p-value / CDF
    psr_stat = (sr_period - expected_max_sr) / sigma_sr
    dsr_value = float(norm.cdf(psr_stat))

    return sr_annual, dsr_value

def calculate_psr(returns, benchmark_sr=0.0):
    """
    PROBABILISTIC SHARPE RATIO (PSR) - Lopez de Prado (2012).
    Calcola la probabilità che il vero Sharpe sia superiore a un benchmark prefissato,
    tenendo conto di skewness e kurtosis.
    """
    if len(returns) < 30 or returns.std() == 0:
        return 0.0

    T = len(returns)
    sk = float(returns.skew())
    ku_excess = float(returns.kurtosis())
    
    sr_period = float(returns.mean() / returns.std())
    benchmark_period = benchmark_sr / np.sqrt(252)
    
    variance_sr = (1.0 - sk * sr_period + ((ku_excess + 2.0) / 4.0) * (sr_period ** 2)) / (T - 1.0)
    sigma_sr = float(np.sqrt(max(variance_sr, 1e-12)))
    
    psr_stat = (sr_period - benchmark_period) / sigma_sr
    return float(norm.cdf(psr_stat))

def calculate_information_coefficient(predictions, actuals):
    """
    Calcola l'Information Coefficient (IC) di Pearson e Spearman Rank IC.
    """
    if len(predictions) != len(actuals) or len(predictions) < 5:
        return {"Pearson_IC": 0.0, "Rank_IC": 0.0}

    s_pred = pd.Series(predictions)
    s_act = pd.Series(actuals)
    
    p_ic = float(s_pred.corr(s_act, method='pearson'))
    r_ic = float(s_pred.corr(s_act, method='spearman'))
    
    return {
        "Pearson_IC": 0.0 if np.isnan(p_ic) else p_ic,
        "Rank_IC": 0.0 if np.isnan(r_ic) else r_ic
    }
