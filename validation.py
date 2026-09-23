import itertools
import numpy as np
import pandas as pd
from scipy.stats import rankdata

class PurgedKFoldWithEmbargo:
    """
    PURGED & EMBARGOED K-FOLD CROSS-VALIDATION
    Implementazione del framework di Marcos Lopez de Prado (AFML, Cap. 7).
    
    1. Purging: Rimuove i dati di addestramento il cui orizzonte di predizione (label)
       si sovrappone all'intervallo del Test Set.
    2. Embargo: Rimuove una finestra di osservazioni immediatamente successiva
       al Test Set per neutralizzare la memoria autoregressiva dei residui.
    """
    def __init__(self, n_splits=5, label_horizon=24, pct_embargo=0.01):
        self.n_splits = n_splits
        self.label_horizon = label_horizon
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        embargo_bars = int(n_samples * self.pct_embargo)
        
        # Suddivisione in n_splits blocchi contigui
        fold_size = n_samples // self.n_splits
        
        for fold in range(self.n_splits):
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else n_samples
            
            test_indices = indices[test_start:test_end]
            
            # PURGING: Elimina le barre prima del test set il cui label si estende nel test
            purge_start = max(0, test_start - self.label_horizon)
            
            # EMBARGO: Elimina le barre subito dopo il test set
            embargo_end = min(n_samples, test_end + embargo_bars)
            
            # I sample di training sono quelli prima di purge_start o dopo embargo_end
            train_mask = (indices < purge_start) | (indices >= embargo_end)
            train_indices = indices[train_mask]
            
            yield train_indices, test_indices

def generate_cscv_combinations(n_partitions=10):
    """
    Genera le combinazioni simmetriche C(S, S/2) per la CSCV.
    """
    if n_partitions % 2 != 0:
        raise ValueError("n_partitions deve essere un numero pari (es. 8, 10, 16).")
    half = n_partitions // 2
    all_blocks = list(range(n_partitions))
    combinations = list(itertools.combinations(all_blocks, half))
    return combinations

def calculate_pbo(matrix_returns, n_partitions=10, trading_days=252):
    """
    PROBABILITY OF BACKTEST OVERFITTING (PBO) & CSCV
    Metodologia di Bailey, Borwein, Lopez de Prado & Zhu (2017).
    
    Parametri:
    - matrix_returns: DataFrame o ndarray (T x N), dove T e' la dimensione temporale
                      e N e' il numero di varianti di strategia/configurazioni testate.
    - n_partitions: Numero pari di blocchi contigui temporali S (default 10).
    
    Ritorna un dizionario completo di diagnostica:
    - PBO: Probabilita' che il modello migliore In-Sample sottoperformi la mediana OOS.
    - Log-Odds Distribution: Distribuzione dei logit dei ranghi OOS.
    - Rank Distribution: Ranghi percentili relativi omega_c in [0, 1].
    - Degradation Slope & Correlation: Misura del deterioramento tra Sharpe IS e OOS.
    - Best Model Frequencies: Frequenza con cui ciascun modello e' stato selezionato IS.
    """
    if isinstance(matrix_returns, pd.DataFrame):
        data = matrix_returns.values
        model_names = list(matrix_returns.columns)
    else:
        data = np.asarray(matrix_returns)
        model_names = [f"Model_{i}" for i in range(data.shape[1])]
        
    T, N = data.shape
    if N < 2:
        raise ValueError("PBO richiede almeno 2 varianti di strategia (N >= 2).")
    if T < n_partitions * 5:
        raise ValueError(f"Campione insufficiente: T={T} per {n_partitions} partizioni.")

    # 1. Divisione in S blocchi contigui
    block_size = T // n_partitions
    blocks = [data[i * block_size : (i + 1) * block_size if i < n_partitions - 1 else T, :] 
              for i in range(n_partitions)]
    
    # 2. Generazione delle C(S, S/2) partizioni combinatorie
    half = n_partitions // 2
    combinations = generate_cscv_combinations(n_partitions)
    n_combinations = len(combinations)
    
    oos_ranks = []
    log_odds = []
    is_sharpes = []
    oos_sharpes = []
    best_is_counts = np.zeros(N, dtype=int)
    
    all_block_indices = set(range(n_partitions))

    def compute_sharpe(ret_matrix):
        means = np.mean(ret_matrix, axis=0)
        stds = np.std(ret_matrix, axis=0)
        # Protezione divisione per zero
        stds = np.where(stds == 0, 1e-9, stds)
        return (means / stds) * np.sqrt(trading_days)

    for is_indices in combinations:
        oos_indices = list(all_block_indices - set(is_indices))
        
        # Concatena i blocchi IS e OOS
        is_data = np.vstack([blocks[b] for b in is_indices])
        oos_data = np.vstack([blocks[b] for b in oos_indices])
        
        # Sharpe per ciascuna delle N strategie
        sr_is = compute_sharpe(is_data)
        sr_oos = compute_sharpe(oos_data)
        
        # Modello ottimale In-Sample
        best_is_idx = np.argmax(sr_is)
        best_is_counts[best_is_idx] += 1
        
        # Rango OOS della strategia migliore in IS
        # Usiamo rankdata: rango 1 = peggiore, N = migliore
        ranks = rankdata(sr_oos)
        best_oos_rank = ranks[best_is_idx]
        
        # Rango percentile normalizzato: omega in (0, 1)
        rel_rank = best_oos_rank / (N + 1.0)
        oos_ranks.append(rel_rank)
        
        # Log-odds lambda = ln(omega / (1 - omega))
        rel_rank_clipped = np.clip(rel_rank, 1e-6, 1.0 - 1e-6)
        lambda_c = np.log(rel_rank_clipped / (1.0 - rel_rank_clipped))
        log_odds.append(lambda_c)
        
        is_sharpes.append(sr_is[best_is_idx])
        oos_sharpes.append(sr_oos[best_is_idx])
        
    oos_ranks = np.array(oos_ranks)
    log_odds = np.array(log_odds)
    is_sharpes = np.array(is_sharpes)
    oos_sharpes = np.array(oos_sharpes)
    
    # 3. Calcolo del PBO: frazione con omega <= 0.5 (sotto la mediana)
    pbo_value = float(np.mean(oos_ranks <= 0.5))
    
    # 4. Decadimento di performance: regressione OOS vs IS
    if len(is_sharpes) > 1 and np.std(is_sharpes) > 0:
        degradation_corr = float(np.corrcoef(is_sharpes, oos_sharpes)[0, 1])
        slope, _ = np.polyfit(is_sharpes, oos_sharpes, 1)
    else:
        degradation_corr = 0.0
        slope = 0.0

    return {
        "pbo": pbo_value,
        "is_overfitted": bool(pbo_value > 0.5),
        "median_oos_rank": float(np.median(oos_ranks)),
        "mean_log_odds": float(np.mean(log_odds)),
        "degradation_corr": degradation_corr,
        "degradation_slope": float(slope),
        "n_combinations": n_combinations,
        "rank_distribution": oos_ranks,
        "log_odds_distribution": log_odds,
        "model_selection_frequencies": dict(zip(model_names, (best_is_counts / n_combinations).tolist()))
    }
