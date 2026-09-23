import yfinance as yf
import pandas as pd
import numpy as np

def get_full_alpha_data(ticker="AAPL", period="1y", interval="1h"):
    """
    Scarica e sincronizza i dati orari dell'asset target e dei driver macro.
    Allinea tutti i ticker al fuso orario di New York ('America/New_York').
    """
    print(f">>> Scaricamento Dataset Macro Sincronizzato per {ticker}...")

    def fetch_and_normalize(t_id):
        raw = yf.download(t_id, period=period, interval=interval, progress=False)
        if raw.empty:
            raise ValueError(f"Dati non disponibili per ticker {t_id}")
        
        # Gestione MultiIndex su colonne introdotta nelle nuove versioni di yfinance
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
            
        # Normalizzazione del Timezone ad America/New_York
        if raw.index.tz is not None:
            raw.index = raw.index.tz_convert("America/New_York").tz_localize(None)
        else:
            raw.index = raw.index.tz_localize("UTC").tz_convert("America/New_York").tz_localize(None)
            
        raw.index = raw.index.floor('h')
        raw = raw[~raw.index.duplicated(keep='first')]
        return raw

    # 1. Asset Target
    main_asset = fetch_and_normalize(ticker)
    df = main_asset[['Close', 'High', 'Low']].copy()

    # 2. Driver Macro (Volatilita', Rendimento Treasury 10Y, Oro Safe-Haven)
    macro_tickers = {"^VIX": "VIX", "^TNX": "TNX", "GC=F": "GOLD"}
    
    for t_id, name in macro_tickers.items():
        try:
            m_data = fetch_and_normalize(t_id)
            df = df.join(m_data[['Close']].rename(columns={'Close': name}), how='left')
        except Exception as e:
            print(f"[!] Warning nel recupero di {name} ({t_id}): {e}")
            # Fallback sintetico in caso di blocco API yfinance
            if name not in df.columns:
                df[name] = 20.0 if name == "VIX" else (4.0 if name == "TNX" else 2000.0)

    # 3. Sincronizzazione: forward-fill per colmare micro-discrepanze di chiusura
    final_df = df.ffill(limit=6).dropna()
    final_df.index.name = "Date"
    
    print(f">>> Dataset pronto: {len(final_df)} barre orarie sincronizzate.")
    return final_df
