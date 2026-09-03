import yfinance as yf
import pandas as pd
import numpy as np

def get_full_alpha_data(ticker="AAPL", period="1y", interval="1h"):
    print(f">>> Scaricamento Dataset Macro per {ticker}...")
    
    # 1. Scarichiamo l'asset principale
    main_asset = yf.download(ticker, period=period, interval=interval, progress=False)
    if main_asset.empty:
        raise Exception(f"Errore: Nessun dato per {ticker}")
    
    if isinstance(main_asset.columns, pd.MultiIndex):
        main_asset.columns = main_asset.columns.get_level_values(0)
    
    # FIX: Pandas ora vuole 'h' minuscolo per l'ora
    main_asset.index = main_asset.index.tz_localize(None).floor('h')
    
    df = main_asset[['Close', 'High', 'Low']].copy()
    df = df[~df.index.duplicated(keep='first')]

    # 2. Driver Macro
    macro_tickers = {"^VIX": "VIX", "^TNX": "TNX", "GC=F": "GOLD"}
    
    for t_id, name in macro_tickers.items():
        print(f"Sincronizzazione driver: {name}...")
        m_data = yf.download(t_id, period=period, interval=interval, progress=False)
        
        if not m_data.empty:
            if isinstance(m_data.columns, pd.MultiIndex):
                m_data.columns = m_data.columns.get_level_values(0)
            
            # FIX: 'h' minuscolo qui anche
            m_data.index = m_data.index.tz_localize(None).floor('h')
            m_data = m_data[~m_data.index.duplicated(keep='first')]
            
            # Uniamo i dati macro alle ore esatte in cui l'asset principale scambia
            df = df.join(m_data['Close'], rsuffix=f'_{name}')
            df = df.rename(columns={f'Close_{name}': name})

    # 3. Pulizia e Allineamento
    final_df = df.ffill().dropna()
    print(f">>> Dataset pronto: {len(final_df)} righe sincronizzate.")
    return final_df