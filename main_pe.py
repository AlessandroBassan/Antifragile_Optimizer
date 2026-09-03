import sys
import os
import pandas as pd

# FIX: Aggiunge la cartella corrente al path per trovare i moduli
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    from utils.data_loader import get_market_data
    from core.physics_engine import PhysicsEngine
    print("Moduli caricati correttamente.")
except ImportError as e:
    print(f"Errore di importazione: {e}")
    input("Premi INVIO per chiudere...")
    sys.exit()

def run_analysis():
    try:
        # 1. Scarico dati (BTC è ottimo per testare la volatilità)
        print("Scaricamento dati reali (BTC-USD)...")
        data = get_market_data(ticker="BTC-USD", period="1mo", interval="1h")
        
        # 2. Inizializzo l'Engine con finestra di 200 ore
        engine = PhysicsEngine(window_size=200)
        
        # 3. Calcolo lo Stato Fisico
        print("Calcolo Hurst, LZC e Regimi in corso...")
        state = engine.compute_state(data)
        
        if state:
            print("\n" + "="*30)
            print("   QUANT PHYSICS VECTOR")
            print("="*30)
            print(f"ASSET:        BTC-USD")
            print(f"HURST (DFA):  {state['hurst']:.4f}")
            print(f"COMPLEXITY:   {state['lzc']:.4f}")
            print(f"VOLATILITY:   {state['volatility']:.4f}")
            print(f"REGIME:       {state['regime']}")
            print("="*30)
        else:
            print("Errore: Dati insufficienti per l'analisi.")

    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")

    # Questo impedisce alla finestra di chiudersi
    print("\nAnalisi completata.")
    input("Premi INVIO per uscire...")

if __name__ == "__main__":
    run_analysis()