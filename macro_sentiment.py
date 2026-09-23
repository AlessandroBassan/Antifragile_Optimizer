class SentimentEngine:
    """
    ENGINE LLM (Layer 2 della Tesi): Analisi del sentiment finanziario via FinBERT.
    Fornisce bias contestuale macro-economico per l'allocatore di rischio.
    """
    _instance = None
    _pipeline = None

    def __init__(self, use_model=True):
        self.use_model = use_model
        if use_model and SentimentEngine._pipeline is None:
            try:
                print(">>> Inizializzazione FinBERT (yiyanghkust/finbert-tone)...")
                from transformers import BertTokenizer, BertForSequenceClassification, pipeline
                model_name = "yiyanghkust/finbert-tone"
                tokenizer = BertTokenizer.from_pretrained(model_name)
                model = BertForSequenceClassification.from_pretrained(model_name)
                SentimentEngine._pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
                print(">>> FinBERT caricato con successo.")
            except Exception as e:
                print(f"[!] Warning FinBERT offline/fallback: {e}")
                SentimentEngine._pipeline = None

    def analyze_news(self, headlines):
        """
        Riceve una lista di titoli/comunicati e restituisce un punteggio aggregato in [-1, +1].
        """
        if not headlines:
            return 0.0

        if SentimentEngine._pipeline is not None:
            try:
                results = SentimentEngine._pipeline(headlines)
                scores = []
                for res in results:
                    label = res['label']
                    score = float(res['score'])
                    if label == 'Positive':
                        scores.append(score)
                    elif label == 'Negative':
                        scores.append(-score)
                    else:
                        scores.append(0.0)
                return float(sum(scores) / len(scores))
            except Exception as e:
                print(f"[!] Errore inferenza FinBERT: {e}")

        # Fallback basato su parole chiave finanziarie (se offline o per test veloci)
        pos_words = {'resilience', 'growth', 'easing', 'recovery', 'rally', 'surge', 'bullish', 'strong'}
        neg_words = {'inflation', 'hikes', 'headwinds', 'crisis', 'tensions', 'recession', 'bearish', 'risk'}
        
        scores = []
        for h in headlines:
            words = set(h.lower().split())
            p_count = len(words & pos_words)
            n_count = len(words & neg_words)
            if p_count + n_count > 0:
                scores.append((p_count - n_count) / (p_count + n_count))
            else:
                scores.append(0.0)
                
        return float(sum(scores) / len(scores)) if scores else 0.0

    def get_contextual_bias(self, sentiment_score):
        """
        Mappa il punteggio continuo in un bias qualitativo per l'ottimizzatore.
        """
        if sentiment_score > 0.25:
            return "BULLISH"
        elif sentiment_score < -0.25:
            return "BEARISH"
        return "NEUTRAL"
