from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch

class SentimentEngine:
    """
    ENGINE LLM (Layer 2 della Tesi)
    Analisi del sentiment sui verbali o news finanziarie.
    """
    def __init__(self):
        print("Inizializzazione FinBERT (LLM Layer 2)...")
        # Carichiamo il modello specifico per il linguaggio finanziario
        model_name = "yiyanghkust/finbert-tone"
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)

    def analyze_news(self, headlines):
        """
        Riceve una lista di titoli e restituisce un punteggio da -1 a 1.
        """
        if not headlines: return 0.0
        
        results = self.nlp(headlines)
        scores = []
        for res in results:
            # FinBERT-tone usa: Neutral (0), Positive (1), Negative (2)
            label = res['label']
            score = res['score']
            if label == 'Positive': scores.append(score)
            elif label == 'Negative': scores.append(-score)
            else: scores.append(0.0)
            
        return sum(scores) / len(scores)

    def get_contextual_bias(self, sentiment_score):
        """
        Converte il sentiment in un bias per l'ottimizzatore stocastico.
        """
        if sentiment_score > 0.3: return "BULLISH_SENTIMENT"
        if sentiment_score < -0.3: return "BEARISH_SENTIMENT"
        return "NEUTRAL_CONTEXT"