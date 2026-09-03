# In fondo a core/macro_sentiment.py
if __name__ == "__main__":
    # Test rapido per vedere se l'AI funziona
    engine = SentimentEngine()
    test_news = [
        "Federal Reserve signals potential interest rate hikes to combat inflation",
        "Bitcoin adoption grows as major banks open crypto trading desks",
        "Market volatility increases amid geopolitical tensions in Europe"
    ]
    score = engine.analyze_news(test_news)
    print(f"\nSENTIMENT SCORE: {score:.4f}")
    print(f"CONTEXT: {engine.get_contextual_bias(score)}")
    input("\nPremi INVIO per chiudere...")