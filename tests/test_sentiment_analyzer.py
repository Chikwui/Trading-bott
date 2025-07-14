from ml.sentiment_analyzer import SentimentAnalyzer

def test_sentiment_default():
    sa = SentimentAnalyzer()
    result = sa.analyze("Some market news text.")
    assert isinstance(result, float)
    assert result == 0.0  # placeholder implementation returns 0.0
