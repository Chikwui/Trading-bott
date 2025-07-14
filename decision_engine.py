from ml.signal_predictor import SignalPredictor
from ml.sentiment_analyzer import SentimentAnalyzer
from utils.logger import logger

class DecisionEngine:
    def __init__(self):
        self.predictor = SignalPredictor()
        self.sentiment = SentimentAnalyzer()

    def decide(self, features, news_text=None):
        """
        Combine ML signals and sentiment to produce trade decisions.
        Returns list of dicts: {'symbol', 'signal', 'confidence'}.
        """
        signals_df = self.predictor.predict(features)
        score = self.sentiment.analyze(news_text) if news_text else 0.0
        logger.debug(f"Sentiment score: {score}")
        # Placeholder: simply return ML signals
        return signals_df.to_dict('records')
