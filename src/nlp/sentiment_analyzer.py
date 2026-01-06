import os
import numpy as np

# sklearn
import joblib

# roberta fallback
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax


class SentimentAnalyzer:
    """
    Priority:
    1) If data/models/sentiment140_tfidf_lr.joblib exists -> use it (fast, supervised on Sentiment140)
    2) Else -> fallback to RoBERTa twitter model
    """

    def __init__(self, model_path="data/models/sentiment140_tfidf_lr.joblib"):
        self.model_path = model_path
        self.labels = ["negative", "neutral", "positive"]

        self.mode = "sklearn" if os.path.exists(model_path) else "roberta"

        if self.mode == "sklearn":
            self.model = joblib.load(model_path)
        else:
            self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.roberta = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.roberta.eval()

    def analyze_sentiment(self, text, neutral_threshold=0.55):
        """
        Returns:
        - label: negative/neutral/positive
        - confidence: max proba
        - sentiment_score: pos - neg
        - scores dict
        """
        if not isinstance(text, str) or not text.strip():
            return {
                "label": "neutral",
                "confidence": 0.0,
                "sentiment_score": 0.0,
                "scores": {"negative": 0.0, "neutral": 1.0, "positive": 0.0},
            }

        if self.mode == "sklearn":
            # Binary model -> negative/positive, we create neutral if confidence low
            proba = self.model.predict_proba([text])[0]  # [neg, pos]
            neg_p, pos_p = float(proba[0]), float(proba[1])
            conf = max(neg_p, pos_p)

            if conf < neutral_threshold:
                label = "neutral"
                scores = {"negative": neg_p, "neutral": 1.0 - conf, "positive": pos_p}
            else:
                label = "positive" if pos_p >= neg_p else "negative"
                scores = {"negative": neg_p, "neutral": 0.0, "positive": pos_p}

            return {
                "label": label,
                "confidence": float(conf),
                "sentiment_score": float(pos_p - neg_p),
                "scores": scores,
            }

        # fallback roberta
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = self.roberta(**inputs)
                logits = outputs.logits[0]

            probs = softmax(logits, dim=-1).cpu().numpy()
            idx = int(np.argmax(probs))
            label = self.labels[idx]
            confidence = float(np.max(probs))
            sentiment_score = float(probs[2] - probs[0])

            return {
                "label": label,
                "confidence": confidence,
                "sentiment_score": sentiment_score,
                "scores": {
                    "negative": float(probs[0]),
                    "neutral": float(probs[1]),
                    "positive": float(probs[2]),
                },
            }

        except Exception:
            return {
                "label": "neutral",
                "confidence": 0.0,
                "sentiment_score": 0.0,
                "scores": {"negative": 0.0, "neutral": 1.0, "positive": 0.0},
            }