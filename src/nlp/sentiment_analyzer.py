from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
import numpy as np

class SentimentAnalyzer:
    def __init__(self):
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.labels = ['negative', 'neutral', 'positive']
        
    def analyze_sentiment(self, text):
        """Analyse le sentiment d'un texte"""
        try:
            # Tokenization
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Prédiction
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = outputs.logits[0].detach().numpy()
                
            # Application softmax pour obtenir les probabilités
            scores = softmax(torch.tensor(scores), dim=-1).numpy()
            
            # Obtenir le label et le score
            predicted_label = self.labels[np.argmax(scores)]
            confidence = np.max(scores)
            sentiment_score = scores[2] - scores[0]  # positive - negative
            
            return {
                'label': predicted_label,
                'confidence': float(confidence),
                'sentiment_score': float(sentiment_score),
                'scores': {
                    'negative': float(scores[0]),
                    'neutral': float(scores[1]),
                    'positive': float(scores[2])
                }
            }
            
        except Exception as e:
            print(f"❌ Erreur d'analyse: {e}")
            return {
                'label': 'neutral',
                'confidence': 0.0,
                'sentiment_score': 0.0,
                'scores': {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0}
            }