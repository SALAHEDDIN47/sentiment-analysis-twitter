import re
import emoji
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging

# Configurer le logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextCleaner:
    def __init__(self):
        # Télécharger les ressources si manquantes
        self._download_nltk_resources()
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Charger spaCy avec gestion d'erreur
        try:
            self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        except OSError:
            logger.warning("Modèle spaCy 'en_core_web_sm' non trouvé. Installation...")
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    
    def _download_nltk_resources(self):
        """Télécharge les ressources NLTK si manquantes"""
        resources = {
            'punkt_tab': 'tokenizers/punkt_tab',
            'punkt': 'tokenizers/punkt',
            'stopwords': 'corpora/stopwords',
            'wordnet': 'corpora/wordnet',
            'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
            'omw-1.4': 'corpora/omw-1.4'
        }
        
        for resource_name, resource_path in resources.items():
            try:
                nltk.data.find(resource_path)
                logger.info(f"✅ {resource_name} déjà disponible")
            except LookupError:
                logger.info(f"📥 Téléchargement de {resource_name}...")
                nltk.download(resource_name)
        
    def clean_text(self, text):
        """Nettoie et prétraite le texte avec gestion d'erreur"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        try:
            # Convertir en minuscule
            text = text.lower()
            
            # Supprimer les URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Supprimer les mentions et hashtags
            text = re.sub(r'@\w+|#\w+', '', text)
            
            # Convertir les émojis en texte
            text = emoji.demojize(text)
            
            # Supprimer la ponctuation et les caractères spéciaux
            text = re.sub(r'[^\w\s]', ' ', text)
            
            # Supprimer les chiffres
            text = re.sub(r'\d+', '', text)
            
            # Supprimer les espaces multiples
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Tokenization et suppression des stopwords
            tokens = word_tokenize(text)
            tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
            
            # Lemmatisation
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            return ' '.join(tokens)
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage du texte: {e}")
            return text  # Retourner le texte original en cas d'erreur
    
    def detect_language(self, text):
        """Détection simple de la langue"""
        if not text:
            return False
            
        english_words = set(['the', 'and', 'is', 'in', 'to', 'of', 'a', 'for'])
        words = set(text.lower().split())
        return len(words.intersection(english_words)) > 2