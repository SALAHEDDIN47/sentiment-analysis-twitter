import re
import emoji
import nltk
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Nettoyage Twitter-aware + négations + emojis + hashtags.

    - URLs -> <URL>
    - Mentions -> <USER>
    - Hashtags -> garde le mot (ex: #Bitcoin -> bitcoin)
    - Répétitions: "soooo" -> "soo"
    - Emojis -> texte (emoji.demojize)
    - Stopwords: on garde les négations (not, no, never)
    - Negation trick: "not good" -> "not_good"
    """

    def __init__(self, keep_special_tokens=True):
        self.keep_special_tokens = keep_special_tokens
        self._download_nltk_resources()

        sw = set(stopwords.words("english"))
        # garder les négations (très important en sentiment)
        self.negation_words = {"not", "no", "never", "n't"}
        self.stop_words = sw - self.negation_words

        self.lemmatizer = WordNetLemmatizer()

    def _download_nltk_resources(self):
        resources = {
            "punkt": "tokenizers/punkt",
            "stopwords": "corpora/stopwords",
            "wordnet": "corpora/wordnet",
            "omw-1.4": "corpora/omw-1.4",
        }
        for name, path in resources.items():
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(name)

    def _normalize_repetitions(self, text: str) -> str:
        # "sooooo" -> "soo"
        return re.sub(r"(.)\1{2,}", r"\1\1", text)

    def _handle_negations(self, tokens):
        """
        Convertit: not good -> not_good (sur 1 mot suivant)
        """
        out = []
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t in {"not", "no", "never"} and i + 1 < len(tokens):
                out.append(f"{t}_{tokens[i+1]}")
                i += 2
            else:
                out.append(t)
                i += 1
        return out

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        try:
            text = text.lower()

            # URLs -> <URL>
            text = re.sub(r"http\S+|www\.\S+", " <URL> ", text)

            # Mentions -> <USER>
            text = re.sub(r"@\w+", " <USER> ", text)

            # Hashtags: garder le mot (#bitcoin -> bitcoin)
            text = re.sub(r"#(\w+)", r"\1", text)

            # enlever RT
            text = re.sub(r"\brt\b", " ", text)

            # normaliser répétitions
            text = self._normalize_repetitions(text)

            # emojis -> texte ":smiling_face:"
            text = emoji.demojize(text, language="en")

            # remplacer ":" "_" dans demojize -> tokens lisibles
            text = text.replace(":", " ").replace("_", " ")

            # enlever ponctuation sauf tokens spéciaux
            # on garde <URL> <USER> si keep_special_tokens
            if self.keep_special_tokens:
                # protéger tokens
                text = text.replace("<url>", " __URL__ ").replace("<user>", " __USER__ ")
                text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
                text = text.replace("__URL__", "<URL>").replace("__USER__", "<USER>")
            else:
                text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

            # enlever chiffres isolés
            text = re.sub(r"\b\d+\b", " ", text)

            # espaces multiples
            text = re.sub(r"\s+", " ", text).strip()

            tokens = word_tokenize(text)

            # supprimer stopwords (mais garder négations)
            tokens = [
                t for t in tokens
                if (t not in self.stop_words)
                and (len(t) > 1)
            ]

            # gérer négations
            tokens = self._handle_negations(tokens)

            # lemmatisation légère
            tokens = [
                self.lemmatizer.lemmatize(t)
                for t in tokens
                if t not in {"<URL>", "<USER>"} or self.keep_special_tokens
            ]

            return " ".join(tokens)

        except Exception as e:
            logger.error(f"Erreur clean_text: {e}")
            return text