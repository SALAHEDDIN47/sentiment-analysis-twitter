import psycopg2
import pandas as pd
from sqlalchemy import create_engine, text
from config.settings import DB_CONFIG


class DatabaseManager:
    """
    PostgreSQL Database Manager

    - save_tweet(tweet_data): insertion avec ON CONFLICT (tweet_id) DO NOTHING
    - insert_tweet(...): compatibilité avec main.py (wrapper vers save_tweet)
    - get_tweets(limit): lecture pour Streamlit
    - init_schema(): crée la table si elle n'existe pas (et colonnes manquantes utiles)
    - count_tweets(): nombre de tweets
    - clear_all(): vider la table (optionnel)
    """

    def __init__(self):
        self.connection_string = (
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
            f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )
        self.engine = create_engine(self.connection_string)
        self.init_schema()  # ✅ assure que la table existe + colonnes nécessaires

    def _get_conn(self):
        return psycopg2.connect(**DB_CONFIG)

    def init_schema(self):
        """
        Crée la table si elle n'existe pas.
        Ajoute aussi des colonnes utiles si votre table tweets vient d'une autre version du projet.
        """
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS tweets (
            tweet_id TEXT PRIMARY KEY,
            text TEXT,
            clean_text TEXT,
            created_at TEXT,
            user_name TEXT,
            user_followers_count INTEGER,
            retweet_count INTEGER,
            favorite_count INTEGER,
            sentiment_label TEXT,
            sentiment_score DOUBLE PRECISION,
            confidence DOUBLE PRECISION,
            language TEXT,
            true_label TEXT
        );
        """

        # Si la table existe déjà mais sans certaines colonnes, on les ajoute.
        # (ex: true_label absent)
        alter_sqls = [
            "ALTER TABLE tweets ADD COLUMN IF NOT EXISTS true_label TEXT;",
            "ALTER TABLE tweets ADD COLUMN IF NOT EXISTS clean_text TEXT;",
            "ALTER TABLE tweets ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION;",
            "ALTER TABLE tweets ADD COLUMN IF NOT EXISTS sentiment_score DOUBLE PRECISION;",
            "ALTER TABLE tweets ADD COLUMN IF NOT EXISTS sentiment_label TEXT;",
            "ALTER TABLE tweets ADD COLUMN IF NOT EXISTS created_at TEXT;",
            "ALTER TABLE tweets ADD COLUMN IF NOT EXISTS user_name TEXT;",
            "ALTER TABLE tweets ADD COLUMN IF NOT EXISTS language TEXT;",
        ]

        try:
            with self._get_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_table_sql)
                    for q in alter_sqls:
                        cursor.execute(q)
                conn.commit()
        except Exception as e:
            print(f"❌ Erreur init_schema: {e}")

    def save_tweet(self, tweet_data: dict) -> bool:
        """
        Sauvegarde un tweet dans la base de données.
        Cette méthode accepte des champs manquants (valeurs par défaut).
        """
        try:
            # valeurs par défaut robustes
            data = {
                "tweet_id": tweet_data.get("tweet_id"),
                "text": tweet_data.get("text", ""),
                "clean_text": tweet_data.get("clean_text", ""),
                "created_at": tweet_data.get("created_at"),
                "user_name": tweet_data.get("user_name"),
                "user_followers_count": tweet_data.get("user_followers_count"),
                "retweet_count": tweet_data.get("retweet_count"),
                "favorite_count": tweet_data.get("favorite_count"),
                "sentiment_label": tweet_data.get("sentiment_label"),
                "sentiment_score": tweet_data.get("sentiment_score"),
                "confidence": tweet_data.get("confidence"),
                "language": tweet_data.get("language", "en"),
                "true_label": tweet_data.get("true_label"),
            }

            # tweet_id est obligatoire pour ON CONFLICT(tweet_id)
            # Pour Sentiment140, on peut générer un tweet_id si absent
            if not data["tweet_id"]:
                # petit id déterministe (évite doublons sur mêmes textes)
                data["tweet_id"] = str(abs(hash((data["text"], data["created_at"], data["user_name"]))))[:18]

            insert_query = """
            INSERT INTO tweets
            (tweet_id, text, clean_text, created_at, user_name, user_followers_count,
             retweet_count, favorite_count, sentiment_label, sentiment_score, confidence, language, true_label)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (tweet_id) DO NOTHING
            """

            with self._get_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(insert_query, (
                        data["tweet_id"],
                        data["text"],
                        data["clean_text"],
                        data["created_at"],
                        data["user_name"],
                        data["user_followers_count"],
                        data["retweet_count"],
                        data["favorite_count"],
                        data["sentiment_label"],
                        data["sentiment_score"],
                        data["confidence"],
                        data["language"],
                        data["true_label"],
                    ))
                conn.commit()

            return True

        except Exception as e:
            print(f"❌ Erreur de sauvegarde: {e}")
            return False

    # ✅ Méthode attendue par ton main.py (compatibilité)
    def insert_tweet(
        self,
        text,
        clean_text,
        sentiment_label,
        confidence,
        sentiment_score,
        user_name=None,
        created_at=None,
        true_label=None,
        tweet_id=None,
        user_followers_count=None,
        retweet_count=None,
        favorite_count=None,
        language="en",
    ):
        tweet_data = {
            "tweet_id": tweet_id,  # peut être None -> auto-généré
            "text": text,
            "clean_text": clean_text,
            "created_at": created_at,
            "user_name": user_name,
            "user_followers_count": user_followers_count,
            "retweet_count": retweet_count,
            "favorite_count": favorite_count,
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "confidence": confidence,
            "language": language,
            "true_label": true_label,
        }
        return self.save_tweet(tweet_data)

    def get_tweets(self, limit=1000) -> pd.DataFrame:
        """
        Récupère les tweets de la base de données (pour Streamlit).
        """
        try:
            query = text("SELECT * FROM tweets ORDER BY created_at DESC NULLS LAST LIMIT :limit")
            return pd.read_sql_query(query, self.engine, params={"limit": int(limit)})
        except Exception as e:
            print(f"❌ Erreur de récupération: {e}")
            return pd.DataFrame()

    def count_tweets(self) -> int:
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM tweets"))
                return int(result.scalar())
        except Exception:
            return 0

    def clear_all(self) -> bool:
        """
        Vide la table (optionnel).
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("TRUNCATE TABLE tweets"))
                conn.commit()
            return True
        except Exception as e:
            print(f"❌ Erreur clear_all: {e}")
            return False