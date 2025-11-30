import psycopg2
import pandas as pd
from sqlalchemy import create_engine
from config.settings import DB_CONFIG

class DatabaseManager:
    def __init__(self):
        self.connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        self.engine = create_engine(self.connection_string)
    
    def save_tweet(self, tweet_data):
        """Sauvegarde un tweet dans la base de données"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()
            
            insert_query = """
            INSERT INTO tweets 
            (tweet_id, text, clean_text, created_at, user_name, user_followers_count, 
             retweet_count, favorite_count, sentiment_label, sentiment_score, confidence, language)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (tweet_id) DO NOTHING
            """
            
            cursor.execute(insert_query, (
                tweet_data['tweet_id'],
                tweet_data['text'],
                tweet_data['clean_text'],
                tweet_data['created_at'],
                tweet_data['user_name'],
                tweet_data['user_followers_count'],
                tweet_data['retweet_count'],
                tweet_data['favorite_count'],
                tweet_data['sentiment_label'],
                tweet_data['sentiment_score'],
                tweet_data['confidence'],
                tweet_data['language']
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Erreur de sauvegarde: {e}")
            return False
    
    def get_tweets(self, limit=1000):
        """Récupère les tweets de la base de données"""
        try:
            query = f"SELECT * FROM tweets ORDER BY created_at DESC LIMIT {limit}"
            return pd.read_sql_query(query, self.engine)
        except Exception as e:
            print(f"❌ Erreur de récupération: {e}")
            return pd.DataFrame()