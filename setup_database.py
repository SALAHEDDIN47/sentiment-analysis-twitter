import psycopg2
from config.settings import DB_CONFIG

def setup_database():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Créer la table si elle n'existe pas
    create_table_query = """
    CREATE TABLE IF NOT EXISTS tweets (
        id SERIAL PRIMARY KEY,
        tweet_id BIGINT UNIQUE,
        text TEXT,
        clean_text TEXT,
        created_at TIMESTAMP,
        user_name VARCHAR(255),
        user_followers_count INTEGER,
        retweet_count INTEGER,
        favorite_count INTEGER,
        sentiment_label VARCHAR(50),
        sentiment_score FLOAT,
        confidence FLOAT,
        language VARCHAR(10),
        collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    cursor.execute(create_table_query)
    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Base de données configurée avec succès!")

if __name__ == "__main__":
    setup_database()