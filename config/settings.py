import os
from dotenv import load_dotenv

load_dotenv()

# Configuration Twitter
TWITTER_CONFIG = {
    'api_key': os.getenv('TWITTER_API_KEY'),
    'api_secret': os.getenv('TWITTER_API_SECRET'),
    'access_token': os.getenv('TWITTER_ACCESS_TOKEN'),
    'access_secret': os.getenv('TWITTER_ACCESS_SECRET'),
    'bearer_token': os.getenv('TWITTER_BEARER_TOKEN')
}

# Configuration Database
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

# Paramètres de collecte
COLLECTION_CONFIG = {
    'languages': ['en'],
    'max_tweets': 10,
    'keywords': ['python', 'machine learning', 'ai', 'artificial intelligence']
}