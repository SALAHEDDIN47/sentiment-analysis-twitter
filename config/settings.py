import os
from dotenv import load_dotenv

load_dotenv()

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