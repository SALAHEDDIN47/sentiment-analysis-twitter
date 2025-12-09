import pandas as pd
import sys
import time
from src.preprocessing.text_cleaner import TextCleaner
from src.nlp.sentiment_analyzer import SentimentAnalyzer
from src.database.db_manager import DatabaseManager

def main():
    print("🚀 STARTING FULL DATA IMPORT...")
    
    # 1. Load the CSV
    try:
        # We read the whole file. 
        # Note: If you have the full Sentiment140 (1.6 Million rows), this takes 5-10 seconds to load.
        cols = ['target', 'ids', 'date', 'flag', 'user', 'text']
        
        # Try reading with headers first, if it fails, use manual headers
        try:
            df = pd.read_csv("data.csv", encoding='latin-1')
            if 'text' not in df.columns:
                 df = pd.read_csv("data.csv", encoding='latin-1', names=cols)
        except:
             df = pd.read_csv("data.csv", encoding='latin-1', names=cols)
            
        total_rows = len(df)
        print(f"✅ Loaded CSV with {total_rows} rows.")
        
    except FileNotFoundError:
        print("❌ CRITICAL ERROR: 'data.csv' file not found.")
        sys.exit(1)

    # 2. Setup Tools
    db_manager = DatabaseManager()
    text_cleaner = TextCleaner()
    sentiment_analyzer = SentimentAnalyzer()

    print("🔄 Processing ALL tweets... (This may take a while)")
    
    # 3. Process EVERYTHING (No .head() limit)
    success_count = 0
    start_time = time.time()

    for index, row in df.iterrows():
        try:
            # Handle column names safely
            text = str(row.get('text', row.get('content', '')))
            user = str(row.get('user', row.get('author', 'unknown')))
            
            if not text or text == "nan":
                continue

            # Process
            clean = text_cleaner.clean_text(text)
            sent = sentiment_analyzer.analyze_sentiment(clean)

            tweet_data = {
                'tweet_id': index, 
                'text': text,
                'clean_text': clean,
                'created_at': pd.Timestamp.now(),
                'user_name': user,
                'user_followers_count': 0, 'retweet_count': 0, 'favorite_count': 0,
                'sentiment_label': sent['label'],
                'sentiment_score': sent['sentiment_score'],
                'confidence': sent['confidence'],
                'language': 'en'
            }

            if db_manager.save_tweet(tweet_data):
                success_count += 1
            
            # Progress Update every 100 tweets
            if success_count % 100 == 0:
                elapsed = time.time() - start_time
                speed = success_count / elapsed if elapsed > 0 else 0
                print(f"⏳ Progress: {success_count}/{total_rows} ({speed:.1f} tweets/sec) - Last: {sent['label']}")

        except Exception as e:
            # Don't crash on one bad row, just print error and continue
            print(f"⚠️ Skipped row {index}: {e}")

    print(f"\n🎉 DONE. Successfully inserted {success_count} tweets into the database.")

if __name__ == "__main__":
    main()