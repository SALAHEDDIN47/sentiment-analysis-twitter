import time
#from src.data_collection.twitter_client import TwitterClient
from src.data_collection.mock_client import MockTwitterClient
from src.preprocessing.text_cleaner import TextCleaner
from src.nlp.sentiment_analyzer import SentimentAnalyzer
from src.database.db_manager import DatabaseManager
from config.settings import COLLECTION_CONFIG

def main():
    # Initialisation des composants
    #twitter_client = TwitterClient()
    twitter_client = MockTwitterClient()
    text_cleaner = TextCleaner()
    sentiment_analyzer = SentimentAnalyzer()
    db_manager = DatabaseManager()
    
    print("🚀 Démarrage de la collecte et analyse des tweets...")
    
    # Collecte des tweets
    for keyword in COLLECTION_CONFIG['keywords']:
        print(f"🔍 Collecte des tweets pour: {keyword}")
        
        tweets_df = twitter_client.collect_tweets(
            query=keyword,
            max_tweets=100
        )
        
        if not tweets_df.empty:
            print(f"📊 {len(tweets_df)} tweets collectés pour '{keyword}'")
            
            # Traitement de chaque tweet
            for index, tweet in tweets_df.iterrows():
                # Nettoyage du texte
                clean_text = text_cleaner.clean_text(tweet['text'])
                
                # Analyse de sentiment
                sentiment_result = sentiment_analyzer.analyze_sentiment(clean_text)
                
                # Préparation des données pour la sauvegarde
                tweet_data = {
                    'tweet_id': tweet['tweet_id'],
                    'text': tweet['text'],
                    'clean_text': clean_text,
                    'created_at': tweet['created_at'],
                    'user_name': tweet['user_name'],
                    'user_followers_count': tweet['user_followers_count'],
                    'retweet_count': tweet['retweet_count'],
                    'favorite_count': tweet['favorite_count'],
                    'sentiment_label': sentiment_result['label'],
                    'sentiment_score': sentiment_result['sentiment_score'],
                    'confidence': sentiment_result['confidence'],
                    'language': tweet['language']
                }
                
                # Sauvegarde dans la base de données
                if db_manager.save_tweet(tweet_data):
                    print(f"✅ Tweet sauvegardé: {sentiment_result['label']} (confiance: {sentiment_result['confidence']:.2f})")
                else:
                    print("❌ Erreur de sauvegarde du tweet")
        
        # Pause pour éviter les limites de rate limiting
        time.sleep(10)
    
    print("✅ Collecte et analyse terminées!")

if __name__ == "__main__":
    main()