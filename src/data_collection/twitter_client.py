import tweepy
import pandas as pd
from datetime import datetime
from config.settings import TWITTER_CONFIG, COLLECTION_CONFIG
import logging

logger = logging.getLogger(__name__)

class TwitterClient:
    def __init__(self):
        self.client_v2 = None
        self.api_v1 = None
        self.setup_clients()
    
    def setup_clients(self):
        """Configure les clients API v1.1 et v2"""
        try:
            # Client API v2 (Bearer Token)
            if TWITTER_CONFIG['bearer_token']:
                self.client_v2 = tweepy.Client(
                    bearer_token=TWITTER_CONFIG['bearer_token'],
                    wait_on_rate_limit=True
                )
                logger.info("✅ Client Twitter API v2 initialisé")
            
            # Client API v1.1 (pour compatibilité)
            if all([TWITTER_CONFIG['api_key'], TWITTER_CONFIG['api_secret'], 
                    TWITTER_CONFIG['access_token'], TWITTER_CONFIG['access_secret']]):
                
                auth = tweepy.OAuthHandler(
                    TWITTER_CONFIG['api_key'], 
                    TWITTER_CONFIG['api_secret']
                )
                auth.set_access_token(
                    TWITTER_CONFIG['access_token'], 
                    TWITTER_CONFIG['access_secret']
                )
                self.api_v1 = tweepy.API(auth, wait_on_rate_limit=True)
                logger.info("✅ Client Twitter API v1.1 initialisé")
                
        except Exception as e:
            logger.error(f"❌ Erreur d'initialisation Twitter: {e}")
    
    def collect_tweets_v2(self, query, max_tweets=100):
        """Utilise l'API v2 (méthode recommandée)"""
        try:
            if not self.client_v2:
                logger.error("❌ Client API v2 non configuré")
                return pd.DataFrame()
            
            tweets_data = []
            tweet_count = 0
            
            # Pagination pour récupérer plus de tweets
            for tweet in tweepy.Paginator(
                self.client_v2.search_recent_tweets,
                query=query,
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'lang'],
                user_fields=['username', 'public_metrics'],
                expansions=['author_id'],
                max_results=min(100, max_tweets)  # 100 max par page
            ).flatten(limit=max_tweets):
                
                tweet_data = {
                    'tweet_id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'user_name': getattr(tweet, 'username', 'unknown'),
                    'user_followers_count': getattr(tweet, 'followers_count', 0),
                    'retweet_count': getattr(tweet, 'retweet_count', 0),
                    'favorite_count': getattr(tweet, 'like_count', 0),
                    'language': getattr(tweet, 'lang', 'en')
                }
                tweets_data.append(tweet_data)
                tweet_count += 1
            
            logger.info(f"✅ {tweet_count} tweets collectés via API v2 pour '{query}'")
            return pd.DataFrame(tweets_data)
            
        except Exception as e:
            logger.error(f"❌ Erreur API v2 pour '{query}': {e}")
            return pd.DataFrame()
    
    def collect_tweets(self, query, max_tweets=100):
        """Méthode principale - essaie d'abord API v2, puis v1.1"""
        # Essayer API v2 d'abord
        df = self.collect_tweets_v2(query, max_tweets)
        
        if not df.empty:
            return df
        
        logger.warning("⚠️ API v2 a échoué, vérifiez votre Bearer Token")
        return pd.DataFrame()