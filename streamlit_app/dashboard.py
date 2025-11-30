import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import sys
import os

# Configuration du chemin
try:
    # Obtenir le chemin racine du projet
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Ajouter au path Python
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Importer le module database
    from src.database.db_manager import DatabaseManager
    
except ImportError as e:
    st.error(f"❌ Erreur d'importation: {e}")
    st.info("""
    **Solution:**
    1. Assurez-vous que tous les fichiers du projet sont présents
    2. Vérifiez la structure des dossiers
    3. Exécutez Streamlit depuis la racine du projet
    """)
    st.stop()

# Configuration de la page
st.set_page_config(
    page_title="🐦 Analyse de Sentiment Twitter",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1DA1F2;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .positive { color: #00ff00; }
    .neutral { color: #ffff00; }
    .negative { color: #ff0000; }
</style>
""", unsafe_allow_html=True)

# Titre de l'application
st.markdown('<h1 class="main-header">🐦 Dashboard d\'Analyse de Sentiment Twitter</h1>', unsafe_allow_html=True)
st.markdown("---")

# Initialisation de la base de données
@st.cache_resource
def init_db():
    try:
        return DatabaseManager()
    except Exception as e:
        st.error(f"❌ Erreur de connexion à la base de données: {e}")
        return None

db_manager = init_db()

if db_manager is None:
    st.error("""
    **Impossible de se connecter à la base de données.**
    
    Vérifiez que:
    - PostgreSQL est démarré
    - La base de données 'twitter_sentiment' existe
    - Les identifiants dans le fichier .env sont corrects
    - Vous avez exécuté `python setup_database.py`
    """)
    st.stop()

# Sidebar pour les filtres
st.sidebar.header("🔧 Filtres et Contrôles")

# Sélecteur de période
date_range = st.sidebar.date_input(
    "📅 Période d'analyse",
    value=(datetime.now() - timedelta(days=7), datetime.now()),
    max_value=datetime.now()
)

# Filtre de sentiment
sentiment_filter = st.sidebar.multiselect(
    "🎭 Filtre de sentiment",
    ['positive', 'neutral', 'negative'],
    default=['positive', 'neutral', 'negative']
)

# Nombre de tweets à afficher
tweet_limit = st.sidebar.slider(
    "📊 Nombre de tweets à afficher",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100
)

# Bouton pour actualiser les données
if st.sidebar.button("🔄 Actualiser les données"):
    st.cache_data.clear()
    st.rerun()

# Récupération des données
@st.cache_data(ttl=300)  # Cache pour 5 minutes
def load_data(limit=1000):
    try:
        return db_manager.get_tweets(limit=limit)
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des données: {e}")
        return pd.DataFrame()

tweets_df = load_data(limit=tweet_limit)

if not tweets_df.empty:
    # Filtrage des données
    tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'])
    mask = (tweets_df['created_at'].dt.date >= date_range[0]) & \
           (tweets_df['created_at'].dt.date <= date_range[1]) & \
           (tweets_df['sentiment_label'].isin(sentiment_filter))
    filtered_df = tweets_df[mask]
    
    # Métriques principales
    st.subheader("📈 Métriques Principales")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_tweets = len(filtered_df)
        st.metric("Total Tweets", total_tweets)
    
    with col2:
        positive_tweets = len(filtered_df[filtered_df['sentiment_label'] == 'positive'])
        st.metric("😊 Positifs", positive_tweets)
    
    with col3:
        neutral_tweets = len(filtered_df[filtered_df['sentiment_label'] == 'neutral'])
        st.metric("😐 Neutres", neutral_tweets)
    
    with col4:
        negative_tweets = len(filtered_df[filtered_df['sentiment_label'] == 'negative'])
        st.metric("😞 Négatifs", negative_tweets)
    
    with col5:
        if total_tweets > 0:
            positive_ratio = (positive_tweets / total_tweets) * 100
            st.metric("📊 Score Positif", f"{positive_ratio:.1f}%")
        else:
            st.metric("📊 Score Positif", "0%")
    
    st.markdown("---")
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution des sentiments (Camembert)
        st.subheader("🎪 Distribution des Sentiments")
        sentiment_counts = filtered_df['sentiment_label'].value_counts()
        
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={
                'positive': "#67F267",
                'neutral': "#FFFF49", 
                'negative': "#F03B3B"
            }
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Sentiment dans le temps
        st.subheader("📅 Évolution Temporelle")
        
        if not filtered_df.empty:
            # Grouper par date et sentiment
            daily_sentiment = filtered_df.groupby([
                filtered_df['created_at'].dt.date, 
                'sentiment_label'
            ]).size().unstack(fill_value=0)
            
            # S'assurer que toutes les colonnes de sentiment existent
            for sentiment in ['positive', 'neutral', 'negative']:
                if sentiment not in daily_sentiment.columns:
                    daily_sentiment[sentiment] = 0
            
            fig_line = px.line(
                daily_sentiment,
                title="Nombre de tweets par sentiment dans le temps",
                labels={'value': 'Nombre de Tweets', 'created_at': 'Date'},
                color_discrete_map={
                    'positive': '#00FF00',
                    'neutral': '#FFFF00',
                    'negative': '#FF0000'
                }
            )
            fig_line.update_layout(showlegend=True)
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Aucune donnée à afficher pour la période sélectionnée.")
    
    # Tableau des tweets récents
    st.subheader("📋 Tweets Récents Analysés")
    
    # Options d'affichage
    display_columns = st.multiselect(
        "Colonnes à afficher :",
        options=['created_at', 'user_name', 'clean_text', 'sentiment_label', 'confidence', 'retweet_count', 'favorite_count'],
        default=['created_at', 'user_name', 'clean_text', 'sentiment_label', 'confidence']
    )
    
    if not filtered_df.empty and display_columns:
        # Fonction de coloration des lignes
        def color_sentiment(row):
            colors = {
        'positive': 'background-color: #90EE90; color: black;',  # Vert clair, texte noir
        'negative': 'background-color: #FFB6C1; color: black;',  # Rose clair, texte noir
        'neutral': 'background-color: #FFFACD; color: black;'    # Jaune très clair, texte noir
    }
            return [colors.get(row['sentiment_label'], '')] * len(row)
        
        # Afficher le dataframe avec style
        display_df = filtered_df[display_columns].head(50)
        styled_df = display_df.style.apply(color_sentiment, axis=1)
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=400
        )
        
        # Bouton de téléchargement
        csv = filtered_df[display_columns].to_csv(index=False)
        st.download_button(
            label="📥 Télécharger les données (CSV)",
            data=csv,
            file_name=f"tweets_sentiment_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("Aucun tweet à afficher avec les filtres sélectionnés.")

else:
    st.warning("📭 Aucune donnée disponible dans la base de données.")
    st.info("""
    **Pour commencer :**
    1. Vérifiez que vous avez exécuté `python main.py` pour collecter des données
    2. Assurez-vous que la base de données contient des tweets
    3. Vérifiez la connexion à PostgreSQL
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        🐦 Dashboard d'Analyse de Sentiment Twitter - Développé avec Streamlit
    </div>
    """,
    unsafe_allow_html=True
)