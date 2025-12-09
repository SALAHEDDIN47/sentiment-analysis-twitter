import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import sys
import os
from dotenv import load_dotenv

# --- NOUVEAU : Import pour l'IA ---
try:
    from groq import Groq
except ImportError:
    Groq = None
# ----------------------------------

# Configuration du chemin et chargement .env
try:
    # Obtenir le chemin racine du projet
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Ajouter au path Python
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Importer le module database
    from src.database.db_manager import DatabaseManager
    load_dotenv(override=True) # Chargement forcé des variables d'environnement
    
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
    page_title="📈 Analyse de Sentiment Twitter",
    page_icon="📈", # NOUVELLE ICÔNE
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PERSONNALISÉ (MINIMALISTE & MODERNE) ---
st.markdown("""
<style>
    /* 1. Couleurs du Thème - Modernes et Professionnelles */
    :root {
        --color-primary: #007AFF; /* Bleu Professionnel (Apple Blue) */
        --color-positive: #28A745; /* Vert (Succès) */
        --color-negative: #DC3545; /* Rouge (Danger) */
        --color-neutral: #FFC107; /* Jaune (Alerte) */
        --color-ai: #A653F7; /* Violet (IA) */
        --color-background-card: #FFFFFF; /* Fond de carte blanc pur */
        --color-border-subtle: #E0E0E0; /* Gris très clair pour les bordures */
    }
    .main-header {
        font-size: 3rem;
        color: var(--color-primary);
        text-align: left; /* Plus minimaliste */
        margin-bottom: 0.5rem;
        font-weight: 300; /* Plus léger */
        border-bottom: 2px solid var(--color-border-subtle);
        padding-bottom: 10px;
    }
    /* Style pour les st.subheader (pour les graphiques) */
    h2 {
        color: #333333; /* Texte foncé pour un contraste maximal */
        border-left: 5px solid var(--color-primary);
        padding-left: 10px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    /* Cartes de métriques (Streamlit metrics) */
    .st-emotion-cache-1629p8f { 
        background-color: var(--color-background-card); 
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); /* Ombre très subtile */
        border: 1px solid var(--color-border-subtle);
    }
    .ai-card {
        background-color: #F8F9FA; /* Gris très clair pour l'IA */
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid var(--color-ai);
        margin-bottom: 2rem;
        font-size: 1.1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .positive { color: var(--color-positive); font-weight: 600; }
    .neutral { color: var(--color-neutral); font-weight: 600; }
    .negative { color: var(--color-negative); font-weight: 600; }
</style>
""", unsafe_allow_html=True)
# -----------------------------

# Titre de l'application
st.markdown('<h1 class="main-header">📈 Dashboard d\'Analyse de Sentiment Twitter</h1>', unsafe_allow_html=True) 
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

# --- FONCTION INTELLIGENTE IA (DIAGNOSTIC INCLUS) ---
def generate_ai_insight(query, total, pos, neu, neg):
    """
    Génère un paragraphe d'analyse via l'API Groq (Llama 3.1)
    """
    api_key = os.getenv("GROQ_API_KEY")
    
    # Calcul des pourcentages (évite la division par zéro)
    pos_pct = (pos/total)*100 if total > 0 else 0
    neg_pct = (neg/total)*100 if total > 0 else 0
    neu_pct = (neu/total)*100 if total > 0 else 0
    
    # 1. Essai avec la Vraie IA (Groq)
    if api_key and Groq:
        try:
            client = Groq(api_key=api_key)
            
            prompt = f"""
            Agis comme un expert en analyse de données sociales.
            Analyse ces statistiques de tweets concernant le sujet "{query if query else 'Global'}":
            - Total tweets: {total}
            - Positifs: {pos_pct:.1f}%
            - Neutres: {neu_pct:.1f}%
            - Négatifs: {neg_pct:.1f}%
            
            Rédige un paragraphe court (3-4 lignes max) et professionnel en français.
            Analyse la tendance dominante. Si le négatif est > 30%, donne une alerte.
            Ne commence pas par "Voici l'analyse", va droit au but.
            """
            
            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
            )
            return completion.choices[0].message.content
            
        except Exception as e:
            # Diagnostic imprimé dans la console
            print(f"❌ ERREUR FATALE GROQ (Basculement en mode Backup): {e}")
            pass

    # 2. Backup (Mode Template)
    dominant = "positive" if pos >= neg and pos >= neu else "négative" if neg > pos else "neutre"
    topic_text = f"concernant '{query}'" if query else "globale"
    
    text = f"D'après l'analyse sémantique de **{total} tweets** {topic_text}, la tendance générale est majoritairement **{dominant}**. "
    text += f"Nous observons que **{pos_pct:.1f}%** de la population exprime une satisfaction claire, tandis que **{neg_pct:.1f}%** des utilisateurs manifestent un mécontentement ou une frustration. "
    text += f"Le reste des discussions (**{neu_pct:.1f}%**) demeure factuel ou neutre."
    
    if neg_pct > 30:
        text += " ⚠️ **Alerte :** Le taux de négativité est élevé, suggérant un problème potentiel nécessitant une investigation immédiate."
    elif pos_pct > 60:
        text += " ✅ **Succès :** L'engagement positif est très fort, indiquant une excellente réception par la communauté."
    else:
        text += " ℹ️ **Observation :** Les avis sont partagés, reflétant un débat nuancé sans consensus clair pour le moment."
        
    return text
# ------------------------------------------

# Sidebar pour les filtres
st.sidebar.header("⚙️ Filtres et Contrôles")

# --- BARRE DE RECHERCHE ---
st.sidebar.subheader("🔎 Recherche")
search_query = st.sidebar.text_input(
    "Mots-clés",
    placeholder="Ex: bitcoin, support, erreur...",
    help="Filtrer les tweets contenant ce mot"
)
st.sidebar.markdown("---")

# Sélecteur de période
date_range = st.sidebar.date_input(
    "🗓️ Période d'analyse",
    value=(datetime.now() - timedelta(days=7), datetime.now()),
    max_value=datetime.now()
)

# Filtre de sentiment
sentiment_filter = st.sidebar.multiselect(
    "🏷️ Filtre de sentiment",
    ['positive', 'neutral', 'negative'],
    default=['positive', 'neutral', 'negative']
)

# ----------------------------------------------------------------------
# --- LIMITE DE TWEETS (Curseur) ---

st.sidebar.subheader("🔢 Nombre de tweets à afficher")

# Utilisation du curseur st.slider
tweet_limit = st.sidebar.slider(
    "Limite de Tweets",
    min_value=100,
    max_value=50000,
    value=5000,
    step=100,
    help="Limite le nombre de tweets récupérés depuis la base de données."
)

# --- MODIFICATION DE LA LIMITE DE TWEETS FINIT ICI ---
# ----------------------------------------------------------------------

# Bouton pour actualiser les données
if st.sidebar.button("🔃 Actualiser les données"): 
    st.cache_data.clear()
    st.rerun()

# Récupération des données
@st.cache_data(ttl=300)
def load_data(limit=1000):
    try:
        return db_manager.get_tweets(limit=limit)
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des données: {e}")
        return pd.DataFrame()

tweets_df = load_data(limit=tweet_limit)

# --- DIAGNOSTIC AJOUTÉ ICI ---
st.sidebar.markdown("---")
if not tweets_df.empty:
    st.sidebar.info(f"✅ **DB Chargée :** {len(tweets_df)} tweets.")
else:
    st.sidebar.warning("❌ **DB Chargée :** 0 tweet.")
# -----------------------------


if not tweets_df.empty:
    
    # --- CONVERSION DE TYPE ET NETTOYAGE RENFORCÉES ---
    try:
        tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'], errors='coerce')
        tweets_df['retweet_count'] = pd.to_numeric(tweets_df['retweet_count'], errors='coerce').fillna(0).astype(int)
        tweets_df['favorite_count'] = pd.to_numeric(tweets_df['favorite_count'], errors='coerce').fillna(0).astype(int)
        tweets_df['confidence'] = pd.to_numeric(tweets_df['confidence'], errors='coerce')
        
        # Supprimer les lignes où created_at est NaT (date non valide)
        tweets_df = tweets_df.dropna(subset=['created_at'])
        
    except KeyError as e:
        st.error(f"❌ Erreur: Colonne manquante dans les données de la base de données: {e}. Vérifiez votre requête SELECT dans db_manager.py.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erreur de conversion de type après chargement: {e}")
        st.stop()
    # --------------------------------------------------
    
    # Masque de base
    mask = (tweets_df['created_at'].dt.date >= date_range[0]) & \
           (tweets_df['created_at'].dt.date <= date_range[1]) & \
           (tweets_df['sentiment_label'].isin(sentiment_filter))
    
    # Application du filtre de recherche
    if search_query:
        search_mask = tweets_df['text'].astype(str).str.contains(search_query, case=False, na=False) | \
                      tweets_df['clean_text'].astype(str).str.contains(search_query, case=False, na=False)
        mask = mask & search_mask

    filtered_df = tweets_df[mask].copy() 
    
    if filtered_df.empty:
        if search_query:
            st.warning(f"🔍 Aucun tweet ne contient le mot : '{search_query}' dans la période/sentiment sélectionné.")
            st.sidebar.warning(f"❌ **Filtré :** 0 tweet.")
        else:
            st.warning("Aucun tweet ne correspond aux filtres sélectionnés.")
            st.sidebar.warning(f"❌ **Filtré :** 0 tweet.")
        
        st.stop()
        
    else:
        st.sidebar.success(f"📈 **Filtré :** {len(filtered_df)} tweets.")
        
        # --- PREPARATION DES DONNÉES D'ENGAGEMENT ---
        filtered_df['total_engagement'] = filtered_df['retweet_count'] + filtered_df['favorite_count']

        # Métriques principales
        st.subheader("✨ Métriques Principales") # NOUVELLE ICÔNE
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_tweets = len(filtered_df)
        positive_tweets = len(filtered_df[filtered_df['sentiment_label'] == 'positive'])
        neutral_tweets = len(filtered_df[filtered_df['sentiment_label'] == 'neutral'])
        negative_tweets = len(filtered_df[filtered_df['sentiment_label'] == 'negative'])
        
        with col1:
            st.metric("Total Tweets", total_tweets)
        with col2:
            st.metric("👍 Positifs", positive_tweets) # NOUVELLE ICÔNE
        with col3:
            st.metric("⚪ Neutres", neutral_tweets) # NOUVELLE ICÔNE
        with col4:
            st.metric("👎 Négatifs", negative_tweets) # NOUVELLE ICÔNE
        with col5:
            if total_tweets > 0:
                positive_ratio = (positive_tweets / total_tweets) * 100
                st.metric("🎯 Score Positif", f"{positive_ratio:.1f}%") # NOUVELLE ICÔNE
            else:
                st.metric("🎯 Score Positif", "0%")
        
        st.markdown("---")

        # --- SECTION GÉNÉRATION IA (INTÉGRÉE ICI) ---
        if total_tweets > 0:
            with st.spinner('💡 L\'IA analyse vos données...'): # NOUVELLE ICÔNE
                ai_insight = generate_ai_insight(
                    search_query, 
                    total_tweets, 
                    positive_tweets, 
                    neutral_tweets, 
                    negative_tweets
                )
            st.markdown(f'<div class="ai-card"><strong>💡 Analyse IA & Insights :</strong><br>{ai_insight}</div>', unsafe_allow_html=True) # NOUVELLE ICÔNE
        # --------------------------------------------
        
        # Ligne 1 de Graphiques (Distribution des Sentiments et Histogramme de Confiance)
        col1, col2 = st.columns(2)
        
        # --- Définir la palette de couleurs pour Plotly (MODERNE & CORRIGÉE) ---
        SENTIMENT_COLOR_MAP = {
            'positive': '#28A745', # Vert
            'neutral': '#FFC107', # Jaune
            'negative': '#DC3545' # Rouge
        }

        with col1:
            # --- GRAPHIQUE 1 : Distribution des Sentiments (Camembert) ---
            st.subheader("📈 Distribution des Sentiments") # NOUVELLE ICÔNE
            sentiment_counts = filtered_df['sentiment_label'].value_counts()
            
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                color=sentiment_counts.index,
                color_discrete_map=SENTIMENT_COLOR_MAP
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # --- GRAPHIQUE 2 : Distribution des Scores de Confiance (Histogramme - Version simplifiée) ---
            st.subheader("🎯 Distribution des Scores de Confiance par Sentiment") # NOUVELLE ICÔNE
            if 'confidence' in filtered_df.columns:
                
                df_plot = filtered_df[['sentiment_label', 'confidence']].dropna() 
                
                fig_hist = px.histogram(
                    df_plot,
                    x="confidence",
                    color="sentiment_label",
                    histnorm='percent',
                    title="Fréquence des Niveaux de Confiance (plus simple)",
                    labels={'confidence': 'Score de Confiance du Modèle', 'percent': 'Pourcentage'},
                    color_discrete_map=SENTIMENT_COLOR_MAP,
                    nbins=10 
                )
                fig_hist.update_layout(barmode='group') # Mode Groupé
                fig_hist.update_traces(opacity=0.9) 
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("📉 La colonne 'confidence' est manquante ou les données sont vides.")

        # ----------------------------------------------------
        # Ligne 2 de Graphiques (Top Utilisateurs et Évolution Temporelle Segmentée)
        st.markdown("---")
        col3, col4 = st.columns(2)
        
        with col3:
            # --- GRAPHIQUE 3 : Top 10 Utilisateurs Actifs ---
            st.subheader("👤 Top 10 Utilisateurs les Plus Actifs") # NOUVELLE ICÔNE
            
            top_users = filtered_df['user_name'].value_counts().nlargest(10).reset_index()
            top_users.columns = ['user_name', 'tweet_count']
            
            fig_users = px.bar(
                top_users,
                x='tweet_count',
                y='user_name',
                orientation='h',
                title="Volume de Tweets par Utilisateur",
                color_continuous_scale=px.colors.sequential.Plotly3,
                template='plotly_white'
            )
            fig_users.update_layout(yaxis={'categoryorder':'total ascending'}) 
            st.plotly_chart(fig_users, use_container_width=True)

        with col4:
            # --- GRAPHIQUE 4 : Évolution Temporelle Segmentée (Barres + Ligne Total) ---
            st.subheader("🗓️ Évolution Temporelle par Sentiment (Barres + Ligne)") # NOUVELLE ICÔNE
            
            daily_sentiment = filtered_df.groupby([
                filtered_df['created_at'].dt.date, 
                'sentiment_label'
            ]).size().reset_index(name='count')
            
            if not daily_sentiment.empty: 
                
                daily_sentiment['created_at'] = daily_sentiment['created_at'].astype(str)
                
                # Calcul du total quotidien pour la ligne de tendance
                daily_total = daily_sentiment.groupby('created_at')['count'].sum().reset_index(name='total_count')
                
                # 1. Créer le tracé des Barres (via Plotly Go)
                fig_evo = go.Figure()

                for sentiment, color in SENTIMENT_COLOR_MAP.items():
                    df_sent = daily_sentiment[daily_sentiment['sentiment_label'] == sentiment]
                    fig_evo.add_trace(go.Bar(
                        x=df_sent['created_at'],
                        y=df_sent['count'],
                        name=sentiment.capitalize(),
                        marker_color=color,
                        opacity=0.8
                    ))

                # 2. Ajouter la Ligne de Tendance (Volume Total)
                LINE_COLOR = '#007AFF' 
                fig_evo.add_trace(go.Scatter(
                    x=daily_total['created_at'],
                    y=daily_total['total_count'],
                    mode='lines+markers',
                    name='Total Quotidien',
                    marker=dict(color=LINE_COLOR, size=8), 
                    line=dict(color=LINE_COLOR, width=3),
                    yaxis='y2' 
                ))

                # 3. Mettre à jour le Layout (titres et axes)
                fig_evo.update_layout(
                    title="Volume Quotidien de Tweets (Sentiment + Total)",
                    xaxis=dict(type='category', title='Date'),
                    # Axe Y principal pour les barres (Count)
                    yaxis=dict(title='Nombre de Tweets (Sentiment)', showgrid=False),
                    # Axe Y secondaire pour la ligne (Total)
                    yaxis2=dict(
                        title='Total Quotidien',
                        overlaying='y',
                        side='right',
                        showgrid=False
                    ),
                    legend_title='Légende',
                    barmode='group',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig_evo, use_container_width=True)
            else:
                st.info("📉 L'évolution n'est pas affichée: les données sont trop rares ou couvrent une seule journée.")

        # ----------------------------------------------------
        
        st.subheader("📝 Tweets Récents Analysés") # NOUVELLE ICÔNE
        
        display_columns = st.multiselect(
            "Colonnes à afficher :",
            options=['created_at', 'user_name', 'clean_text', 'sentiment_label', 'confidence', 'retweet_count', 'favorite_count', 'total_engagement'],
            default=['created_at', 'user_name', 'clean_text', 'sentiment_label', 'confidence']
        )
        
        if display_columns:
            def color_sentiment(row):
                # Couleurs très claires pour ne pas surcharger la table
                colors = {
                    'positive': 'background-color: #E6F7ED; color: black;', 
                    'negative': 'background-color: #FEE7E9; color: black;',
                    'neutral': 'background-color: #FFF9E6; color: black;'
                }
                return [colors.get(row['sentiment_label'], '')] * len(row)
            
            display_df = filtered_df[display_columns].head(50)
            styled_df = display_df.style.apply(color_sentiment, axis=1)
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            csv = filtered_df[display_columns].to_csv(index=False)
            st.download_button(
                label="📥 Télécharger les données (CSV)",
                data=csv,
                file_name=f"tweets_sentiment_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

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
        📈 Dashboard d'Analyse de Sentiment Twitter - Développé avec Streamlit
    </div>
    """,
    unsafe_allow_html=True
)