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
    load_dotenv(override=True)  # Chargement forcé des variables d'environnement

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
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PERSONNALISÉ (MINIMALISTE & MODERNE) ---
st.markdown("""
<style>
    :root {
        --color-primary: #007AFF;
        --color-positive: #28A745;
        --color-negative: #DC3545;
        --color-neutral: #FFC107;
        --color-ai: #A653F7;
        --color-background-card: #FFFFFF;
        --color-border-subtle: #E0E0E0;
    }
    .main-header {
        font-size: 3rem;
        color: var(--color-primary);
        text-align: left;
        margin-bottom: 0.5rem;
        font-weight: 300;
        border-bottom: 2px solid var(--color-border-subtle);
        padding-bottom: 10px;
    }
    h2 {
        color: #333333;
        border-left: 5px solid var(--color-primary);
        padding-left: 10px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .st-emotion-cache-1629p8f {
        background-color: var(--color-background-card);
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid var(--color-border-subtle);
    }
    .ai-card {
        background-color: #F8F9FA;
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

# Titre
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
    - La base de données existe
    - Les identifiants dans le fichier .env sont corrects
    """)
    st.stop()

# --- FONCTION IA ---
def generate_ai_insight(query, total, pos, neu, neg):
    api_key = os.getenv("GROQ_API_KEY")
    pos_pct = (pos/total)*100 if total > 0 else 0
    neg_pct = (neg/total)*100 if total > 0 else 0
    neu_pct = (neu/total)*100 if total > 0 else 0

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
            print(f"❌ ERREUR GROQ (fallback): {e}")

    dominant = "positive" if pos >= neg and pos >= neu else "négative" if neg > pos else "neutre"
    topic_text = f"concernant '{query}'" if query else "globale"
    text = f"D'après l'analyse sémantique de **{total} tweets** {topic_text}, la tendance générale est majoritairement **{dominant}**. "
    text += f"Nous observons que **{pos_pct:.1f}%** exprime une satisfaction claire, tandis que **{neg_pct:.1f}%** manifestent un mécontentement. "
    text += f"Le reste (**{neu_pct:.1f}%**) demeure neutre."
    if neg_pct > 30:
        text += " ⚠️ **Alerte :** taux négatif élevé."
    elif pos_pct > 60:
        text += " ✅ **Succès :** engagement positif très fort."
    else:
        text += " ℹ️ **Observation :** avis partagés."
    return text

# Sidebar
st.sidebar.header("⚙️ Filtres et Contrôles")

st.sidebar.subheader("🔎 Recherche")
search_query = st.sidebar.text_input(
    "Mots-clés",
    placeholder="Ex: bitcoin, support, erreur...",
    help="Filtrer les tweets contenant ce mot"
)
st.sidebar.markdown("---")

# Charger un petit sample pour déterminer la période disponible dans la DB
@st.cache_data(ttl=300)
def load_data(limit=1000):
    try:
        return db_manager.get_tweets(limit=limit)
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des données: {e}")
        return pd.DataFrame()

tmp_df = load_data(limit=2000)

disable_date_filter = False
if not tmp_df.empty and 'created_at' in tmp_df.columns:
    tmp_dates = pd.to_datetime(tmp_df['created_at'].astype(str), errors='coerce', utc=True)
    if tmp_dates.isna().mean() > 0.7:
        st.sidebar.warning("⚠️ Dates created_at non parsables → filtre date désactivé.")
        disable_date_filter = True
        date_range = (datetime.now().date() - timedelta(days=365), datetime.now().date())
    else:
        min_d = tmp_dates.min().date()
        max_d = tmp_dates.max().date()
        date_range = st.sidebar.date_input(
            "🗓️ Période d'analyse",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d
        )
else:
    disable_date_filter = True
    date_range = (datetime.now().date() - timedelta(days=365), datetime.now().date())

sentiment_filter = st.sidebar.multiselect(
    "🏷️ Filtre de sentiment",
    ['positive', 'neutral', 'negative'],
    default=['positive', 'neutral', 'negative']
)

st.sidebar.subheader("🔢 Nombre de tweets à afficher")
tweet_limit = st.sidebar.slider(
    "Limite de Tweets",
    min_value=100,
    max_value=50000,
    value=5000,
    step=100,
    help="Limite le nombre de tweets récupérés depuis la base de données."
)

# Filtre confidence
st.sidebar.subheader("🎯 Confiance du modèle")
min_confidence = st.sidebar.slider(
    "Confidence minimale",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.01,
    help="Filtre les tweets selon la confiance du modèle."
)

if st.sidebar.button("🔃 Actualiser les données"):
    st.cache_data.clear()
    st.rerun()

tweets_df = load_data(limit=tweet_limit)

st.sidebar.markdown("---")
if not tweets_df.empty:
    st.sidebar.info(f"✅ **DB Chargée :** {len(tweets_df)} tweets.")
else:
    st.sidebar.warning("❌ **DB Chargée :** 0 tweet.")

if tweets_df.empty:
    st.warning("📭 Aucune donnée disponible dans la base de données.")
    st.info("""
    **Pour commencer :**
    1. Exécutez `python main.py` pour insérer des données
    2. Vérifiez la connexion à PostgreSQL
    """)
    st.stop()

# Nettoyage/conversions
try:
    tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'].astype(str), errors='coerce', utc=True)
    tweets_df['retweet_count'] = pd.to_numeric(tweets_df.get('retweet_count', 0), errors='coerce').fillna(0).astype(int)
    tweets_df['favorite_count'] = pd.to_numeric(tweets_df.get('favorite_count', 0), errors='coerce').fillna(0).astype(int)
    tweets_df['confidence'] = pd.to_numeric(tweets_df.get('confidence', None), errors='coerce')
except Exception as e:
    st.error(f"❌ Erreur de conversion: {e}")
    st.stop()

# Couleurs Plotly
SENTIMENT_COLOR_MAP = {
    'positive': '#28A745',
    'neutral': '#FFC107',
    'negative': '#DC3545'
}

# Masque
mask = tweets_df['sentiment_label'].isin(sentiment_filter)

if not disable_date_filter:
    mask = mask & (tweets_df['created_at'].dt.date >= date_range[0]) & (tweets_df['created_at'].dt.date <= date_range[1])

if 'confidence' in tweets_df.columns:
    mask = mask & (tweets_df['confidence'].fillna(0) >= min_confidence)

if search_query:
    search_mask = tweets_df['text'].astype(str).str.contains(search_query, case=False, na=False) | \
                  tweets_df['clean_text'].astype(str).str.contains(search_query, case=False, na=False)
    mask = mask & search_mask

filtered_df = tweets_df[mask].copy()

if filtered_df.empty:
    st.warning("Aucun tweet ne correspond aux filtres sélectionnés.")
    st.sidebar.warning("❌ **Filtré :** 0 tweet.")
    st.stop()

st.sidebar.success(f"📈 **Filtré :** {len(filtered_df)} tweets.")

# Engagement + colonnes temps
filtered_df['total_engagement'] = filtered_df['retweet_count'] + filtered_df['favorite_count']
filtered_df['day'] = filtered_df['created_at'].dt.date
filtered_df['hour'] = filtered_df['created_at'].dt.hour

# KPIs
st.subheader("✨ Métriques Principales")
col1, col2, col3, col4, col5 = st.columns(5)

total_tweets = len(filtered_df)
positive_tweets = (filtered_df['sentiment_label'] == 'positive').sum()
neutral_tweets = (filtered_df['sentiment_label'] == 'neutral').sum()
negative_tweets = (filtered_df['sentiment_label'] == 'negative').sum()

with col1:
    st.metric("Total Tweets", total_tweets)
with col2:
    st.metric("👍 Positifs", positive_tweets)
with col3:
    st.metric("⚪ Neutres", neutral_tweets)
with col4:
    st.metric("👎 Négatifs", negative_tweets)
with col5:
    score_pos = (positive_tweets / total_tweets) * 100 if total_tweets else 0
    st.metric("🎯 Score Positif", f"{score_pos:.1f}%")

st.markdown("---")

# IA insight
with st.spinner("💡 L'IA analyse vos données..."):
    ai_insight = generate_ai_insight(search_query, total_tweets, positive_tweets, neutral_tweets, negative_tweets)
st.markdown(f'<div class="ai-card"><strong>💡 Analyse IA & Insights :</strong><br>{ai_insight}</div>', unsafe_allow_html=True)

# Ligne 1 graphes
c1, c2 = st.columns(2)

with c1:
    st.subheader("📈 Distribution des Sentiments")
    sentiment_counts = filtered_df['sentiment_label'].value_counts()
    fig_pie = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        color=sentiment_counts.index,
        color_discrete_map=SENTIMENT_COLOR_MAP
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with c2:
    st.subheader("🎯 Confiance par sentiment (Histogramme)")
    df_plot = filtered_df[['sentiment_label', 'confidence']].dropna()
    if not df_plot.empty:
        fig_hist = px.histogram(
            df_plot,
            x="confidence",
            color="sentiment_label",
            histnorm='percent',
            nbins=10,
            color_discrete_map=SENTIMENT_COLOR_MAP,
            labels={'confidence': 'Score de Confiance', 'percent': 'Pourcentage'}
        )
        fig_hist.update_layout(barmode='group')
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Pas de données de confiance.")

# Ligne 2 graphes
st.markdown("---")
c3, c4 = st.columns(2)

with c3:
    st.subheader("👤 Top 10 Utilisateurs les Plus Actifs")
    if 'user_name' in filtered_df.columns:
        top_users = filtered_df['user_name'].fillna("Unknown").value_counts().nlargest(10).reset_index()
        top_users.columns = ['user_name', 'tweet_count']
        fig_users = px.bar(
            top_users,
            x='tweet_count',
            y='user_name',
            orientation='h',
            title="Volume de Tweets par Utilisateur",
            template='plotly_white'
        )
        fig_users.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_users, use_container_width=True)
    else:
        st.info("Colonne user_name manquante.")

with c4:
    st.subheader("🗓️ Évolution Temporelle par Sentiment (Barres + Ligne)")
    daily_sentiment = filtered_df.groupby([filtered_df['day'], 'sentiment_label']).size().reset_index(name='count')

    if not daily_sentiment.empty:
        daily_sentiment['day'] = daily_sentiment['day'].astype(str)
        daily_total = daily_sentiment.groupby('day')['count'].sum().reset_index(name='total_count')

        fig_evo = go.Figure()
        for s, color in SENTIMENT_COLOR_MAP.items():
            df_s = daily_sentiment[daily_sentiment['sentiment_label'] == s]
            fig_evo.add_trace(go.Bar(x=df_s['day'], y=df_s['count'], name=s.capitalize(), marker_color=color, opacity=0.85))

        LINE_COLOR = '#007AFF'
        fig_evo.add_trace(go.Scatter(
            x=daily_total['day'], y=daily_total['total_count'],
            mode='lines+markers', name='Total',
            marker=dict(color=LINE_COLOR, size=8),
            line=dict(color=LINE_COLOR, width=3),
            yaxis='y2'
        ))

        fig_evo.update_layout(
            title="Volume Quotidien (Sentiment + Total)",
            xaxis=dict(type='category', title='Date'),
            yaxis=dict(title='Nombre de Tweets (Sentiment)', showgrid=False),
            yaxis2=dict(title='Total', overlaying='y', side='right', showgrid=False),
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_evo, use_container_width=True)
    else:
        st.info("Pas assez de données pour l'évolution temporelle.")

# --- NOUVEAUX GRAPHES ---
st.markdown("---")
st.subheader("🔥 Heatmap : Volume de Tweets par Jour et Sentiment")
heat_df = filtered_df.groupby(['day', 'sentiment_label']).size().reset_index(name='count')
heat_pivot = heat_df.pivot(index='sentiment_label', columns='day', values='count').fillna(0)
fig_heat = px.imshow(
    heat_pivot,
    aspect="auto",
    labels=dict(x="Jour", y="Sentiment", color="Nombre"),
    title="Heatmap Sentiment × Jour"
)
st.plotly_chart(fig_heat, use_container_width=True)

st.subheader("📦 Confiance du modèle par Sentiment (Boxplot)")
df_conf = filtered_df[['sentiment_label', 'confidence']].dropna()
if not df_conf.empty:
    fig_box_conf = px.box(
        df_conf,
        x='sentiment_label',
        y='confidence',
        color='sentiment_label',
        color_discrete_map=SENTIMENT_COLOR_MAP,
        title="Distribution de la confiance par sentiment"
    )
    st.plotly_chart(fig_box_conf, use_container_width=True)
else:
    st.info("Pas assez de données pour le boxplot de confiance.")

st.subheader("💬 Engagement par Sentiment (Boxplot + Moyenne)")
df_eng = filtered_df[['sentiment_label', 'total_engagement']].dropna()
if not df_eng.empty:
    fig_box_eng = px.box(
        df_eng,
        x='sentiment_label',
        y='total_engagement',
        color='sentiment_label',
        color_discrete_map=SENTIMENT_COLOR_MAP,
        title="Engagement (RT + Likes) par sentiment"
    )
    st.plotly_chart(fig_box_eng, use_container_width=True)

    eng_mean = df_eng.groupby('sentiment_label')['total_engagement'].mean().sort_values(ascending=False).reset_index()
    fig_eng_mean = px.bar(
        eng_mean,
        x='sentiment_label',
        y='total_engagement',
        color='sentiment_label',
        color_discrete_map=SENTIMENT_COLOR_MAP,
        title="Engagement moyen par sentiment",
        labels={'total_engagement': 'Engagement moyen'}
    )
    st.plotly_chart(fig_eng_mean, use_container_width=True)
else:
    st.info("Pas assez de données pour les graphes d'engagement.")

st.subheader("📌 Relation : Confiance vs Engagement (Scatter)")
scatter_df = filtered_df[['confidence', 'total_engagement', 'sentiment_label']].dropna()
if not scatter_df.empty:
    if len(scatter_df) > 5000:
        scatter_df = scatter_df.sample(5000, random_state=42)

    fig_scatter = px.scatter(
        scatter_df,
        x='confidence',
        y='total_engagement',
        color='sentiment_label',
        color_discrete_map=SENTIMENT_COLOR_MAP,
        opacity=0.6,
        title="Confiance du modèle vs Engagement",
        labels={'total_engagement': 'Engagement total'}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Pas assez de données pour le scatter.")
# ------------------------

# Table tweets
st.subheader("📝 Tweets Récents Analysés")
display_columns = st.multiselect(
    "Colonnes à afficher :",
    options=['created_at', 'user_name', 'clean_text', 'sentiment_label', 'confidence',
             'retweet_count', 'favorite_count', 'total_engagement'],
    default=['created_at', 'user_name', 'clean_text', 'sentiment_label', 'confidence']
)

if display_columns:
    def color_sentiment(row):
        colors = {
            'positive': 'background-color: #E6F7ED; color: black;',
            'negative': 'background-color: #FEE7E9; color: black;',
            'neutral': 'background-color: #FFF9E6; color: black;'
        }
        return [colors.get(row.get('sentiment_label', ''), '')] * len(row)

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