import os
import pandas as pd
from tqdm import tqdm

from src.preprocessing.text_cleaner import TextCleaner
from src.nlp.sentiment_analyzer import SentimentAnalyzer
from src.database.db_manager import DatabaseManager


# =============================
# CONFIG
# =============================
DATASET_PATH = "data/raw/data.csv"
MAX_ROWS = 200000        # ⚠️ pour tests (mettre None pour tout le dataset)
SAVE_TO_DB = True       # mettre False si tu veux juste tester
SHOW_PROGRESS = True


# =============================
# LOAD DATASET
# =============================
def load_sentiment140(path, max_rows=None):
    """
    Sentiment140 format:
    target, id, date, flag, user, text
    target: 0 = negative, 4 = positive
    """
    cols = ["target", "ids", "date", "flag", "user", "text"]

    df = pd.read_csv(
        path,
        encoding="latin-1",
        names=cols,
        nrows=max_rows
    )

    # garder seulement 0 et 4
    df = df[df["target"].isin([0, 4])].copy()

    # mapping labels
    df["true_label"] = df["target"].map({
        0: "negative",
        4: "positive"
    })

    return df[["date", "user", "text", "true_label"]]


# =============================
# MAIN PIPELINE
# =============================
def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset introuvable : {DATASET_PATH}")

    print("📥 Chargement du dataset Sentiment140...")
    df = load_sentiment140(DATASET_PATH, MAX_ROWS)
    print(f"Tweets chargés : {len(df)}")

    cleaner = TextCleaner()
    analyzer = SentimentAnalyzer()
    db = DatabaseManager()

    results = []

    iterator = tqdm(df.iterrows(), total=len(df)) if SHOW_PROGRESS else df.iterrows()

    print("🧠 Analyse des sentiments en cours...")
    for _, row in iterator:
        raw_text = str(row["text"])
        clean_text = cleaner.clean_text(raw_text)

        if not clean_text.strip():
            continue

        analysis = analyzer.analyze_sentiment(clean_text)

        results.append({
            "created_at": row["date"],
            "user_name": row["user"],
            "text": raw_text,
            "clean_text": clean_text,
            "true_label": row["true_label"],
            "sentiment_label": analysis["label"],
            "confidence": analysis["confidence"],
            "sentiment_score": analysis["sentiment_score"]
        })

    results_df = pd.DataFrame(results)
    print("✅ Analyse terminée")

    # =============================
    # EVALUATION GLOBALE
    # =============================
    print("\n📊 Évaluation globale :")

    comparable = results_df[results_df["sentiment_label"] != "neutral"]

    if not comparable.empty:
        accuracy = (
            comparable["sentiment_label"] == comparable["true_label"]
        ).mean()

        print(f"Accuracy (sans neutral) : {accuracy:.4f}")
    else:
        print("⚠️ Aucun tweet comparable (trop de neutral)")

    print("\nDistribution des prédictions :")
    print(results_df["sentiment_label"].value_counts(normalize=True))

    # =============================
    # SAVE TO DATABASE
    # =============================
    if SAVE_TO_DB:
        print("\n💾 Sauvegarde dans la base de données...")
        for _, r in results_df.iterrows():
            db.insert_tweet(
                text=r["text"],
                clean_text=r["clean_text"],
                sentiment_label=r["sentiment_label"],
                confidence=r["confidence"],
                sentiment_score=r["sentiment_score"],
                user_name=r["user_name"],
                created_at=r["created_at"],
                true_label=r["true_label"]
            )
        print("✅ Données enregistrées dans la DB")

    print("\n🎉 Pipeline terminé avec succès")


# =============================
if __name__ == "__main__":
    main()