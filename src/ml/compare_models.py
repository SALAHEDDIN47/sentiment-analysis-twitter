# src/ml/compare_models.py
# Comparaison de modèles (Sentiment140) : Logistic Regression vs Linear SVM vs Naive Bayes
# Sorties:
# - data/models/model_comparison.csv  (résumé)
# - data/models/model_reports.txt     (rapports détaillés + matrices)
#
# ⚙️ Réglages PC faible:
#   MAX_SAMPLES = 50000
#   MAX_FEATURES = 3000

import os
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

from src.preprocessing.text_cleaner import TextCleaner

DATASET_PATH = "data/raw/data.csv"

# ----- Réglages performance -----
MAX_SAMPLES = 200_000     # Mets 50_000 si PC lent
MAX_FEATURES = 5000       # Mets 3000 si RAM faible
NGRAM_RANGE = (1, 2)      # 1-gram + 2-gram (souvent meilleur)
MIN_DF = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42
# -------------------------------


def load_sentiment140(path: str) -> pd.DataFrame:
    """
    Charge Sentiment140 (Kaggle) renommé en data.csv.
    Format d'origine:
    target,id,date,flag,user,text
    target: 0 = négatif, 4 = positif
    """
    cols = ["target", "id", "date", "flag", "user", "text"]
    df = pd.read_csv(path, encoding="latin-1", names=cols)

    df = df[df["target"].isin([0, 4])].copy()
    df["label"] = df["target"].map({0: "negative", 4: "positive"})

    if MAX_SAMPLES and len(df) > MAX_SAMPLES:
        df = df.sample(MAX_SAMPLES, random_state=RANDOM_STATE)

    return df


def ensure_dirs():
    os.makedirs("data/models", exist_ok=True)


def main():
    ensure_dirs()

    print("📥 Chargement dataset...")
    df = load_sentiment140(DATASET_PATH)
    print("✅ Rows:", len(df))

    print("🧹 Nettoyage NLP...")
    cleaner = TextCleaner()
    df["clean_text"] = df["text"].astype(str).apply(cleaner.clean_text)

    X = df["clean_text"]
    y = df["label"]

    print("✂️ Split train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("🔠 TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Modèles comparés (classiques & légitimes)
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "LinearSVC": LinearSVC(),
        "MultinomialNB": MultinomialNB()
    }

    results = []
    detailed_lines = []

    for name, model in models.items():
        print("\n" + "=" * 60)
        print(f"🤖 Modèle: {name}")
        print("=" * 60)

        # Temps d'entraînement
        t0 = time.time()
        model.fit(X_train_vec, y_train)
        train_time = time.time() - t0

        # Temps de prédiction
        t1 = time.time()
        y_pred = model.predict(X_test_vec)
        pred_time = time.time() - t1

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)

        cm = confusion_matrix(y_test, y_pred)

        print(f"✅ Accuracy:  {acc:.4f}")
        print(f"✅ F1_macro:  {f1m:.4f}")
        print(f"✅ Precision: {prec:.4f}")
        print(f"✅ Recall:    {rec:.4f}")
        print(f"⏱️ Train time: {train_time:.2f}s | Predict time: {pred_time:.2f}s")
        print("📌 Confusion matrix:")
        print(cm)
        print("📌 Classification report:")
        print(classification_report(y_test, y_pred))

        results.append({
            "model": name,
            "accuracy": acc,
            "f1_macro": f1m,
            "precision_macro": prec,
            "recall_macro": rec,
            "train_time_sec": train_time,
            "predict_time_sec": pred_time
        })

        detailed_lines.append("\n" + "=" * 60)
        detailed_lines.append(f"MODEL: {name}")
        detailed_lines.append("=" * 60)
        detailed_lines.append(f"Accuracy: {acc:.4f}")
        detailed_lines.append(f"F1_macro: {f1m:.4f}")
        detailed_lines.append(f"Precision_macro: {prec:.4f}")
        detailed_lines.append(f"Recall_macro: {rec:.4f}")
        detailed_lines.append(f"Train_time_sec: {train_time:.2f}")
        detailed_lines.append(f"Predict_time_sec: {pred_time:.2f}")
        detailed_lines.append("Confusion matrix:")
        detailed_lines.append(str(cm))
        detailed_lines.append("Classification report:")
        detailed_lines.append(classification_report(y_test, y_pred))

    # Résumé final
    results_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    print("\n📊 Résumé comparaison (trié par F1_macro):")
    print(results_df)

    # Sauvegardes
    results_df.to_csv("data/models/model_comparison.csv", index=False)
    with open("data/models/model_reports.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(detailed_lines))

    print("\n✅ Fichiers générés :")
    print(" - data/models/model_comparison.csv")
    print(" - data/models/model_reports.txt")
    print("\n📸 Capture d'écran à faire :")
    print(" - le tableau résumé ci-dessus")
    print(" - une matrice de confusion + report pour chaque modèle")


if __name__ == "__main__":
    main()