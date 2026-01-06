import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from src.preprocessing.text_cleaner import TextCleaner


def load_sentiment140(csv_path: str) -> pd.DataFrame:
    """
    Sentiment140 Kaggle: columns = [target, ids, date, flag, user, text]
    target: 0 = negative, 4 = positive
    """
    cols = ["target", "ids", "date", "flag", "user", "text"]
    df = pd.read_csv(csv_path, encoding="latin-1", names=cols)

    # garder uniquement 0 et 4
    df = df[df["target"].isin([0, 4])].copy()

    # map labels
    df["label"] = df["target"].map({0: "negative", 4: "positive"})
    return df[["text", "label"]]


def main():
    # 1) path dataset
    # Exemple: data/raw/data.csv
    dataset_path = os.getenv(
        "SENTIMENT140_PATH",
        "data/raw/data.csv"
    )

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset introuvable: {dataset_path}\n"
            f"Mets le fichier dans data/raw/ ou définis SENTIMENT140_PATH."
        )

    print("✅ Loading dataset...")
    df = load_sentiment140(dataset_path)
    print("Rows:", len(df))

    # 2) cleaning
    cleaner = TextCleaner()
    print("🧹 Cleaning text (this can take some minutes)...")
    df["clean"] = df["text"].astype(str).apply(cleaner.clean_text)

    # enlever vides / doublons
    df = df[df["clean"].str.strip().astype(bool)]
    df = df.drop_duplicates(subset=["clean"])

    X = df["clean"]
    y = df["label"]

    # 3) split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) model pipeline
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=200000
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                n_jobs=-1,
                class_weight="balanced"
            ))
        ]
    )

    print("🤖 Training...")
    model.fit(X_train, y_train)

    print("\n📌 Evaluation:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=["negative", "positive"]))

    # 5) save model
    os.makedirs("data/models", exist_ok=True)
    out_path = "data/models/sentiment140_tfidf_lr.joblib"
    joblib.dump(model, out_path)
    print(f"\n✅ Model saved to: {out_path}")


if __name__ == "__main__":
    main()