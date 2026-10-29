"""Train Radian AI's own fake news detection model.

Dataset: Kaggle "Fake and Real News Dataset".
Put Fake.csv and True.csv inside a folder named  data/  next to this file.

Run:     python train_model.py
Output:  fake_news_model.joblib  (used by predict.py and the app)
"""
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def clean(text):
    """Normalise text so the model learns content, not shortcuts."""
    text = str(text)
    # Every real article in this dataset starts like "WASHINGTON (Reuters) -".
    # If we keep it, the model just learns "Reuters = real" and cheats.
    text = re.sub(r"^.{0,120}?\(Reuters\)\s*-\s*", "", text)
    text = text.replace("Reuters", "")
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # remove links
    text = re.sub(r"[^a-z0-9\s]", " ", text)             # remove symbols
    return re.sub(r"\s+", " ", text).strip()


def load_data():
    fake = pd.read_csv("data/Fake.csv")
    real = pd.read_csv("data/True.csv")
    fake["label"] = 1   # 1 = fake
    real["label"] = 0   # 0 = real
    df = pd.concat([fake, real], ignore_index=True)
    df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).map(clean)
    df = df[df["content"].str.len() > 20].drop_duplicates("content")
    return df


def build_model(classifier):
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7, min_df=2,
                                  ngram_range=(1, 2), max_features=200_000)),
        ("clf", classifier),
    ])


def main():
    df = load_data()
    print(f"Loaded {len(df)} articles: {df.label.sum()} fake, {(df.label == 0).sum()} real\n")

    # 80% for training, 20% kept aside for testing (never seen during training)
    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(),
    }

    best_name, best_model, best_acc = None, None, 0.0
    for name, clf in candidates.items():
        model = build_model(clf)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"{name:20s} accuracy: {acc:.2%}")
        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc

    print(f"\nBest model: {best_name} ({best_acc:.2%})\n")
    preds = best_model.predict(X_test)
    print(classification_report(y_test, preds, target_names=["Real", "Fake"]))

    cm = confusion_matrix(y_test, preds)
    print("Confusion matrix:")
    print(f"                 Predicted Real   Predicted Fake")
    print(f"Actually Real    {cm[0][0]:>14}   {cm[0][1]:>14}")
    print(f"Actually Fake    {cm[1][0]:>14}   {cm[1][1]:>14}")

    joblib.dump(best_model, "fake_news_model.joblib")
    print("\nSaved trained model to fake_news_model.joblib")


if __name__ == "__main__":
    main()