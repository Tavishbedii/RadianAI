"""Use Radian AI's trained model to check new text.

Run:  python predict.py   (then type or paste a headline/message)
"""
import joblib
from train_model import clean

model = joblib.load("fake_news_model.joblib")


def predict(text):
    """Return (verdict, confidence). Confidence is None for models without probabilities."""
    cleaned = clean(text)
    label = model.predict([cleaned])[0]
    verdict = "Likely fake" if label == 1 else "Likely real"
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba([cleaned])[0].max()
    return verdict, confidence


if __name__ == "__main__":
    print("Radian AI - offline model. Press Enter on an empty line to quit.\n")
    while True:
        text = input("Paste a headline or message: ").strip()
        if not text:
            break
        verdict, confidence = predict(text)
        if confidence is not None:
            print(f"  -> {verdict} ({confidence:.0%} confident)\n")
        else:
            print(f"  -> {verdict}\n")