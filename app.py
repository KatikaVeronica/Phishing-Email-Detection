import re
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

print("Starting Phishing Email Detection Model...")

# -------------------------------------------------------------
# 1. Feature Extractor
# -------------------------------------------------------------
class EmailStructuralFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.suspicious_keywords = [
            "urgent", "verify", "suspended", "account", "login",
            "password", "security", "update", "bank", "free", "action required"
        ]

    def fit(self, X, y=None):
        return self

    def _extract_features(self, text):
        text_lower = text.lower()
        url_count = len(re.findall(r"https?://\S+|www\.\S+", text))
        ip_url_count = len(re.findall(r"https?://\d{1,3}(?:\.\d{1,3}){3}", text))
        keyword_hits = sum(text_lower.count(kw) for kw in self.suspicious_keywords)
        exclamation_count = text.count("!")
        words = text.split()
        caps_count = sum(1 for w in words if w.isupper() and len(w) > 1)
        length = len(text)
        return [url_count, ip_url_count, keyword_hits, exclamation_count, caps_count, length]

    def transform(self, X):
        return np.array([self._extract_features(text) for text in X])

# -------------------------------------------------------------
# 2. Dataset
# -------------------------------------------------------------
data = {
    "text": [
        "URGENT: Your bank account is locked! Click http://192.168.1.1/login to verify your password immediately.",
        "Hey team, attached are the meeting notes and slide deck from this morning's retrospective.",
        "Security Alert! Unauthorized login attempt detected. Go to http://secure-update-portal.com to reset credentials.",
        "Can we reschedule our sync to 3:00 PM tomorrow? Let me know if that works for you.",
        "Congratulations! You won a $1,000 gift card. Claim your reward now at http://bit.ly/claim-prize-free",
        "Hi Sarah, please find the Q3 financial audit report attached for your review.",
        "Action required: Your mailbox storage is full. Verify your account at http://mail-verify-alert.net to prevent suspension.",
        "Don't forget to submit your expense reports by end of day Friday.",
        "FINAL NOTICE: Wire transfer failed. Update payment details immediately at http://bank-wire-sec.com",
        "Thanks for the coffee catch-up! Sending over the documentation we talked about."
    ] * 20,
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 20
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
)

# -------------------------------------------------------------
# 3. Pipeline & Training
# -------------------------------------------------------------
pipeline = Pipeline([
    ("features", FeatureUnion([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)),
        ("heuristics", EmailStructuralFeatureExtractor())
    ])),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

print("Training the model...")
pipeline.fit(X_train, y_train)

# -------------------------------------------------------------
# 4. Evaluation
# -------------------------------------------------------------
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")
print("Confusion Matrix:")
print(pd.DataFrame(cm, index=["Actual: Safe", "Actual: Phishing"], columns=["Predicted: Safe", "Predicted: Phishing"]))

# -------------------------------------------------------------
# 5. Save the Model
# -------------------------------------------------------------
joblib.dump(pipeline, "phishing_detector_model.pkl")
print("\n[+] Model saved to disk as 'phishing_detector_model.pkl'!")

def classify_email(email_content: str):
    pred = pipeline.predict([email_content])[0]
    prob = pipeline.predict_proba([email_content])[0]
    status = "Phishing" if pred == 1 else "Safe"
    confidence = prob[pred] * 100
    return f"Classification: {status} ({confidence:.1f}% confidence)"

# -------------------------------------------------------------
# 6. Interactive Live Tester
# -------------------------------------------------------------
print("\n" + "="*50)
print(" LIVE EMAIL CHECKER")
print(" Type or paste an email and hit Enter.")
print(" Type 'exit' and hit Enter when you want to stop.")
print("="*50)

while True:
    user_input = input("\nEnter email: ")
    if user_input.strip().lower() == "exit":
        print("Exiting checker. Goodbye!")
        break
    if user_input.strip():
        print("--> " + classify_email(user_input))