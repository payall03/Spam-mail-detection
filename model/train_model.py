import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import os

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\₹|\$|[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def train_and_save_model():
    df = pd.read_csv("data/cleaned_spam.csv")

    # Convert labels if needed
    if df['label'].dtype == 'object':
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    df.dropna(subset=['text', 'label'], inplace=True)
    df['cleaned_text'] = df['text'].apply(clean_text)

    X = df['cleaned_text']
    y = df['label']

    # Better vectorizer settings
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=3000
    )
    X_vec = vectorizer.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, stratify=y, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained with accuracy: {acc * 100:.2f}%")

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/spam_classifier.pkl")
    joblib.dump(vectorizer, "model/vectorizer.pkl")
    print("✅ Model and vectorizer saved!")

if __name__ == "__main__":
    train_and_save_model()
