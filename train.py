# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from preprocess import clean_tweet

# Load dataset
df = pd.read_csv("tweets.csv")
print("Dataset loaded:", df.shape)
print("Columns available:", df.columns)

# Rename comment → tweet
df.rename(columns={'comment': 'tweet'}, inplace=True)

# Keep only tweet + label
df = df[['tweet', 'label']].dropna()

# Use only a small sample (for faster training)
df = df.sample(3000, random_state=42).reset_index(drop=True)
print("Using subset of data:", df.shape)

# Clean tweets
print("🧹 Cleaning tweets...")
df['cleaned'] = df['tweet'].apply(clean_tweet)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# TF-IDF Vectorization
print("🔤 Converting to TF-IDF...")
vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
print("🤖 Training model...")
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "sarcasm_model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")
print("\n✅ Model and vectorizer saved successfully!")
