import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from preprocess import clean_text

# Ensure model folder exists
os.makedirs("../model", exist_ok=True)

# Load dataset
df = pd.read_csv("../data/IFND.csv", encoding='latin-1')

# Fix columns
df = df[['Statement', 'Label']]
df.columns = ['text', 'label']

# Fix labels
df['label'] = df['label'].str.lower()

# Clean text
df['clean_text'] = df['text'].apply(clean_text)
df = df[df['clean_text'] != ""]

# Features and labels
X = df['clean_text']
y = df['label']

# Convert text → numbers
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "../model/model.pkl")
joblib.dump(vectorizer, "../model/vectorizer.pkl")