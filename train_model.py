
import os
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Create model folder
os.makedirs("model", exist_ok=True)

# Load UCI dataset
data = pd.read_csv(
    "dataset/spam.csv",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

# Clean dataset
data = data.dropna()
data["label"] = data["label"].str.strip().str.lower()
data["message"] = data["message"].astype(str).str.strip()

data = data[data["label"].isin(["ham", "spam"])]
data = data[data["message"] != ""]

# Convert labels
data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data["message"],
    data["label"],
    test_size=0.20,
    random_state=42,
    stratify=data["label"]
)

# Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )),
    ("classifier", MultinomialNB(alpha=0.5))
])

# Train model
print("Training model...")
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Calculate performance
metrics = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(precision_score(y_test, predictions, zero_division=0)),
    "recall": float(recall_score(y_test, predictions, zero_division=0)),
    "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
    "confusion_matrix": confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).tolist(),
    "total_messages": int(len(data)),
    "training_messages": int(len(X_train)),
    "testing_messages": int(len(X_test))
}

# Save model
joblib.dump(model, "model/spam_model.pkl")

# Save evaluation results
with open("model/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

# Print results
print("\nMODEL TRAINING COMPLETED!")
print("Accuracy:", round(metrics["accuracy"] * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["HAM", "SPAM"],
    zero_division=0
))

print("\nModel saved successfully!")
