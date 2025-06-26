# WorkloadSentimentMonitor/app/classification_train_model.py

import pandas as pd
import numpy as np
import os
import re
import joblib
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# For reproducibility
import random

random.seed(42)
np.random.seed(42)

# Scikit-learn for machine learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # MODIFIED: Import a more robust classifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# --- Download VADER lexicon if not present ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    print("Downloading VADER lexicon...");
    nltk.download('vader_lexicon')

# --- Configuration Constants ---
DATA_DIR = os.path.join('..', 'data');
TRAIN_CSV_FILENAME = 'employee_analysis_train_dataset.csv'
DATA_FILE_PATH = os.path.join(DATA_DIR, TRAIN_CSV_FILENAME)
MODEL_FILENAME = 'sentiment_classifier.pkl';
MODEL_SAVE_PATH = MODEL_FILENAME
REPORT_FILENAME = 'classification_report.json';
REPORT_SAVE_PATH = REPORT_FILENAME
TEXT_COLUMN = 'text';
TARGET_COLUMN = 'sentiment'


def get_vader_category(text, analyzer):
    score = analyzer.polarity_scores(text)['compound']
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def main():
    print(f"Loading data from '{DATA_FILE_PATH}'...")
    df = pd.read_csv(DATA_FILE_PATH);
    df.dropna(subset=[TEXT_COLUMN, TARGET_COLUMN], inplace=True)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

    X, y = df[TEXT_COLUMN], df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    report_data = {}

    # --- MODIFIED: Train a BALANCED Classification Model ---
    print("\n--- Evaluating: Balanced Classification Model (Logistic Regression) ---")
    model_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english')),
        # Use Logistic Regression with class_weight='balanced' to handle skewed data
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])

    # Define a new grid of parameters for Logistic Regression
    parameters = {
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'classifier__C': [0.1, 1, 10],  # C is the inverse of regularization strength
    }

    grid_search = GridSearchCV(model_pipeline, parameters, cv=5, n_jobs=-1, scoring='f1_weighted')

    print("Starting grid search to find the best balanced model parameters...")
    grid_search.fit(X_train, y_train)

    print("\nBest parameters for the balanced model:");
    print(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    y_pred_classifier = best_model.predict(X_test)
    class_labels = best_model.classes_

    report_str_classifier = classification_report(y_test, y_pred_classifier, labels=class_labels)
    cm_classifier = confusion_matrix(y_test, y_pred_classifier, labels=class_labels)

    report_data['Classification Model'] = {
        "classification_report": report_str_classifier,
        "confusion_matrix": cm_classifier.tolist(),
        "confusion_matrix_labels": class_labels.tolist()
    }
    print("\nPerformance of the NEW BALANCED model on the test set:")
    print(report_str_classifier)

    # --- Part 2: Evaluate the VADER Model (No change here) ---
    print("\n--- Evaluating: Vader Model ---")
    vader_analyzer = SentimentIntensityAnalyzer()
    y_pred_vader = X_test.apply(lambda text: get_vader_category(text, vader_analyzer))
    report_str_vader = classification_report(y_test, y_pred_vader, labels=class_labels, zero_division=0)
    cm_vader = confusion_matrix(y_test, y_pred_vader, labels=class_labels)
    report_data['Vader Model'] = {"classification_report": report_str_vader, "confusion_matrix": cm_vader.tolist(),
                                  "confusion_matrix_labels": class_labels.tolist()}

    # --- Part 3: Save all reports and the NEW BEST model ---
    print(f"\nSaving combined performance report to '{REPORT_SAVE_PATH}'...")
    with open(REPORT_SAVE_PATH, 'w') as f: json.dump(report_data, f, indent=4)
    print("Performance report saved successfully.")

    print(f"\nSaving BEST trained classifier to '{MODEL_SAVE_PATH}'...")
    joblib.dump(best_model, MODEL_SAVE_PATH)
    print("Model saved successfully in the 'app' folder.")


if __name__ == "__main__":
    main()