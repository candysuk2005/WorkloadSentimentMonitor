The key features of Workload Sentiment Monitor project
-------------------------------------------------------
External Data Ingestion & Preprocessing:
Loads employee review data from an external CSV file (employee_reviews.csv).
Performs significant data cleaning, including text normalization (lowercase, symbol/URL removal), and translation of non-English text to prepare it for analysis. This processed data is saved (e.g., as employee_reviews_PROCESSED.csv).

Custom NLP Sentiment Analysis Model Training & Application:
A custom sentiment analysis model is trained using scikit-learn on a manually labeled subset of employee review texts (specifically the 'pros&cons' sections).
This trained model is then applied to predict sentiment (Positive, Negative, Neutral) from the preprocessed employee review texts.

Contextual Sentiment Refinement (Keyword Spotting):
The sentiment predictions are further refined using a rule-based keyword spotting mechanism to identify critical phrases indicative of high stress or burnout that might be subtly expressed. This adjusts sentiment scores for risk calculation.

Burnout Risk Scoring:
A combined burnout risk score is generated for each review. This score synthesizes:
The (adjusted) sentiment derived from the review text.
Simulated workload metrics (e.g., tasks assigned, tasks overdue) to proxy operational pressures.
The risk is categorized (e.g., Low, Medium, High) based on predefined rules.

Interactive Dashboard & Visualization:
A Streamlit application provides an interactive dashboard with charts for managers.
It visualizes the sentiment analysis results, simulated workload indicators, and the calculated burnout risk level for each employee review.
Actionable insights or recommendations are provided based on the assessed risk.

User Interface with Pagination:
The dashboard incorporates pagination to allow users to navigate efficiently through a large number of employee reviews and their corresponding analyses.

How is the Burnout Risk Score Calculated? Think of it like a simple equation:

High Workload + Negative Feelings = High Burnout Risk

We measure two things for each employee review:
Workload: Are they overloaded with tasks? (This makes up 60% of the score).
Sentiment: How do they sound in their written comments? Are they positive or negative? (This makes up 40% of the score).
The system combines these two factors to produce a single risk level: Low, Medium, or High, each with a recommended action.

Custom Sentiment Model: Training & Application (`train_sentiment_model.py`)
-------------------------------------------------------
This script builds a custom model to predict employee sentiment using `scikit-learn`.
The model is trained on a small set of manually labeled 'pros&cons' text from employee reviews. It is then applied to a larger, preprocessed dataset of employee review texts (using combined 'summary' and 'pros&cons') to predict sentiment, which is then qualitatively assessed.

Workflow:

1.  Training Data:
    *   Loads ~100 manually labeled employee reviews from `data/employee_analysis_train_dataset.csv`.
    *   Input features: 'text' column (derived from 'pros&cons').
    *   Target labels: 'sentiment' column ("Positive", "Negative", "Neutral") representing employee sentiment.

2.  Application Data (for Sentiment Prediction):
    *   Loads ~1000 preprocessed (cleaned/translated) employee reviews from `data/employee_reviews_PROCESSED.csv`.
    *   The model will predict employee sentiment for text combined from relevant columns in this file (e.g., from 'summary_translated' and 'pros&cons_translated' columns).

3.  Process:
    *   Text from both datasets is cleaned (lowercase, remove URLs/non-alphanumerics).
    *   A TF-IDF vectorizer is FIT ONLY on the 100 training texts ('pros&cons' derived) and then USED TO TRANSFORM texts from both datasets.
    *   A Logistic Regression model is TRAINED ONLY on the 100 manually labeled training records to learn patterns of employee sentiment.

4.  Application & Output:
    *   The trained model PREDICTS employee sentiment (Positive, Negative, Neutral) for the combined text of the 1000 application records.
    *   The script outputs a sample of these predicted sentiments alongside the input text for qualitative assessment.
    *   The trained model and TF-IDF vectorizer are saved to the `trained_model/` directory as `.pkl` files.

Note: The primary goal is to demonstrate the model training pipeline using high-quality labels derived from 'pros&cons' text to understand employee sentiment, and then to observe the model's sentiment predictions when applied to a broader set of employee review texts.

How to Run This Project:
--------------------------
pip install -r requirements.txt
Under your working directory, WorkloadSentimentMonitor > python -m streamlit run app/app.py

Links and Websites for Reference:
--------------------------
External data from kaggle (Employee reviews with star ratings):
https://www.kaggle.com/datasets/malavikashamesh/employee-reviews-with-star-ratings?resource=download
NLP (General):
Wikipedia - Natural Language Processing: A good overview.
https://en.wikipedia.org/wiki/Natural_language_processing
Stanford NLP Group: A leading research group with many resources.
https://nlp.stanford.edu/
NLTK (Natural Language Toolkit) Book/Website: The library VADER is part of, with extensive documentation and tutorials on various NLP tasks.
https://www.nltk.org/
https://www.nltk.org/book/ (Online book for learning NLP with NLTK)
VADER:
Original VADER Paper (Highly Recommended): "VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text" by C.J. Hutto and Eric Gilbert. Searching this title will lead you to the PDF. This is the most authoritative source.
A direct link often found: https://ojs.aaai.org/index.php/ICWSM/article/view/14550/14399 (This is from the ICWSM conference proceedings).
NLTK's VADER Module Documentation:
https://www.nltk.org/api/nltk.sentiment.vader.html
You can also find usage examples by searching "NLTK VADER sentiment analysis tutorial" on Google. Many blogs and sites (like GeeksforGeeks, Towards Data Science) have practical examples.
GitHub Repository for VADER (often referenced by NLTK):
https://github.com/cjhutto/vaderSentiment (This contains the lexicon and source code if you want to dig really deep).