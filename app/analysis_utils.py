# WorkloadSentimentMonitor/app/analysis_utils.py
import pandas as pd

# This module-level variable will be set by app.py
_vader_analyzer = None


def set_analyzer_instance(analyzer):
    """Sets the VADER analyzer instance for use within this module."""
    global _vader_analyzer
    _vader_analyzer = analyzer


def calculate_workload_metric(tasks_assigned, tasks_overdue):
    # This function remains unchanged
    if tasks_assigned == 0:
        return 0
    overdue_penalty = (tasks_overdue / tasks_assigned) * 10
    base_score = min(tasks_assigned / 2, 5)
    return min(base_score + overdue_penalty, 10)


def get_sentiment_score(text):
    # This function remains unchanged
    if not _vader_analyzer:
        return 0.0
    return _vader_analyzer.polarity_scores(text)['compound']


def calculate_burnout_risk(workload_score, sentiment_score, text):
    # This function remains unchanged
    sentiment_used_for_risk = sentiment_score
    if "urgent" in text.lower() or "asap" in text.lower() or "overwhelmed" in text.lower():
        sentiment_used_for_risk = max(-0.9, sentiment_score - 0.5)

    risk_score = (workload_score * 0.6) + ((1 - (sentiment_used_for_risk + 1) / 2) * 10 * 0.4)
    risk_level, color, recommendation = "Low", "#28a745", "No immediate action needed. Continue standard monitoring."
    if risk_score > 7:
        risk_level, color, recommendation = "High", "#dc3545", "High burnout risk indicated. RECOMMENDATION: Immediate 1:1 check-in required. Discuss workload, well-being, offer support resources, and actively consider task redistribution or deadline adjustments."
    elif risk_score > 4:
        risk_level, color, recommendation = "Medium", "#ffc107", "Medium burnout risk. RECOMMENDATION: Proactively check in during the next team meeting. Monitor workload trends and offer support."
    return risk_level, color, recommendation, sentiment_used_for_risk


# --- NEW: Function to get overall sentiment distribution ---
def _get_vader_category(text):
    """Helper to categorize a single text using VADER."""
    score = get_sentiment_score(text)
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def get_sentiment_distribution(df, model_name, classifier_model=None):
    """
    Calculates the distribution of sentiments for the entire DataFrame.

    Args:
        df (pd.DataFrame): The full DataFrame of employee reviews.
        model_name (str): The name of the model to use ("Vader Model" or "Classification Model").
        classifier_model: The loaded scikit-learn pipeline model.

    Returns:
        pd.Series: A pandas Series with sentiment counts (e.g., Positive: 50, Negative: 20).
    """
    if df.empty:
        return pd.Series(dtype='int64')

    if model_name == "Vader Model":
        sentiments = df['communication'].apply(_get_vader_category)
        return sentiments.value_counts()

    elif model_name == "Classification Model":
        if classifier_model:
            predictions = classifier_model.predict(df['communication'])
            return pd.Series(predictions).value_counts()
        else:
            return pd.Series(dtype='int64')  # Return empty series if model not loaded

    return pd.Series(dtype='int64')
# --- END NEW ---