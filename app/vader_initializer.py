import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk  # Keep nltk import here for the download call


def initialize_vader():
    """Initializes the VADER sentiment analyzer, downloads lexicon if needed."""
    vader_lexicon_was_downloaded = False
    analyzer_instance = None
    try:
        analyzer_instance = SentimentIntensityAnalyzer()
    except LookupError:
        try:
            with st.spinner("Downloading VADER lexicon for sentiment analysis..."):
                nltk.download('vader_lexicon')
            analyzer_instance = SentimentIntensityAnalyzer()
            vader_lexicon_was_downloaded = True
        except Exception as e:
            st.error(
                f"CRITICAL: Failed to download/initialize VADER lexicon. Error: {e}. "
                "Sentiment features will be impaired. Please check internet/NLTK setup."
            )

    # UI Feedback for VADER download
    if analyzer_instance is None and not vader_lexicon_was_downloaded:
        st.warning(
            "Sentiment analyzer (VADER) could not be initialized. Sentiment scores will default to neutral (0.0).")
    elif vader_lexicon_was_downloaded and analyzer_instance is not None:
        st.success("VADER lexicon for NLTK Sentiment Analysis was successfully downloaded and initialized.")

    return analyzer_instance