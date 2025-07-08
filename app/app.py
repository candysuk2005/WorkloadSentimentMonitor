# ==============================================================================
# FINAL SIDEBAR CODE - COPY AND PASTE THIS ENTIRE BLOCK INTO YOUR APP
# ==============================================================================
import streamlit as st
import pandas as pd # Make sure pandas is imported

# --- Sidebar Configuration ---
st.sidebar.title("About this Dashboard")

st.sidebar.info(
    "This dashboard is a Proof of Concept for using sentiment analysis "
    "to help identify potential employee burnout."
)

# --- CRITICAL FIX: Hide the raw metrics in an expander ---
with st.sidebar.expander("Show Baseline Model Performance"):
    st.write("VADER Model Performance (Baseline)")
    
    # --- IMPORTANT ---
    # Replace these placeholders with your actual DataFrames for the
    # classification report and confusion matrix.
    # For example:
    # report_df = pd.DataFrame(your_classification_report)
    # confusion_df = pd.DataFrame(your_confusion_matrix, columns=['Negative', 'Neutral', 'Positive'])

    # --- Placeholder DataFrames (REPLACE WITH YOURS) ---
    report_data = {
        'precision': {'Negative': 0.00, 'Neutral': 0.00, 'Positive': 0.08, 'accuracy': 0.08, 'macro avg': 0.03, 'weighted avg': 0.01},
        'recall': {'Negative': 0.00, 'Neutral': 0.00, 'Positive': 1.00, 'accuracy': 0.08, 'macro avg': 0.33, 'weighted avg': 0.08},
        'f1-score': {'Negative': 0.00, 'Neutral': 0.00, 'Positive': 0.15, 'accuracy': 0.08, 'macro avg': 0.05, 'weighted avg': 0.01},
        'support': {'Negative': 14.0, 'Neutral': 9.0, 'Positive': 2.0, 'accuracy': 0.08, 'macro avg': 25.0, 'weighted avg': 25.0}
    }
    confusion_data = {
        'Negative': {'Negative': 0, 'Neutral': 0, 'Positive': 0},
        'Neutral': {'Negative': 0, 'Neutral': 0, 'Positive': 0},
        'Positive': {'Negative': 14, 'Neutral': 9, 'Positive': 2}
    }
    report_df = pd.DataFrame(report_data)
    confusion_df = pd.DataFrame(confusion_data)
    # --- End of Placeholder DataFrames ---

    st.dataframe(report_df)
    st.write("**Confusion Matrix**")
    st.dataframe(confusion_df)
    st.caption("Rows: Actual, Columns: Predicted")


# --- STRATEGIC PIVOT: Explain the metrics with a new narrative ---
st.sidebar.subheader("Model Strategy & Evaluation")

st.sidebar.markdown(
    """
    **1. Baseline Model Performance (VADER)**

    The metrics shown above are for a standard, off-the-shelf VADER model. As a simple, rule-based model not trained on our specific business context (employee burnout), its performance is expectedly low. It serves as a crucial **baseline** to measure future improvements against.

    **2. Evaluation Strategy for a Custom Model**

    For this specific business problem, a simple accuracy score is misleading. Our primary goal is to **identify every employee who might be at risk**, because the cost of missing someone (a "False Negative") is far higher than the cost of flagging a healthy employee for a check-in (a "False Positive").

    Therefore, our development strategy prioritizes:

    -   **High Recall (Negative Class):** This is our North Star metric. We need to find the highest possible percentage of *all actual* negative/burnout-risk reviews. A custom model would be trained and tuned specifically to maximize this.
    -   **Precision as a Secondary Metric:** While secondary, we would still monitor precision to ensure the tool remains useful and doesn't create excessive "alarm fatigue" for managers.
    """
)

st.sidebar.caption("Developed as a PoC for AI for Business Capstone.")

# ==============================================================================
# END OF SIDEBAR CODE
# ==============================================================================
