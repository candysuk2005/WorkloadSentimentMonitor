# WorkloadSentimentMonitor/app/app.py

import sys
import os
import joblib
import json
import streamlit as st
import pandas as pd

# --- Add 'app' directory and project root to sys.path ---
_current_script_dir = os.path.dirname(os.path.abspath(__file__))
if _current_script_dir not in sys.path:
    sys.path.insert(0, _current_script_dir)
_project_root = os.path.dirname(_current_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# --- End of path modification ---

from field_extraction.extract_fields import load_employee_data
import vader_initializer
import analysis_utils
import ui_components

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Employee Burnout Risk PoC")

# --- Session State Initialization ---
if 'current_page' not in st.session_state: st.session_state.current_page = 0
if 'selected_model' not in st.session_state: st.session_state.selected_model = "Classification Model"

# --- Model and Analyzer Initialization ---
analyzer = vader_initializer.initialize_vader()
analysis_utils.set_analyzer_instance(analyzer)


@st.cache_resource
def load_classifier_model():
    model_path = os.path.join(_current_script_dir, 'sentiment_classifier.pkl')
    try:
        model = joblib.load(model_path); print("Classification model loaded successfully."); return model
    except FileNotFoundError:
        print(f"Error: sentiment_classifier.pkl not found at {model_path}"); return None


@st.cache_data
def load_report_data():
    report_path = os.path.join(_current_script_dir, 'classification_report.json')
    try:
        with open(report_path, 'r') as f:
            report = json.load(f)
        print("Performance report loaded successfully.");
        return report
    except FileNotFoundError:
        print(f"Error: classification_report.json not found at {report_path}"); return None


classifier_model = load_classifier_model()
report_data = load_report_data()

# --- Main Application UI ---
st.title("AI for Business Capstone Project: Employee Burnout Risk PoC")
st.markdown("A Proof of Concept for analyzing workload and sentiment to identify potential employee burnout.")

# The sidebar display must come first
ui_components.display_sidebar_info(report_data)

employee_df_full = load_employee_data()

if employee_df_full.empty:
    st.error("Failed to load or process data. Check logs and file paths.");
    st.stop()
else:
    # --- MODIFIED: Pie chart is now DYNAMIC again based on the selected model ---
    @st.cache_data
    def get_cached_sentiment_counts(_df, model_name, _classifier_model):
        print(f"Calculating full sentiment distribution for {model_name}...")
        return analysis_utils.get_sentiment_distribution(_df, model_name, _classifier_model)


    # The call is now dynamic, using the selection from the sidebar
    sentiment_counts = get_cached_sentiment_counts(employee_df_full, st.session_state.selected_model, classifier_model)

    ui_components.display_sentiment_distribution_chart(sentiment_counts)
    st.markdown("---")
    # --- END MODIFIED ---

    st.success(f"Successfully loaded data for {len(employee_df_full)} reviews.")
    st.markdown("---")
    total_items = len(employee_df_full)
    start_index = st.session_state.current_page * ui_components.ITEMS_PER_PAGE
    end_index = min(start_index + ui_components.ITEMS_PER_PAGE, total_items)
    employee_df_page = employee_df_full.iloc[start_index:end_index]
    with st.expander("View Raw Employee Data (Current Page)"):
        st.dataframe(employee_df_page)
    st.markdown("---")
    if not employee_df_page.empty:
        for index, emp_row in employee_df_page.iterrows():
            ui_components.display_review_details(emp_row, index, classifier_model)
    if total_items > 0: ui_components.display_pagination_controls(total_items)