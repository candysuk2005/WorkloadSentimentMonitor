# WorkloadSentimentMonitor/app/ui_components.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt  # Import Matplotlib
import analysis_utils

ITEMS_PER_PAGE = 10


# --- NEW: Function to display the pie chart ---
def display_sentiment_distribution_chart(sentiment_counts):
    """Renders a pie chart of sentiment distribution."""
    if sentiment_counts.empty:
        st.warning("No data available to display sentiment distribution.")
        return

    st.subheader("Overall Sentiment Distribution")

    # Define consistent colors
    color_map = {
        'Positive': '#28a745',  # Green
        'Negative': '#dc3545',  # Red
        'Neutral': '#ffc107'  # Yellow/Amber
    }

    # Ensure the colors align with the labels, even if a category is missing
    colors = [color_map.get(label, '#6c757d') for label in sentiment_counts.index]

    fig, ax = plt.subplots(figsize=(5, 3))  # Create a figure and axes
    ax.pie(
        sentiment_counts.values,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor='w')  # Donut chart style
    )
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig)  # Display the plot in Streamlit


# --- END NEW ---

def display_review_details(emp_row, index_val, classifier_model):
    # This function remains unchanged.
    # ... (rest of the function is the same) ...
    employee_id_display = emp_row.get('employee_id', f"Review_{index_val}")
    st.subheader(f"Review ID: {employee_id_display}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### Workload Insights (Simulated)")
        tasks_assigned = emp_row.get('tasks_assigned', 0)
        tasks_overdue = emp_row.get('tasks_overdue', 0)
        st.write(f"Sim. Tasks Assigned: {tasks_assigned}")
        st.write(f"Sim. Tasks Overdue: {tasks_overdue}")
        workload_metric = analysis_utils.calculate_workload_metric(tasks_assigned, tasks_overdue)
        st.metric(label="Workload Pressure Score", value=f"{workload_metric:.2f}/10",
                  help="Score 0-10. Higher indicates more pressure. (Simulated for reviews)")
    communication_text = emp_row.get('communication', "")
    sentiment_for_risk_calc = 0.0
    with col2:
        st.markdown("##### Sentiment Analysis")
        text_area_key = f"comm_{employee_id_display}"
        st.text_area("Review Text Snippet:", value=communication_text, height=120, disabled=True, key=text_area_key)
        if st.session_state.selected_model == "Vader Model":
            vader_sentiment_score = analysis_utils.get_sentiment_score(communication_text)
            st.metric(label="VADER Sentiment Score", value=f"{vader_sentiment_score:.2f}",
                      help="Score -1 (negative) to +1 (positive). Raw VADER output.")
            _, _, _, sentiment_for_risk_calc = analysis_utils.calculate_burnout_risk(workload_metric,
                                                                                     vader_sentiment_score,
                                                                                     communication_text)
        elif st.session_state.selected_model == "Classification Model":
            if classifier_model:
                prediction = classifier_model.predict([communication_text])[0]
                st.metric(label="Classification Model Prediction", value=prediction,
                          help="Predicted sentiment from the trained Naive Bayes classifier.")
                sentiment_map = {'Positive': 0.8, 'Neutral': 0.0, 'Negative': -0.8}
                sentiment_for_risk_calc = sentiment_map.get(prediction, 0.0)
            else:
                st.error("Classifier model not loaded. Cannot get prediction.")
                sentiment_for_risk_calc = 0.0
    risk_level, color, recommendation, _ = analysis_utils.calculate_burnout_risk(workload_metric,
                                                                                 sentiment_for_risk_calc,
                                                                                 communication_text)
    with col3:
        st.markdown("##### Burnout Risk Assessment")
        st.markdown(f"**Calculated Risk Level:**")
        st.markdown(f"<h3 style='color:{color};'>{risk_level} Risk</span>", unsafe_allow_html=True)
        st.info(f"**Actionable Insight:** {recommendation}")
    st.markdown("---")


def display_pagination_controls(total_items):
    # This function remains unchanged.
    # ... (rest of the function is the same) ...
    st.markdown("---")
    total_pages = (total_items + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
    if total_pages > 0:
        st.write(f"Page {st.session_state.current_page + 1} of {total_pages}")
    col_prev, _, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("⬅️ Previous", use_container_width=True, disabled=(st.session_state.current_page == 0)):
            st.session_state.current_page -= 1;
            st.rerun()
    with col_next:
        disable_next = (st.session_state.current_page + 1) >= total_pages
        if st.button("Next ➡️", use_container_width=True, disabled=disable_next):
            st.session_state.current_page += 1;
            st.rerun()


def display_sidebar_info(report_data):
    # This function remains unchanged.
    # ... (rest of the function is the same) ...
    st.sidebar.header("About this PoC")
    st.sidebar.info("A Proof of Concept for using sentiment analysis to help identify potential employee burnout.")
    st.sidebar.header("Controls")

    def on_model_change():
        print(f"'{st.session_state.selected_model}' model is selected.")

    st.sidebar.selectbox(label="Choose a Sentiment Model", options=["Vader Model", "Classification Model"],
                         key='selected_model', on_change=on_model_change)
    st.sidebar.markdown("---")
    st.sidebar.header(f"{st.session_state.selected_model} Performance")
    if report_data and st.session_state.selected_model in report_data:
        model_report = report_data[st.session_state.selected_model]
        st.sidebar.subheader("Classification Report")
        st.sidebar.text(model_report['classification_report'])
        st.sidebar.subheader("Confusion Matrix")
        st.sidebar.caption(f"(Rows: Actual, Columns: Predicted)")
        matrix_df = pd.DataFrame(model_report['confusion_matrix'], columns=model_report['confusion_matrix_labels'],
                                 index=model_report['confusion_matrix_labels'])
        st.sidebar.table(matrix_df)
    else:
        st.sidebar.warning("Performance report not found. Please run the training script.")
    st.sidebar.markdown("---")
    st.sidebar.header("How to Interpret the Results")
    st.sidebar.markdown("""
        **Primary Goal:** For this business problem, the main goal is to identify every employee who might be at risk. Missing a negative case (a "False Negative") is the worst possible outcome.
        **What to Look For:**
        *   **Negative Recall:** This is the most important metric. A high recall means the model is good at finding all the *actual* negative reviews.
        *   **Negative Precision:** A high precision means that when the model says a review is negative, it's usually correct (fewer false alarms).
        For a burnout detector, high recall is generally preferred, even if it means slightly lower precision.
        """)
    st.sidebar.markdown("---")
    st.sidebar.caption("Developed as a PoC for AI for Business Capstone.")