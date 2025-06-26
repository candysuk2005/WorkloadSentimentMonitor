# WorkloadSentimentMonitor/field_extraction/extract_fields.py

import pandas as pd
import random
import re
import os


def light_clean_text(text):
    """Light cleaning: lowercase, remove URLs, normalize whitespace."""
    if not isinstance(text, str):
        return ""
    text_processed = str(text).lower()
    text_processed = re.sub(r'http\S+|www\.\S+', '', text_processed, flags=re.MULTILINE)
    text_processed = re.sub(r'\s+', ' ', text_processed).strip()
    # Optional: A very light VADER-friendly punctuation pass if summary is very messy
    # text_processed = re.sub(r'[^a-z0-9\s.,!?\'"-]', '', text_processed)
    return text_processed


def load_employee_data(data_folder="data",
                       filename="employee_reviews_PROCESSED.csv"):  # <<< USE TUTOR'S SCRIPT OUTPUT
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_script_dir)
    data_file_path = os.path.join(project_root_dir, data_folder, filename)

    print(f"extract_fields.py: Loading preprocessed sample from '{data_file_path}'")
    try:
        df = pd.read_csv(data_file_path, on_bad_lines='skip', keep_default_na=False, na_filter=False)
        if df.empty:
            print(f"extract_fields.py: Warning - File '{filename}' is empty.")
            return pd.DataFrame()
        print(f"extract_fields.py: Loaded {len(df)} records from '{filename}'.")

        # --- Construct the 'communication' field ---
        communications = []
        for index, row in df.iterrows():
            summary_part = ""
            if 'summary' in df.columns and pd.notna(row['summary']) and str(row['summary']).strip():
                summary_part = light_clean_text(str(row['summary']))
                if summary_part.lower() == "not mentioned":  # If summary itself is "not mentioned"
                    summary_part = ""

            pros_cons_part = ""
            if 'pros&cons' in df.columns and pd.notna(row['pros&cons']) and str(row['pros&cons']).strip():
                # This 'pros&cons' column is assumed to be already cleaned & translated by tutor's script
                # We just check if it's the placeholder "not mentioned"
                temp_pros_cons = str(row['pros&cons']).strip()  # No further cleaning here, assume it's ready
                if temp_pros_cons.lower() != "not mentioned":
                    pros_cons_part = temp_pros_cons

            # Combine parts
            final_text_parts = []
            if summary_part:
                final_text_parts.append(summary_part)
            if pros_cons_part:
                final_text_parts.append(pros_cons_part)

            communications.append(" ".join(final_text_parts).strip())

        df['communication'] = communications

        # Filter out rows where communication is now effectively empty or too short
        df = df[df['communication'].str.strip().str.len() > 5]

        if df.empty:
            print(
                f"extract_fields.py: Warning - No valid data after constructing 'communication' field from '{filename}'.")
            return pd.DataFrame()

        # Simulate workload metrics
        df['employee_id'] = [f"SampleReviewer_{idx}" for idx in df.index]
        df['tasks_assigned'] = [random.randint(5, 15) for _ in range(len(df))]
        df['tasks_overdue'] = [random.randint(0, 7) for _ in range(len(df))]
        df['tasks_overdue'] = df.apply(
            lambda r: min(r['tasks_overdue'], r['tasks_assigned']) if r['tasks_assigned'] > 0 else 0,
            axis=1
        )

        final_df = df[['employee_id', 'tasks_assigned', 'tasks_overdue', 'communication']].copy()
        print(f"extract_fields.py: Prepared {len(final_df)} reviews for the app from '{filename}'.")
        return final_df

    except FileNotFoundError:
        print(f"extract_fields.py: ERROR - File '{filename}' not found at '{data_file_path}'.")
        return pd.DataFrame()
    except Exception as e:
        print(f"extract_fields.py: ERROR processing '{filename}': {e}")
        import traceback
        print(traceback.format_exc())
        return pd.DataFrame()


if __name__ == '__main__':
    print("--- Testing extract_fields.py directly (simplified version) ---")
    test_df = load_employee_data()

    if not test_df.empty:
        print(f"\nLoaded and processed {len(test_df)} records.")
        print("\nSample data (first 5 rows):")
        print(test_df.head())
        if len(test_df) > 0:
            print("\nSample 'communication' (first entry):")
            print(f"'{test_df['communication'].iloc[0]}'")
            # Example: Check the case where pros&cons might have been "not mentioned"
            # Find a row in your employee_reviews_sample_translated1.csv where 'pros&cons' is 'not mentioned'
            # and 'summary' has text. Let's say its original index was 3 (0-indexed).
            if len(test_df) > 3 and 'Reviewer_3' in test_df[
                'employee_id'].values:  # Assuming original index 3 becomes Reviewer_3
                print("\nExample for a row that might have had 'pros&cons' as 'not mentioned':")
                print(f"'{test_df[test_df['employee_id'] == 'SampleReviewer_3']['communication'].iloc[0]}'")
    else:
        print("\nFailed to load/process data. Check console.")