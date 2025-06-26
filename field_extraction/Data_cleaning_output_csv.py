
import pandas as pd
import re
from deep_translator import GoogleTranslator
from pathlib import Path

# Step 1: Load the dataset
script_dir = Path(__file__).parent
project_root = script_dir.parent
data_file = project_root / 'data' / 'employee_reviews.csv'
df = pd.read_csv(data_file, encoding='ISO-8859-1')

# Step 2: Clean the 'pros&cons' column
def clean_pros_cons(text):
    if pd.isna(text):
        return 'not mentioned'

    text = str(text).lower()
    text = text.encode('ascii', errors='ignore').decode('ascii')
    text = re.sub(r'[\*\+\-•]', ' ', text)  # remove unwanted symbols
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['pros&cons'] = df['pros&cons'].apply(clean_pros_cons)

# Step 3: Take sample of first 1000 rows for testing dataset and 100 rows for train dataset
#sample_df = df.head(1000).copy()
sample_df = df.tail(100).copy()

# Step 4: Translate and overwrite the same column
def translate_to_english(text):
    try:
        if pd.isna(text) or text.strip() == '':
            return 'not mentioned'
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        return text

sample_df['pros&cons'] = sample_df['pros&cons'].apply(translate_to_english)

# Step 5: Save result
# Cleaned Test dataset
#output_file = project_root / 'data' / 'employee_reviews_PROCESSED.csv'

# Cleaned Train dataset
output_file = project_root / 'data' / 'employee_reviews_TRAIN.csv'
sample_df.to_csv(output_file, index=False)

# Step 6: Preview result
print(sample_df['pros&cons'].head(10))