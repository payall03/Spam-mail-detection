import pandas as pd
import re
import os

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)       # Remove special chars
    text = re.sub(r'\s+', ' ', text)      # Remove extra spaces
    return text.strip()

def load_and_clean_data(input_path="data/spam.csv", output_path="data/cleaned_spam.csv"):
    # Load original dataset
    df = pd.read_csv(input_path, encoding='latin-1')

    # Keep only relevant columns
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']

    # Drop missing values
    df.dropna(inplace=True)

    # Drop duplicates
    df.drop_duplicates(subset='text', inplace=True)

    # Clean text
    df['text'] = df['text'].apply(clean_text)

    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Cleaned data saved to {output_path}")
    return df
