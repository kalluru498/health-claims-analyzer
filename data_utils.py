import pandas as pd
import re

# Basic cleaner to normalize whitespace, strip, and lowercase
def clean_text_column(df, column='comment'):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    df[column] = df[column].astype(str)
    df[column] = df[column].apply(lambda x: re.sub(r'\s+', ' ', x).strip().lower())
    return df

# Function to safely load a CSV and validate schema
def load_claims_csv(file):
    df = pd.read_csv(file)
    required_columns = ['claim_id', 'comment', 'category', 'specialty', 'insurance_type', 'cpt_code', 'amount_expected', 'amount_paid']
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    return clean_text_column(df, 'comment')

# Helper to calculate payment gap
def compute_payment_gap(df):
    df['payment_gap'] = df['amount_expected'] - df['amount_paid']
    return df
