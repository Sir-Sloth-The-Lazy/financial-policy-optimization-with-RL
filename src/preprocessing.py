import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

# Features
# Moved Features back to Normal/OneHot pipeline based on findings
NUMERIC_FEATURES = [
    'loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 
    'fico_range_low', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 
    'total_acc', 'mort_acc', 'pub_rec_bankruptcies', 
    # New
    'credit_hist_years', 'balance_to_income', 'installment_to_monthly_inc'
]

CATEGORICAL_FEATURES = [
    'term', 'grade', 'sub_grade', 'emp_length',  # Reverted Ordinals back to OneHot
    'home_ownership', 'verification_status', 'purpose', 'addr_state', 
    'initial_list_status', 'application_type'
]

def load_and_engineer_data(filepath, sample_size=None):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False, nrows=sample_size)
    valid_status = ['Fully Paid', 'Charged Off']
    df = df[df['loan_status'].isin(valid_status)].copy()
    df['target'] = df['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)
    
    # Feature Engineering
    # 1. Credit History Length
    if 'issue_d' in df.columns and 'earliest_cr_line' in df.columns:
        df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
        df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
        df['credit_hist_years'] = (df['issue_d'] - df['earliest_cr_line']).dt.days / 365.25
        df['credit_hist_years'] = df['credit_hist_years'].fillna(df['credit_hist_years'].median())

    # 2. Outlier Clipping (99th percentile)
    clip_dict = {
        'annual_inc': 264000.0,
        'revol_bal': 107000.0,
        'total_acc': 61.0
    }
    for col, limit in clip_dict.items():
        if col in df.columns:
            df[col] = df[col].clip(upper=limit)

    # 3. Ratios
    # Fill defaults for calculation
    df['annual_inc'] = df['annual_inc'].fillna(df['annual_inc'].median())
    df['installment'] = df['installment'].fillna(df['installment'].median())
    
    df['balance_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
    df['installment_to_monthly_inc'] = df['installment'] / ((df['annual_inc'] / 12) + 1)
    
    return df

def get_preprocessing_pipeline():
    # Standard Pipeline for all Numerics
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # OneHot Pipeline for all Categoricals
    onehot_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # Using OneHot is better for DL often as it creates distinct orthogonal inputs
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', onehot_transformer, CATEGORICAL_FEATURES)
        ])

    return preprocessor

def process_and_save(input_path, output_dir, sample_size=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = load_and_engineer_data(input_path, sample_size)
    
    # Check for missing cols
    X_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[X_cols]
    y = df['target'].values
    
    print("Fitting preprocessing pipeline...")
    pipeline = get_preprocessing_pipeline()
    X_processed = pipeline.fit_transform(X)
    
    print(f"Processed Shape: {X_processed.shape}")
    
    np.save(os.path.join(output_dir, 'X.npy'), X_processed)
    np.save(os.path.join(output_dir, 'y.npy'), y)
    print(f"Saved processed data to {output_dir}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    RAW_DATA = os.path.join(BASE_DIR, 'data/raw/accepted_2007_to_2018.csv')
    PROCESSED_DIR = os.path.join(BASE_DIR, 'data/processed')
    
    process_and_save(RAW_DATA, PROCESSED_DIR, sample_size=None)
