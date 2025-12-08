import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/raw/accepted_2007_to_2018.csv')
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '../')

def advanced_eda():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, low_memory=False, nrows=200000)
    target_loans = ['Fully Paid', 'Charged Off']
    df = df[df['loan_status'].isin(target_loans)].copy()
    df['target'] = df['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)
    
    # 1. Skewness Analysis
    skew_cols = ['annual_inc', 'revol_bal', 'total_acc', 'loan_amnt']
    print("\nSkewness Analysis:")
    for col in skew_cols:
        skew = df[col].skew()
        print(f"{col}: {skew:.4f}")
        
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), kde=True, bins=50)
        plt.title(f'Distribution of {col} (Skew: {skew:.2f})')
        plt.xlim(0, df[col].quantile(0.99)) # Limit x-axis to ignore extreme outliers for viz
        plt.savefig(os.path.join(ARTIFACTS_DIR, f'dist_{col}.png'))
        plt.close()

    # 2. Categorical Analysis: Grade & Sub-Grade
    print("\nAnalyzing Grades...")
    grade_order = sorted(df['grade'].unique())
    plt.figure(figsize=(10, 5))
    sns.barplot(x='grade', y='target', data=df, order=grade_order, errorbar=None)
    plt.title('Default Rate by Grade')
    plt.ylabel('Default Rate')
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'default_by_grade.png'))
    plt.close()

    # Sub-grade monotonicity
    sub_grade_order = sorted(df['sub_grade'].unique())
    plt.figure(figsize=(14, 6))
    sns.barplot(x='sub_grade', y='target', data=df, order=sub_grade_order, errorbar=None)
    plt.title('Default Rate by Sub-Grade')
    plt.xticks(rotation=45)
    plt.ylabel('Default Rate')
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'default_by_subgrade.png'))
    plt.close()
    
    # 3. Emp Length
    print("Analyzing Employment Length...")
    emp_order = [
        '< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', 
        '6 years', '7 years', '8 years', '9 years', '10+ years'
    ]
    plt.figure(figsize=(10, 5))
    sns.barplot(x='emp_length', y='target', data=df, order=emp_order, errorbar=None)
    plt.title('Default Rate by Employment Length')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'default_by_emp_length.png'))
    plt.close()

    # 4. Feature Engineering Exploration
    # Credit History Length
    if 'issue_d' in df.columns and 'earliest_cr_line' in df.columns:
        df['issue_d'] = pd.to_datetime(df['issue_d'])
        df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])
        df['credit_hist_years'] = (df['issue_d'] - df['earliest_cr_line']).dt.days / 365.25
        
        plt.figure(figsize=(8, 4))
        sns.boxplot(x='loan_status', y='credit_hist_years', data=df)
        plt.title('Credit History Length by Loan Status')
        plt.savefig(os.path.join(ARTIFACTS_DIR, 'dist_credit_hist.png'))
        plt.close()
        
    print("Advanced EDA complete. Images saved.")

if __name__ == "__main__":
    advanced_eda()
