import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/raw/accepted_2007_to_2018.csv')

def analyze_features():
    print("Loading data...")
    # Load slightly larger sample for better correlation estimates
    df = pd.read_csv(DATA_PATH, low_memory=False, nrows=200000)
    
    # Filter for target loans
    target_loans = ['Fully Paid', 'Charged Off']
    df = df[df['loan_status'].isin(target_loans)].copy()
    
    # Encode target: Charged Off = 1, Fully Paid = 0
    df['target'] = df['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)
    
    print(f"Filtered Dataset Shape: {df.shape}")
    print(f"Default Rate: {df['target'].mean():.4f}")

    # Drop columns with high missing values (>50%)
    missing_frac = df.isnull().mean()
    drop_cols = missing_frac[missing_frac > 0.5].index
    print(f"Dropping {len(drop_cols)} columns with >50% missing values.")
    df_clean = df.drop(columns=drop_cols)
    
    # Select numeric columns for correlation
    numeric_df = df_clean.select_dtypes(include=[np.number])
    
    # Correlation with target
    print("\ncalculating correlations...")
    correlations = numeric_df.corr()['target'].sort_values(ascending=False)
    
    print("\nTop 10 Positive Correlations:")
    print(correlations.head(11).iloc[1:]) # Skip target itself
    
    print("\nTop 10 Negative Correlations:")
    print(correlations.tail(10))
    
    # Save correlation plot of top features
    top_corr_features = correlations.abs().sort_values(ascending=False).head(15).index
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df[top_corr_features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Top Features')
    plt.tight_layout()
    plt.savefig('notebooks/correlation_matrix.png')
    print("Saved correlation_matrix.png")

    # Plot distributions of a few key features split by target
    key_features = ['int_rate', 'dti', 'annual_inc', 'fico_range_low']
    for feat in key_features:
        if feat in df_clean.columns:
            plt.figure(figsize=(8, 4))
            # Limit annual_inc for visibility
            data_to_plot = df_clean
            if feat == 'annual_inc':
                data_to_plot = df_clean[df_clean[feat] < 200000]
            
            sns.boxplot(x='loan_status', y=feat, data=data_to_plot)
            plt.title(f'{feat} by Loan Status')
            plt.savefig(f'notebooks/dist_{feat}.png')
            print(f"Saved dist_{feat}.png")

if __name__ == "__main__":
    if not os.path.exists('notebooks'): assert False, "notebooks dir missing"
    analyze_features()
