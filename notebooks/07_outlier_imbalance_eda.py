import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/raw/accepted_2007_to_2018.csv')
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '../')

def targeted_eda():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, low_memory=False, nrows=200000)
    target_loans = ['Fully Paid', 'Charged Off']
    df = df[df['loan_status'].isin(target_loans)].copy()
    df['target'] = df['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)
    
    # 1. Outlier Analysis & Clipping Thresholds
    outlier_features = ['annual_inc', 'revol_bal', 'total_acc']
    print("\nOutlier Analysis (99th percentile):")
    
    for col in outlier_features:
        p99 = df[col].quantile(0.99)
        print(f"{col}: 99th percentile = {p99:.2f}")
        
        # Visualize original vs clipped
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.histplot(df[col].dropna(), bins=50)
        plt.title(f'Original {col}')
        
        plt.subplot(1, 2, 2)
        clipped_data = df[col].clip(upper=p99)
        sns.histplot(clipped_data.dropna(), bins=50)
        plt.title(f'Clipped (at {p99:.0f}) {col}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACTS_DIR, f'outlier_check_{col}.png'))
        plt.close()

    # 2. Class Imbalance
    print("\nClass Imbalance Analysis:")
    counts = df['target'].value_counts()
    print(counts)
    
    neg_count = counts[0]
    pos_count = counts[1]
    scale_pos_weight = neg_count / pos_count
    
    print(f"Negative (Fully Paid): {neg_count}")
    print(f"Positive (Charged Off): {pos_count}")
    print(f"Recommended pos_weight: {scale_pos_weight:.4f}")
    
    with open(os.path.join(ARTIFACTS_DIR, 'class_weight_recommendation.txt'), 'w') as f:
        f.write(str(scale_pos_weight))

if __name__ == "__main__":
    targeted_eda()
