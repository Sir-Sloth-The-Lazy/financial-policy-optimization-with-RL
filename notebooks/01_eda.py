import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/raw/accepted_2007_to_2018.csv')

def run_eda():
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        return

    print(f"Loading data from {DATA_PATH}...")
    # Load a sample for initial inspection to save memory/time
    df = pd.read_csv(DATA_PATH, low_memory=False, nrows=100000)
    
    print(f"\nDataset Shape (Sample): {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nColumn Info:")
    print(df.info())

    print("\nMissing Values (Top 20):")
    missing = df.isnull().sum().sort_values(ascending=False).head(20)
    print(missing)

    # Target Variable Analysis
    if 'loan_status' in df.columns:
        print("\nLoan Status Distribution:")
        print(df['loan_status'].value_counts(normalize=True))
        
        plt.figure(figsize=(12, 6))
        sns.countplot(y='loan_status', data=df, order=df['loan_status'].value_counts().index)
        plt.title('Loan Status Distribution (First 100k rows)')
        plt.tight_layout()
        output_img = os.path.join(os.path.dirname(__file__), 'loan_status_distribution.png')
        plt.savefig(output_img)
        print(f"\nSaved distribution plot to {output_img}")
    else:
        print("\n'loan_status' column not found!")

if __name__ == "__main__":
    run_eda()
