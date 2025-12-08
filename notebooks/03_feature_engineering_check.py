import numpy as np
import os
import matplotlib.pyplot as plt

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/processed')

def inspect_processed_data():
    x_path = os.path.join(PROCESSED_DIR, 'X.npy')
    y_path = os.path.join(PROCESSED_DIR, 'y.npy')
    names_path = os.path.join(PROCESSED_DIR, 'feature_names.npy')
    
    if not os.path.exists(x_path):
        print("Processed data not found.")
        return

    print("Loading processed data...")
    X = np.load(x_path)
    y = np.load(y_path)
    try:
        feature_names = np.load(names_path)
    except:
        feature_names = ["Feature " + str(i) for i in range(X.shape[1])]
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Feature names ({len(feature_names)}):")
    print(feature_names[:10])
    
    print("\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))
    
    print("\nChecking for NaNs in X:")
    if np.isnan(X).any():
        print("WARNING: NaNs found in X!")
    else:
        print("No NaNs found in X. Clean.")
        
    print("\nFeature Statistics (First 5 features):")
    for i in range(min(5, X.shape[1])):
        print(f"{feature_names[i]}: Mean={X[:, i].mean():.2f}, Std={X[:, i].std():.2f}")

if __name__ == "__main__":
    inspect_processed_data()
