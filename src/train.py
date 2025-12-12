import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data/processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

def load_processed_data():
    X = np.load(os.path.join(DATA_DIR, 'X.npy'))
    y = np.load(os.path.join(DATA_DIR, 'y.npy'))
    try:
        feature_names = np.load(os.path.join(DATA_DIR, 'feature_names.npy'))
    except:
        feature_names = None
    return X, y, feature_names

def train_models():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print("Loading data...")
    X, y, feature_names = load_processed_data()
    
    # Split: 70% Train, 15% Val, 15% Test
    # Here we'll just do 80/20 Train/Test for simplicity in this script, 
    # as we aren't doing heavy hyperparameter tuning yet.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    }
    
    best_auc = 0
    best_model_name = ""
    best_model = None
    
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_prob)
        print(f"{name} ROC-AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))
        
        results[name] = {'auc': auc, 'model': model}
        
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_model = model
            
    print(f"\nBest Model: {best_model_name} with AUC: {best_auc:.4f}")
    
    # Save best model
    model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"Saved best model to {model_path}")

    # Feature Importance (if applicable)
    if feature_names is not None and hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("\nTop 10 Features:")
        for i in range(10):
            print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

if __name__ == "__main__":
    train_models()
