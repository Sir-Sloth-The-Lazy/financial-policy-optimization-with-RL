import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data/processed')
MODEL_PATH = os.path.join(BASE_DIR, 'models/best_model.pkl')

def evaluate_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return

    print("Loading data and model...")
    X = np.load(os.path.join(DATA_DIR, 'X.npy'))
    y = np.load(os.path.join(DATA_DIR, 'y.npy'))
    model = joblib.load(MODEL_PATH)
    
    # We need the test set used in training. 
    # Ideally we'd save the indices, but for this simple script we'll re-split with same seed.
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(BASE_DIR, 'roc_curve.png'))
    print("Saved roc_curve.png")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fully Paid', 'Charged Off'])
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix.png'))
    print("Saved confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()
