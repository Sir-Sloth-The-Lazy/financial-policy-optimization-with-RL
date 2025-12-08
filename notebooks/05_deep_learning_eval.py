import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from dataset import LoanDataset
from models import LoanClassifier

BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_DIM = 128
DROPOUT = 0.3

def evaluate_dl_model():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data/processed')
    MODEL_PATH = os.path.join(BASE_DIR, 'models/best_dl_model.pth')
    
    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return

    print("Loading test data...")
    test_ds = LoanDataset(DATA_DIR, mode='test')
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    input_dim = test_ds[0][0].shape[0]
    model = LoanClassifier(input_dim, hidden_dim=HIDDEN_DIM, dropout_rate=DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    targets = []
    preds = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_pred = model(X_batch)
            targets.extend(y_batch.numpy())
            preds.extend(y_pred.cpu().numpy())
            
    targets = np.array(targets)
    preds = np.array(preds)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(targets, preds)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'DL ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('DL Model ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_dl.png')
    print("Saved roc_curve_dl.png")
    
    # Confusion Matrix
    binary_preds = (preds > 0.5).astype(int)
    cm = confusion_matrix(targets, binary_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fully Paid', 'Charged Off'])
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Greens)
    plt.title('DL Model Confusion Matrix')
    plt.savefig('confusion_matrix_dl.png')
    print("Saved confusion_matrix_dl.png")

if __name__ == "__main__":
    evaluate_dl_model()
