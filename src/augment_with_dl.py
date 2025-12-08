import torch
import numpy as np
import os
import sys
from torch.utils.data import TensorDataset, DataLoader

# Add src to path
sys.path.append(os.path.dirname(__file__))
from models import LoanClassifier

BATCH_SIZE = 1024
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def augment_features():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data/processed')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    print("Loading data...")
    X = np.load(os.path.join(DATA_DIR, 'X.npy')).astype(np.float32)
    
    # Load DL Model
    print("Loading DL Model...")
    input_dim = X.shape[1]
    # We used default hidden_dim=128 in train_dl.py
    dl_model = LoanClassifier(input_dim, hidden_dim=128, dropout_rate=0.3).to(DEVICE)
    dl_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_dl_model.pth'), map_location=DEVICE))
    dl_model.eval()
    
    # Run Inference
    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_probs = []
    
    print("Running inference to get Prob(Default)...")
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(DEVICE)
            y_pred = dl_model(X_batch)
            probs = y_pred.cpu().numpy()
            all_probs.append(probs)
            
    all_probs = np.concatenate(all_probs, axis=0)
    
    # Augment X
    # X shape: (N, D) -> (N, D+1)
    X_augmented = np.concatenate([X, all_probs], axis=1)
    
    save_path = os.path.join(DATA_DIR, 'X_risk_aware.npy')
    np.save(save_path, X_augmented)
    print(f"Saved augmented data to {save_path}. New shape: {X_augmented.shape}")

if __name__ == "__main__":
    augment_features()
