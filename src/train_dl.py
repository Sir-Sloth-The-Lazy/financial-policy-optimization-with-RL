import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from sklearn.metrics import roc_auc_score, f1_score
from dataset import LoanDataset
from models import LoanClassifier

# Hyperparameters
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 10
HIDDEN_DIM = 128
DROPOUT = 0.3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data/processed')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    print(f"Device: {DEVICE}")
    
    # Datasets
    train_ds = LoanDataset(DATA_DIR, mode='train')
    val_ds = LoanDataset(DATA_DIR, mode='val')
    test_ds = LoanDataset(DATA_DIR, mode='test')
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # Model
    input_dim = train_ds[0][0].shape[0]
    model = LoanClassifier(input_dim, hidden_dim=HIDDEN_DIM, dropout_rate=DROPOUT).to(DEVICE)
    
    # Calculate pos_weight for imbalance
    # From EDA we found approx 4.0
    POS_WEIGHT = torch.tensor([4.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_val_auc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            # For BCEWithLogits, we need raw logits, so remove Sigmoid from model?
            # Or assume model still has Sigmoid and use BCELoss? 
            # Ideally: Use BCEWithLogitsLoss for numerical stability and implicit Sigmoid.
            # But our model has Sigmoid at the end. 
            # Let's remove Sigmoid in the model training? 
            # OR just use BCELoss with manual weight.
            # PyTorch BCELoss allows checks for weights? Yes 'weight' arg is batch-wise, not class-wise.
            # Actually BCEWithLogitsLoss is better. Let's adjust model output in loop or change model class.
            # EASIEST: Just use the raw logits (remove Sigmoid from model or just inverse it?)
            # Let's simple use BCELoss but we need to re-implement weighting manually or 
            # use a trick.
            
            # Better approach: 
            # 1. Use the model as is (Sigmoid output).
            # 2. Manually weight the loss.
            # weight element-wise: w = pos_weight if y=1 else 1.
            
            # Let's go with modifying the loss calc manually for simplicity without changing model class file
            y_pred = model(X_batch)
            
            # Manual weighted BCE
            # loss = - [ weight * y * log(p) + (1-y) * log(1-p) ]
            # clamped for stability
            p = y_pred.clamp(min=1e-7, max=1-1e-7)
            loss = -(POS_WEIGHT * y_batch * torch.log(p) + (1 - y_batch) * torch.log(1 - p))
            loss = loss.mean()
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_targets = []
        val_preds = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_pred = model(X_batch)
                val_targets.extend(y_batch.numpy())
                val_preds.extend(y_pred.cpu().numpy())
                
        val_auc = roc_auc_score(val_targets, val_preds)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f} - Val AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_dl_model.pth'))
            
    print(f"Best Val AUC: {best_val_auc:.4f}")
    
    # Test Evaluation
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_dl_model.pth')))
    model.eval()
    test_targets = []
    test_preds = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_pred = model(X_batch)
            test_targets.extend(y_batch.numpy())
            test_preds.extend(y_pred.cpu().numpy())
            
    test_targets = np.array(test_targets)
    test_preds = np.array(test_preds)
    
    test_auc = roc_auc_score(test_targets, test_preds)
    # F1 Score needs binary predictions
    test_binary = (test_preds > 0.5).astype(int)
    test_f1 = f1_score(test_targets, test_binary)
    
    print(f"\nFinal Test Metrics:")
    print(f"AUC: {test_auc:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    
    return test_auc, test_f1

if __name__ == "__main__":
    train_model()
