import torch
import d3rlpy
import numpy as np
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from models import LoanClassifier

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def analyze_divergence():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data/processed')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    # 1. Load Data
    print("Loading Data...")
    X = np.load(os.path.join(DATA_DIR, 'X.npy')).astype(np.float32)
    y = np.load(os.path.join(DATA_DIR, 'y.npy'))
    
    # We need features to display (e.g., loan_amnt, int_rate).
    # Since we don't have the raw DF loaded, we'll just use the numpy features.
    # We know that 'int_rate' and 'loan_amnt' were scaled.
    # It's better to load the original DF for readability.
    
    # Load raw DF for lookup
    raw_df_path = os.path.join(BASE_DIR, 'data/raw/accepted_2007_to_2018.csv')
    # We need to replicate the exact filtering from preprocessing to match indices?
    # This is tricky because preprocessing drops columns and rows.
    # Better: Use the feature names if available or just raw comparison.
    
    # Let's rely on X, y and Model Predictions.
    
    # Test Split (Last 15% as per dataset.py random split 42)
    # Replicate split indices
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    split_ratio = (0.7, 0.15, 0.15)
    train_end = int(len(X) * split_ratio[0])
    val_end = int(len(X) * (split_ratio[0] + split_ratio[1]))
    test_indices = indices[val_end:]
    
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    # 2. Get DL Predictions (Prob Default)
    print("Running DL Model...")
    dl_model = LoanClassifier(X.shape[1], hidden_dim=128, dropout_rate=0.3).to(DEVICE)
    dl_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_dl_model.pth'), map_location=DEVICE))
    dl_model.eval()
    
    with torch.no_grad():
        inputs = torch.from_numpy(X_test).to(DEVICE)
        dl_preds = dl_model(inputs).cpu().numpy().flatten()
        
    # 3. Get RL Actions (V1 - Risk Neutral/Yield Chasing)
    # The prompt explicitly asks why RL approves high risk. This applies to V1 (Task 5).
    print("Running RL Agent (V1)...")
    cql = d3rlpy.algos.DiscreteCQLConfig().create(device='cpu')
    
    # Init dummy
    dummy_obs = np.random.random((2, X.shape[1])).astype(np.float32)
    dummy_actions = np.array([0, 1], dtype=np.int32)
    dummy_rewards = np.array([0, 0], dtype=np.float32)
    dummy_terminals = np.array([1, 1], dtype=np.float32)
    dataset = d3rlpy.dataset.MDPDataset(dummy_obs, dummy_actions, dummy_rewards, dummy_terminals)
    cql.build_with_dataset(dataset)
    
    cql.load_model(os.path.join(MODEL_DIR, 'cql_agent.d3'))
    rl_actions = cql.predict(X_test)
    
    # 4. Find Divergence: DL Says "High Risk" (Prob > 0.5), RL Says "Approve" (Action 1)
    # Divergence Mask
    high_risk_mask = dl_preds > 0.5
    rl_approve_mask = rl_actions == 1
    
    divergence_mask = high_risk_mask & rl_approve_mask
    divergence_indices = np.where(divergence_mask)[0]
    
    print(f"\n--- Divergence Analysis ---")
    print(f"Total Test Samples: {len(X_test)}")
    print(f"DL Rejected (High Risk): {np.sum(high_risk_mask)}")
    print(f"RL Approved: {np.sum(rl_approve_mask)}")
    print(f"Overlap (High Risk BUT Approved by RL): {len(divergence_indices)}")
    
    if len(divergence_indices) > 0:
        print("\nTaking first 5 divergence examples...")
        sample_idxs = divergence_indices[:5]
        
        # We need to see *why*. The "Hint: Think about reward" suggests Interest Rate.
        # We need to recover the Interest Rate for these samples.
        # The 'int_rate' is likely one of the features.
        # Check feature_names.npy
        feature_names = np.load(os.path.join(DATA_DIR, 'feature_names.npy'), allow_pickle=True)
        int_rate_idx = np.where(feature_names == 'int_rate')[0][0]
        loan_amnt_idx = np.where(feature_names == 'loan_amnt')[0][0]
        
        # Note: X is scaled. We can't see raw values easily.
        # But we can compare them relative to the mean (0).
        
        for i in sample_idxs:
            idx = i
            prob = dl_preds[idx]
            action = rl_actions[idx] # Should be 1
            outcome = y_test[idx] # 1 = Default, 0 = Paid
            
            # Scaled Features
            scaled_int = X_test[idx, int_rate_idx]
            scaled_loan = X_test[idx, loan_amnt_idx]
            
            print(f"\nEx #{i}:")
            print(f"  DL Prob(Default): {prob:.4f} (High Risk)")
            print(f"  RL Action: {action} (Approve)")
            print(f"  Actual Outcome: {'Default' if outcome==1 else 'Paid'}")
            print(f"  Scaled Int Rate: {scaled_int:.4f}")
            print(f"  Scaled Loan Amnt: {scaled_loan:.4f}")
            
            # Explanation
            if scaled_int > 0:
                print("  -> Interest Rate is ABOVE average. RL likely sees high potential reward.")
            else:
                print("  -> Interest Rate is below average. RL logic unclear (maybe behavioral bias).")

    # Save IDs for user inspection?
    # No, just printing is fine for the report.

if __name__ == "__main__":
    analyze_divergence()
