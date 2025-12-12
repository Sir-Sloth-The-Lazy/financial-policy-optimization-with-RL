import torch
import d3rlpy
import numpy as np
import pandas as pd
import os
import sys
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from dataset import LoanDataset
from models import LoanClassifier
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1024

def run_analysis():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data/processed')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    # --- 1. Load Data ---
    print("Loading Test Data...")
    test_ds = LoanDataset(DATA_DIR, mode='test')
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load Rewards and Original features for interpretation if needed
    # We need to map back to original indices to get rewards and raw features
    all_rewards = np.load(os.path.join(DATA_DIR, 'rl_rewards.npy'))
    processed_X = np.load(os.path.join(DATA_DIR, 'X.npy')) # We might need this to reverse transform if we want raw values
    
    # The dataset uses a fixed seed, so indices are deterministic.
    # However, to be safe and simple, let's trust the dataset indices if exposed, 
    # OR (since dataset.py might not expose them easily in __getitem__), we re-generate the split indices logic
    # checking dataset.py content again would be good, but assuming standard split:
    # modification: LoanDataset uses a fixed random seed (42) and sklearn's train_test_split.
    # We can rely on the fact that we're using the 'test' mode.
    
    # Let's get the indices from the dataset object if possible, otherwise we trust the loader order matches 'test' split order
    # LoanDataset in current src usually stores indices.
    test_indices = test_ds.indices
    test_rewards = all_rewards[test_indices]
    
    # --- 2. Load DL Model ---
    print("Loading DL Model...")
    input_dim = test_ds[0][0].shape[0]
    dl_model = LoanClassifier(input_dim, hidden_dim=128, dropout_rate=0.3).to(DEVICE)
    dl_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_dl_model.pth'), map_location=DEVICE))
    dl_model.eval()
    
    # --- 3. Load RL Agent ---
    print("Loading RL Agent...")
    cql = d3rlpy.algos.DiscreteCQLConfig().create(device='cpu')
    
    # Dummy build to initialize
    cql.build_with_dataset(d3rlpy.dataset.MDPDataset(
        observations=np.random.random((2, input_dim)).astype(np.float32),
        actions=np.array([0, 1], dtype=np.int32),
        rewards=np.array([0, 0], dtype=np.float32),
        terminals=np.array([1, 1], dtype=np.float32),
    ))
    cql.load_model(os.path.join(MODEL_DIR, 'cql_agent.d3'))
    
    # --- 4. Inference ---
    print("Running Inference...")
    dl_probs = []
    rl_actions = []
    targets = []
    
    all_X_np = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # DL
            X_device = X_batch.to(DEVICE)
            y_pred = dl_model(X_device).cpu().numpy().flatten()
            dl_probs.extend(y_pred)
            
            # RL
            X_np = X_batch.cpu().numpy()
            all_X_np.append(X_np)
            actions = cql.predict(X_np)
            rl_actions.extend(actions)
            
            targets.extend(y_batch.numpy().flatten())
            
    dl_probs = np.array(dl_probs)
    rl_actions = np.array(rl_actions)
    targets = np.array(targets)
    all_X_np = np.concatenate(all_X_np, axis=0) # For lookup
    
    # --- 5. Metrics ---
    
    # DL Metrics
    dl_auc = roc_auc_score(targets, dl_probs)
    # F1 at threshold 0.5
    dl_preds_binary = (dl_probs > 0.5).astype(int)
    dl_f1 = f1_score(targets, dl_preds_binary)
    
    # RL Metrics (Policy Value)
    # RL Action 1=Approve, 0=Deny.
    # Reward structure: Paid = +Profit, Default = -Loss.
    # If Action=0, Reward=0 (Opportunity cost is 0 in this formulation, we just don't gain/lose money).
    # But wait, the reward matrix 'rl_rewards.npy' has the actual PnL for that loan.
    # So if we Approve (Action=1), we get test_rewards[i]. If Deny (Action=0), we get 0.
    
    rl_value = np.sum(rl_actions * test_rewards)
    
    # DL Policy Value (for comparison)
    # DL approves if Prob(Default) < Threshold.
    # Note: dl_probs is Prob(Default) usually.
    # Let's verify target encoding: y=1 is Charged Off (Default).
    # So Prob(y=1) is Prob(Default).
    # We want to Approve if Prob(Default) is LOW.
    # DL Action = 1 if Prob < Threshold
    dl_threshold = 0.5
    dl_policy_actions = (dl_probs < dl_threshold).astype(int)
    dl_policy_value = np.sum(dl_policy_actions * test_rewards)
    
    print("\n--- Analysis Results ---")
    print(f"DL AUC: {dl_auc:.4f}")
    print(f"DL F1 (Default detection): {dl_f1:.4f}")
    print(f"DL Policy Value (Approve if P(Default)<0.5): ${dl_policy_value:,.2f}")
    print(f"RL Policy Value: ${rl_value:,.2f}")
    
    # --- 6. Disagreement Examples ---
    # Case A: DL Says High Risk (Probs > 0.5, so Deny), RL Says Approve (Action=1)
    # This implies RL sees potential profit despite high risk, or DL is wrong.
    
    case_a_mask = (dl_probs > 0.6) & (rl_actions == 1) # DL very sure it's default, RL approves
    case_a_indices = np.where(case_a_mask)[0]
    
    # Case B: DL Says Low Risk (Probs < 0.2, so Approve), RL Says Deny (Action=0)
    # This implies RL sees low profit or potential risk DL missed?
    case_b_mask = (dl_probs < 0.2) & (rl_actions == 0)
    case_b_indices = np.where(case_b_mask)[0]
    
    print(f"\nDisagreements:")
    print(f"DL High Risk (Deny) but RL Approves: {len(case_a_indices)} instances")
    print(f"DL Low Risk (Approve) but RL Denies: {len(case_b_indices)} instances")
    
    results = {
        'metrics': {
            'dl_auc': dl_auc,
            'dl_f1': dl_f1,
            'dl_value': dl_policy_value,
            'rl_value': rl_value
        },
        'examples': []
    }
    
    # Extract some details for these examples
    # We need access to feature values. all_X_np is scaled.
    # It's hard to interpret scaled values directly.
    # However, we can dump the raw values if we saved X.npy (processed)
    # Wait, X.npy is also processed.
    # We don't have easy access to raw CSV rows unless we reload CSV and match indices.
    # For now, let's save the indices and scaled features.
    
    if len(case_a_indices) > 0:
        idx = case_a_indices[0] # Take first one
        results['examples'].append({
            'type': 'DL_Deny_RL_Approve',
            'index': int(idx),
            'dl_prob': float(dl_probs[idx]),
            'rl_action': int(rl_actions[idx]),
            'reward': float(test_rewards[idx]),
            'target': int(targets[idx]), # Did they actually default?
            'features': all_X_np[idx].tolist()
        })
        
    if len(case_b_indices) > 0:
        idx = case_b_indices[0]
        results['examples'].append({
            'type': 'DL_Approve_RL_Deny',
            'index': int(idx),
            'dl_prob': float(dl_probs[idx]),
            'rl_action': int(rl_actions[idx]),
            'reward': float(test_rewards[idx]),
            'target': int(targets[idx]),
            'features': all_X_np[idx].tolist()
        })
        
    # Save results to JSON
    import json
    with open('analysis_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        print("Detailed results saved to analysis_results.json")

if __name__ == "__main__":
    run_analysis()
