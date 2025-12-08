import torch
import d3rlpy
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from dataset import LoanDataset
from models import LoanClassifier
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1024

def compare_models():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data/processed')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    # 1. Load Data (Test Set)
    print("Loading Test Data...")
    test_ds = LoanDataset(DATA_DIR, mode='test')
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # We also need the 'raw' rewards/loan_amnts for the test set to calculate Policy Value
    # The Dataset returns (X, y). It doesn't return loan_amt/int_rate.
    # We need to map back to the original dataframe or load the matched rewards.
    # Since dataset.py uses deterministic split (seed 42), we can replicate it.
    
    # Load all rewards and index them
    all_rewards = np.load(os.path.join(DATA_DIR, 'rl_rewards.npy'))
    # Note: rl_rewards has augmented data appended at the end.
    # The first half is the original data.
    original_count = len(all_rewards) // 2
    original_rewards = all_rewards[:original_count]
    
    # Get indices from dataset
    test_indices = test_ds.indices
    test_rewards = original_rewards[test_indices]
    
    # 2. Load DL Model
    print("Loading DL Model...")
    input_dim = test_ds[0][0].shape[0]
    dl_model = LoanClassifier(input_dim, hidden_dim=128, dropout_rate=0.3).to(DEVICE)
    dl_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_dl_model.pth'), map_location=DEVICE))
    dl_model.eval()
    
    # 3. Load RL Agent
    print("Loading RL Agent...")
    cql = d3rlpy.algos.DiscreteCQLConfig().create(device='cpu')
    
    # We need to initialize the network with correct shapes before loading weights
    # Input shape: (143,) from test_ds
    # Action size: 2 (Binary: Deny/Approve)
    # We can do this by calling build_with_dataset with a dummy dataset
    
    dummy_obs = np.random.random((2, input_dim)).astype(np.float32)
    dummy_actions = np.array([0, 1], dtype=np.int32)
    dummy_rewards = np.array([0, 0], dtype=np.float32)
    dummy_terminals = np.array([1, 1], dtype=np.float32)
    
    dummy_dataset = d3rlpy.dataset.MDPDataset(
        observations=dummy_obs,
        actions=dummy_actions,
        rewards=dummy_rewards,
        terminals=dummy_terminals,
    )
    
    cql.build_with_dataset(dummy_dataset)
    
    # Now load weights
    cql.load_model(os.path.join(MODEL_DIR, 'cql_agent.d3'))
    
    # 4. Run Inference
    dl_probs = []
    rl_actions = []
    targets = []
    
    print("Running Inference...")
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # DL Prediction
            X_batch = X_batch.to(DEVICE)
            y_pred = dl_model(X_batch)
            # Remove sigmoid for raw logits if using BCEWithLogits? 
            # Wait, model has Sigmoid at end.
            # So y_pred is probability of Default.
            probs = y_pred.cpu().numpy().flatten()
            dl_probs.extend(probs)
            
            # RL Prediction
            # d3rlpy expects numpy
            # Scale? RL agent was trained on StandardScaled X from preprocessing.py.
            # Dataset X is also from same source. So X_batch is correct.
            X_np = X_batch.cpu().numpy()
            actions = cql.predict(X_np)
            rl_actions.extend(actions)
            
            targets.extend(y_batch.numpy().flatten())
            
    dl_probs = np.array(dl_probs)
    rl_actions = np.array(rl_actions)
    targets = np.array(targets)
    test_rewards = np.array(test_rewards) # Unit is Dollar
    
    # 5. Define Policies
    # DL Policy: Approve if Prob(Default) < Threshold
    # Let's find optimal threshold maximizing F1 or Value? Standard is 0.5.
    dl_threshold = 0.5
    dl_actions = (dl_probs < dl_threshold).astype(int) # 1 = Approve
    
    # RL Policy: rl_actions (1=Approve, 0=Deny)
    
    # 6. Compare Metrics
    
    # Financial Value
    # Value = Sum of (Action * Reward)
    # Note: Reward already accounts for (Approve & Paid) vs (Approve & Default).
    # If Action=0 (Deny), Gain is 0.
    
    dl_value = np.sum(dl_actions * test_rewards)
    rl_value = np.sum(rl_actions * test_rewards)
    max_possible_value = np.sum(test_rewards[test_rewards > 0]) # Oracle policy
    
    print(f"\n--- Policy Comparison (Test Set N={len(targets)}) ---")
    print(f"DL Policy Value: ${dl_value:,.2f}")
    print(f"RL Policy Value: ${rl_value:,.2f}")
    print(f"Oracle Value:    ${max_possible_value:,.2f}")
    
    # Detailed Decision Stats
    print("\nDecision Statistics:")
    print(f"DL Approval Rate: {np.mean(dl_actions):.2%}")
    print(f"RL Approval Rate: {np.mean(rl_actions):.2%}")
    
    # Agreement
    agreement = np.mean(dl_actions == rl_actions)
    print(f"Agreement Rate: {agreement:.2%}")
    
    # 7. Visualization
    # Scatter: DL Prob vs Q-Values
    # We need Q-values. cql.predict_value(x, action)
    # Let's get Q(s, Approve)
    
    # Predict value returns max Q? or Q for specific action?
    # predict_value(x, action)
    X_test_all = []
    for x, y in test_loader:
        X_test_all.append(x.cpu().numpy())
    X_test_all = np.concatenate(X_test_all, axis=0)
    
    # Q-values for Action 1 (Approve)
    # We usually pass actions as shape (N,)
    approve_actions = np.ones(len(X_test_all), dtype=np.int32)
    q_values_approve = cql.predict_value(X_test_all, approve_actions)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(dl_probs, q_values_approve, c=targets, cmap='coolwarm', alpha=0.5, s=10)
    plt.axvline(x=0.5, color='gray', linestyle='--')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.xlabel('DL Probability of Default')
    plt.ylabel('RL Q-Value (Approve)')
    plt.title('DL Risk vs RL Reward')
    plt.colorbar(label='Ground Truth (1=Default)')
    plt.savefig('risk_reward_comparison.png')
    print("Saved risk_reward_comparison.png")
    
    # Save metrics to text
    with open('model_comparison_metrics.txt', 'w') as f:
        f.write(f"DL_Value:{dl_value}\n")
        f.write(f"RL_Value:{rl_value}\n")
        f.write(f"Agreement:{agreement}\n")

if __name__ == "__main__":
    compare_models()
