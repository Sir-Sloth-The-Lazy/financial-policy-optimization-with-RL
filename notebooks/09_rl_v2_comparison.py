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

def compare_models_v2():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data/processed')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    # 1. Load Data
    # For v2, we need X_risk_aware. But Dataset class loads X.npy.
    # We must construct the augmented features manually for the test set.
    
    # Load raw X and y
    X = np.load(os.path.join(DATA_DIR, 'X.npy')).astype(np.float32)
    y = np.load(os.path.join(DATA_DIR, 'y.npy')) # labels
    probs = np.load(os.path.join(DATA_DIR, 'X_risk_aware.npy'))[:, -1:] # last col is prob
    
    # Reconstruct same test split
    # LoanDataset uses deterministic shuffle with seed 42
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split_ratio = (0.7, 0.15, 0.15)
    train_end = int(len(X) * split_ratio[0])
    val_end = int(len(X) * (split_ratio[0] + split_ratio[1]))
    test_indices = indices[val_end:]
    
    X_test = X[test_indices]
    probs_test = probs[test_indices]
    y_test = y[test_indices]
    
    # Augmented X for RL
    X_test_aug = np.concatenate([X_test, probs_test], axis=1)
    
    # Test Rewards (Unscaled, Risk-Neutral for fairness OR Risk-Averse?)
    # We should evaluate on the OBJECTIVE metric.
    # User asked to improve RL model.
    # Let's evaluate on:
    # 1. Risk-Neutral Value (Real dollars)
    # 2. Risk-Averse Value (Metric we optimized)
    
    # We need to recalculate rewards for test set indices
    # We can load the rewards/loan_amnts arrays we saved in rl_preprocessing.py? No, that was full set.
    # Let's load 'rl_rewards.npy' (v1 - risk neutral) and 'rl_v2_rewards.npy' (v2 - risk averse)
    
    rl_rewards_all = np.load(os.path.join(DATA_DIR, 'rl_rewards.npy'))
    original_count = len(rl_rewards_all) // 2
    r1_test = rl_rewards_all[:original_count][test_indices]
    
    rl_v2_rewards_all = np.load(os.path.join(DATA_DIR, 'rl_v2_rewards.npy'))
    r2_test = rl_v2_rewards_all[:original_count][test_indices]
    
    # 2. Load Models
    cql_v2 = d3rlpy.algos.DiscreteCQLConfig().create(device='cpu')
    
    # Build dummy
    dummy_obs = np.random.random((2, X_test_aug.shape[1])).astype(np.float32)
    dummy_actions = np.array([0, 1], dtype=np.int32)
    dummy_rewards = np.array([0, 0], dtype=np.float32)
    dummy_terminals = np.array([1, 1], dtype=np.float32)
    dummy_dataset = d3rlpy.dataset.MDPDataset(dummy_obs, dummy_actions, dummy_rewards, dummy_terminals)
    cql_v2.build_with_dataset(dummy_dataset)
    
    cql_v2.load_model(os.path.join(MODEL_DIR, 'cql_agent_risk_aware.d3'))
    
    # 3. Predict
    rl_actions_v2 = cql_v2.predict(X_test_aug)
    
    # 4. Metrics
    
    # A. Policy Value (Risk-Neutral Dollars)
    # Using r1_test where default is -Principal
    val_neutral = np.sum(rl_actions_v2 * r1_test)
    
    # B. Policy Value (Risk-Averse)
    # Using r2_test where default is -5 * Principal
    val_averse = np.sum(rl_actions_v2 * r2_test)
    
    print(f"\n--- RL V2 (Risk-Aware) Evaluation (N={len(y_test)}) ---")
    print(f"Approval Rate: {np.mean(rl_actions_v2):.2%}")
    print(f"Risk-Neutral Value (Real $): ${val_neutral:,.2f}")
    print(f"Risk-Averse Value (Optimization Target): ${val_averse:,.2f}")
    
    # Comparison with V1 (From previous run logic)
    # We know V1 approved ~91% and had Value -$11M (Neutral)
    
    # Check what kind of loans it approves
    approved_mask = rl_actions_v2 == 1
    approved_defaults = y_test[approved_mask]
    default_rate = np.mean(approved_defaults)
    print(f"Default Rate among Approved: {default_rate:.2%} (Base rate: {np.mean(y_test):.2%})")
    
if __name__ == "__main__":
    compare_models_v2()
