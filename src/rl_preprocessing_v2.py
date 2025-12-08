import numpy as np
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(__file__))
from preprocessing import load_and_engineer_data

def prepare_rl_data_v2(data_dir, raw_path, sample_size=100000):
    print("Loading engineered data for reward calculation (V2)...")
    # We load df just for Loan Amount and Interest Rate
    df = load_and_engineer_data(raw_path, sample_size=sample_size)
    
    # Load Risk-Aware Features
    X_path = os.path.join(data_dir, 'X_risk_aware.npy')
    if not os.path.exists(X_path):
        raise FileNotFoundError("Run src/augment_with_dl.py first!")
    
    X = np.load(X_path)
    # Check alignment
    assert len(X) == len(df)

    rewards = []
    
    # REWARD SHAPING
    # Penalty Multiplier
    PENALTY_FACTOR = 5.0
    
    for idx, row in df.iterrows():
        if row['loan_status'] == 'Fully Paid':
            # Reward = Interest
            r = row['loan_amnt'] * (row['int_rate'] / 100.0)
        else:
            # Reward = - Penalty * Principal
            r = -PENALTY_FACTOR * row['loan_amnt']
        rewards.append(r)
        
    rewards = np.array(rewards, dtype=np.float32)
    actions = np.ones(len(df), dtype=np.int32)
    terminals = np.ones(len(df), dtype=np.float32)
    
    # AUGMENTATION with Deny (Action=0, Reward=0)
    # This is crucial for CQL to learn the alternative
    X_augs = X
    actions_augs = np.zeros(len(df), dtype=np.int32)
    rewards_augs = np.zeros(len(df), dtype=np.float32)
    terminals_augs = np.ones(len(df), dtype=np.float32)
    
    # Concatenate
    X_final = np.concatenate([X, X_augs], axis=0)
    actions_final = np.concatenate([actions, actions_augs], axis=0)
    rewards_final = np.concatenate([rewards, rewards_augs], axis=0)
    terminals_final = np.concatenate([terminals, terminals_augs], axis=0)
    
    # Save as V2
    np.save(os.path.join(data_dir, 'rl_v2_observations.npy'), X_final)
    np.save(os.path.join(data_dir, 'rl_v2_actions.npy'), actions_final)
    np.save(os.path.join(data_dir, 'rl_v2_rewards.npy'), rewards_final)
    np.save(os.path.join(data_dir, 'rl_v2_terminals.npy'), terminals_final)
    
    print(f"Saved RL V2 (Risk-Aware) data to {data_dir}")
    print(f"Avg Reward (Scaled Penalty): {np.mean(rewards):.2f}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    RAW_DATA = os.path.join(BASE_DIR, 'data/raw/accepted_2007_to_2018.csv')
    PROCESSED_DIR = os.path.join(BASE_DIR, 'data/processed')
    
    prepare_rl_data_v2(PROCESSED_DIR, RAW_DATA, sample_size=None)
