import numpy as np
import pandas as pd
import os
import sys

# Add src to path to import preprocessing
sys.path.append(os.path.dirname(__file__))
from preprocessing import load_and_engineer_data

def prepare_rl_data(data_dir, raw_path, sample_size=100000):
    print("Loading engineered data for reward calculation...")
    # Ensure this matches exactly how X.npy was generated
    df = load_and_engineer_data(raw_path, sample_size)
    
    # Calculate Rewards
    # Reward = loan_amnt * (int_rate / 100) if Fully Paid
    # Reward = -loan_amnt if Charged Off
    rewards = []
    
    # Check consistency with saved X
    # X_path = os.path.join(data_dir, 'X.npy')
    # if os.path.exists(X_path):
    #     X = np.load(X_path)
    #     assert len(X) == len(df), f"Shape mismatch! X: {len(X)}, DF: {len(df)}"

    for idx, row in df.iterrows():
        # loan_status has been filtered to only Fully Paid / Charged Off
        if row['loan_status'] == 'Fully Paid':
            # Simplified profit: Total interest gained
            # Note: Ideally we'd use 'total_rec_int' from dataset if available, 
            # but user prompt specified: loan_amnt * (int_rate)
            # int_rate is percentage e.g. 13.5
            r = row['loan_amnt'] * (row['int_rate'] / 100.0)
        else:
            # Loss of principal
            # User specified: - loan_amnt
            r = -row['loan_amnt']
        rewards.append(r)
        
    rewards = np.array(rewards, dtype=np.float32)
    
    # Actions
    # All rows in this dataset are accepted loans => Action = 1 (Approve)
    actions = np.ones(len(df), dtype=np.int32)
    
    # Terminals
    # All are terminal steps (one-shot decision)
    terminals = np.ones(len(df), dtype=np.float32)
    
    # AUGMENTATION: Add "Deny" actions (Action=0, Reward=0)
    # Since we know Deny always yields 0, we can add these transitions to the dataset
    # so the agent learns Q(s, 0) -> 0 effectively.
    # Otherwise, with only Action 1 in data, Q(s, 0) is undefined/random.
    
    X_processed = np.load(os.path.join(data_dir, 'X.npy')) # Load X to match
    X_augs = X_processed
    actions_augs = np.zeros(len(df), dtype=np.int32)
    rewards_augs = np.zeros(len(df), dtype=np.float32)
    terminals_augs = np.ones(len(df), dtype=np.float32)
    
    # Concatenate
    X_final = np.concatenate([X_processed, X_augs], axis=0)
    actions_final = np.concatenate([actions, actions_augs], axis=0)
    rewards_final = np.concatenate([rewards, rewards_augs], axis=0)
    terminals_final = np.concatenate([terminals, terminals_augs], axis=0)
    
    # Save
    # Helper to save combined for d3rlpy
    # We will overwrite the previous inputs or save as 'rl_*.npy'
    np.save(os.path.join(data_dir, 'rl_observations.npy'), X_final)
    np.save(os.path.join(data_dir, 'rl_actions.npy'), actions_final)
    np.save(os.path.join(data_dir, 'rl_rewards.npy'), rewards_final)
    np.save(os.path.join(data_dir, 'rl_terminals.npy'), terminals_final)
    
    print(f"Saved RL data to {data_dir}")
    print(f"Count: {len(rewards)}")
    print(f"Avg Reward: {np.mean(rewards):.2f}")
    print(f"Min Reward: {np.min(rewards):.2f}")
    print(f"Max Reward: {np.max(rewards):.2f}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    RAW_DATA = os.path.join(BASE_DIR, 'data/raw/accepted_2007_to_2018.csv')
    PROCESSED_DIR = os.path.join(BASE_DIR, 'data/processed')
    
    prepare_rl_data(PROCESSED_DIR, RAW_DATA, sample_size=None)
