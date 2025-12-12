import d3rlpy
import numpy as np
import os
import torch

def train_rl_agent():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data/processed')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    print("Loading RL dataset...")
    observations = np.load(os.path.join(DATA_DIR, 'rl_observations.npy'))
    actions = np.load(os.path.join(DATA_DIR, 'rl_actions.npy'))
    rewards = np.load(os.path.join(DATA_DIR, 'rl_rewards.npy'))
    terminals = np.load(os.path.join(DATA_DIR, 'rl_terminals.npy'))
    
    # Scale Rewards: divide by 1000 so values are in range ~[-35, 10] instead of [-35000, 10000]
    # This helps gradient stability.
    rewards_scaled = rewards / 1000.0
    
    # Create MDP Dataset
    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards_scaled,
        terminals=terminals,
    )
    
    # Define Algorithm: Discrete CQL
    # We use a Config object as per d3rlpy v2
    cql = d3rlpy.algos.DiscreteCQLConfig(
        learning_rate=3e-4,
        alpha=1.0, # Conservative Regularization weight
        batch_size=256,
    ).create(device='cpu') 
    
    print("Fitting CQL Agent for 10,000 steps...")
    
    # Training
    cql.fit(
        dataset,
        n_steps=10000,
        n_steps_per_epoch=1000,
        save_interval=10,
    )
    
    # Save
    save_path = os.path.join(MODEL_DIR, 'cql_agent.d3')
    cql.save_model(save_path)
    print(f"Saved CQL Agent to {save_path}")
    
    # ------------------
    # Quick Evaluation
    # ------------------
    # Load ground truth targets to check behavior
    y_path = os.path.join(DATA_DIR, 'y.npy')
    if os.path.exists(y_path):
        y = np.load(y_path)
        # The RL dataset (augmented) has doubled size. The first half corresponds to real data.
        real_len = len(y)
        
        # Predict on real data
        real_obs = observations[:real_len]
        pred_actions = cql.predict(real_obs)
        
        # y=1 is Charge Off (Default), y=0 is Fully Paid
        # RL Action 1 = Approve, 0 = Deny
        
        # Approval Rates
        total_approved = np.mean(pred_actions)
        
        # Approval Rate on BAD loans (y=1)
        approvals_on_bad = pred_actions[y == 1]
        rate_bad = np.mean(approvals_on_bad)
        
        # Approval Rate on GOOD loans (y=0)
        approvals_on_good = pred_actions[y == 0]
        rate_good = np.mean(approvals_on_good)
        
        print("\n--- RL Agent Preliminary Analysis ---")
        print(f"Overall Approval Rate: {total_approved:.2%}")
        print(f"Approval Rate on Defaults (Bad): {rate_bad:.2%}")
        print(f"Approval Rate on Paid (Good): {rate_good:.2%}")
        
        # Compare to naive "Approve All" (which was the historical policy)
        # History: 100% Approved. 
        # If RL approves < 100% of Good loans, it's conservative (losing opportunity).
        # If RL approves < 100% of Bad loans, it's saving money.
        
        # Calculate approximate Policy Value (Normalized)
        # Value = Sum of rewards for approved loans
        # We use unscaled rewards for interpretation
        real_rewards = rewards[:real_len]
        
        # Historical Value (Approving everything in the dataset)
        hist_value = np.sum(real_rewards)
        
        # RL Value (Approving only what RL selected)
        # If pred_action is 1, we get the reward. If 0, we get 0.
        rl_value = np.sum(real_rewards * pred_actions)
        
        print(f"\nHistorical Portfolio Value: ${hist_value:,.2f}")
        print(f"RL Portfolio Value:       ${rl_value:,.2f}")
        print(f"Improvement:              ${rl_value - hist_value:,.2f}")

if __name__ == "__main__":
    train_rl_agent()
