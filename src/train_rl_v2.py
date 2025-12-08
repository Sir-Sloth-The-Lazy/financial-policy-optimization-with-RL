import d3rlpy
import numpy as np
import os
import torch

def train_rl_agent_v2():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data/processed')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    print("Loading RL V2 dataset...")
    observations = np.load(os.path.join(DATA_DIR, 'rl_v2_observations.npy'))
    actions = np.load(os.path.join(DATA_DIR, 'rl_v2_actions.npy'))
    rewards = np.load(os.path.join(DATA_DIR, 'rl_v2_rewards.npy'))
    terminals = np.load(os.path.join(DATA_DIR, 'rl_v2_terminals.npy'))
    
    # Scale Rewards: divide by 1000 again
    rewards_scaled = rewards / 1000.0
    
    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards_scaled,
        terminals=terminals,
    )
    
    # Increase Alpha for more conservatism? 
    # Or rely on the reward shaping.
    cql = d3rlpy.algos.DiscreteCQLConfig(
        learning_rate=3e-4,
        alpha=2.0, # Increased from 1.0 to be more conservative
        batch_size=256,
    ).create(device='cpu')
    
    print("Fitting Risk-Sensitive CQL Agent...")
    cql.fit(
        dataset,
        n_steps=10000,
        n_steps_per_epoch=1000,
        save_interval=10,
    )
    
    save_path = os.path.join(MODEL_DIR, 'cql_agent_risk_aware.d3')
    cql.save_model(save_path)
    print(f"Saved Risk-Aware Agent to {save_path}")

    # --- Quick Check ---
    y = np.load(os.path.join(DATA_DIR, 'y.npy'))
    real_len = len(y)
    real_obs = observations[:real_len]
    
    pred_actions = cql.predict(real_obs)
    approval_rate = np.mean(pred_actions)
    rate_bad = np.mean(pred_actions[y == 1])
    
    print("\n--- Risk-Aware Agent Stats ---")
    print(f"Approval Rate: {approval_rate:.2%}")
    print(f"Approval Rate on Defaults: {rate_bad:.2%}")

if __name__ == "__main__":
    train_rl_agent_v2()
