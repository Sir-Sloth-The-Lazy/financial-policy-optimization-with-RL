import d3rlpy
import numpy as np
import pandas as pd
import os
import sys
import torch

# Add src to path
sys.path.append(os.path.dirname(__file__))
from preprocessing import load_and_engineer_data

def run_grid_search():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data/processed')
    RAW_PATH = os.path.join(BASE_DIR, 'data/raw/accepted_2007_to_2018.csv')
    MODEL_DIR = os.path.join(BASE_DIR, 'models/grid_search')
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 1. Load Data
    print("Loading Data for Grid Search...")
    # Load DF for Reward Calculation
    df = load_and_engineer_data(RAW_PATH, sample_size=None)
    
    # Load Features: X_risk_aware (144 dim)
    # Ensure augment_with_dl.py was run (Task 7)
    X_path = os.path.join(DATA_DIR, 'X_risk_aware.npy')
    if not os.path.exists(X_path):
        print("X_risk_aware.npy not found! Please run src/augment_with_dl.py first.")
        return
        
    X = np.load(X_path).astype(np.float32)
    y = np.load(os.path.join(DATA_DIR, 'y.npy'))
    
    # Split Indices (Replicate same split)
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    split_ratio = (0.7, 0.15, 0.15)
    train_end = int(len(X) * split_ratio[0])
    test_indices = indices[int(len(X) * (split_ratio[0] + split_ratio[1])):]
    
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    # 2. Grid Config
    PENALTIES = [1.0, 2.0, 5.0]
    ALPHAS = [0.5, 2.0, 10.0]
    
    results = []
    
    print(f"Starting Grid Search. Total combinations: {len(PENALTIES) * len(ALPHAS)}")
    
    for p in PENALTIES:
        # Pre-calculate Rewards for this penalty
        # This is strictly related to how we construct the TRAINING dataset.
        train_rewards = []
        for idx, row in df.iterrows():
            if row['loan_status'] == 'Fully Paid':
                r = row['loan_amnt'] * (row['int_rate'] / 100.0)
            else:
                r = -p * row['loan_amnt']
            train_rewards.append(r)
        
        train_rewards = np.array(train_rewards, dtype=np.float32)
        
        # Build Augmented Dataset (Approved + Denied)
        # Approved part
        obs_app = X
        act_app = np.ones(len(X), dtype=np.int32)
        rew_app = train_rewards
        term_app = np.ones(len(X), dtype=np.float32)
        
        # Denied part (Synthetic)
        obs_den = X
        act_den = np.zeros(len(X), dtype=np.int32)
        rew_den = np.zeros(len(X), dtype=np.float32) # Always 0
        term_den = np.ones(len(X), dtype=np.float32)
        
        # Combine
        obs_full = np.concatenate([obs_app, obs_den], axis=0)
        act_full = np.concatenate([act_app, act_den], axis=0)
        rew_full = np.concatenate([rew_app, rew_den], axis=0)
        term_full = np.concatenate([term_app, term_den], axis=0)
        
        # Scale Rewards for Training Stability
        rew_scaled = rew_full / 1000.0
        
        dataset = d3rlpy.dataset.MDPDataset(
            observations=obs_full,
            actions=act_full,
            rewards=rew_scaled,
            terminals=term_full,
        )
        
        # Inner Loop: Alpha
        for alpha in ALPHAS:
            model_name = f"cql_p{p}_a{alpha}"
            print(f"\nTraining {model_name}...")
            
            cql = d3rlpy.algos.DiscreteCQLConfig(
                learning_rate=3e-4,
                alpha=alpha,
                batch_size=256,
            ).create(device='cpu') # Use CPU for stability logic, GPU if faster
            
            # Train (Short run for Grid Search)
            # 5000 steps is enough to see convergence/direction given dataset size
            # Train (Short run for Grid Search)
            # 5000 steps is enough to see convergence/direction given dataset size
            cql.fit(
                dataset,
                n_steps=5000,
                n_steps_per_epoch=1000,
            )
            
            # Evaluate on Test Set
            # Predict Actions
            # Note: Test set only has X_test (real applicants). 
            test_actions = cql.predict(X_test)
            
            approval_rate = np.mean(test_actions)
            
            # Approval Rate on Defaults (Bad Approval Rate)
            bad_indices = (y_test == 1)
            bad_approval_rate = np.mean(test_actions[bad_indices])
            
            # Policy Value (Normalized Estimate)
            # We calculate this using the USER specified Penalty P for consistency?
            # Or always using P=1 (Profit)?
            # Usually, verify against P=1 (Real Dollars) and P=User (Optimization Target).
            # Let's track REAL DOLLARS (P=1)
            
            # Calculate P=1 Rewards for Test Set
            # We need to map test indices back to DF
            test_df = df.iloc[test_indices]
            val_real_usd = 0.0
            for i, (idx, row) in enumerate(test_df.iterrows()):
                action = test_actions[i]
                if action == 1:
                    if row['loan_status'] == 'Fully Paid':
                        val_real_usd += row['loan_amnt'] * (row['int_rate'] / 100.0)
                    else:
                        val_real_usd -= row['loan_amnt'] # Real loss is P=1
            
            print(f"  -> Approval: {approval_rate:.2%}")
            print(f"  -> Bad Rate: {bad_approval_rate:.2%}")
            print(f"  -> Value ($): {val_real_usd:,.0f}")
            
            # Save
            cql.save_model(os.path.join(MODEL_DIR, f"{model_name}.d3"))
            
            results.append({
                "Penalty": p,
                "Alpha": alpha,
                "Approval_Rate": approval_rate,
                "Bad_Approval_Rate": bad_approval_rate,
                "Policy_Value_USD": val_real_usd
            })
            
    # Save Results
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(MODEL_DIR, "grid_results.csv"), index=False)
    print("\n--- Grid Search Complete ---")
    print(res_df.sort_values("Policy_Value_USD", ascending=False))

if __name__ == "__main__":
    run_grid_search()
