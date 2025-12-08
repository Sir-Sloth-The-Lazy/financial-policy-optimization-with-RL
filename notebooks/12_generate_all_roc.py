import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import d3rlpy

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/src')
from models import LoanClassifier

def generate_roc_comparison():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data/processed')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    # 1. Load Test Data
    print("Loading Data...")
    X = np.load(os.path.join(DATA_DIR, 'X.npy')).astype(np.float32)
    y = np.load(os.path.join(DATA_DIR, 'y.npy'))
    
    # Determine Test Split (last 15%)
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    split_ratio = (0.7, 0.15, 0.15)
    test_indices = indices[int(len(X) * (split_ratio[0] + split_ratio[1])):]
    
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    # Risk-Aware Features for RL v2
    # If X_risk_aware exists, we need it for v2 models
    X_risk_path = os.path.join(DATA_DIR, 'X_risk_aware.npy')
    has_risk_aware = os.path.exists(X_risk_path)
    if has_risk_aware:
        X_risk = np.load(X_risk_path).astype(np.float32)
        X_risk_test = X_risk[test_indices]
    
    plt.figure(figsize=(10, 8))
    
    # ==========================
    # 2. Deep Learning Model
    # ==========================
    print("Evaluating Deep Learning Model...")
    input_dim = X.shape[1]
    dl_model = LoanClassifier(input_dim, hidden_dim=128, dropout_rate=0.3)
    dl_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_dl_model.pth'), map_location='cpu'))
    dl_model.eval()
    
    with torch.no_grad():
        logits = dl_model(torch.from_numpy(X_test))
        probs = torch.sigmoid(logits).numpy().flatten()
        
    fpr_dl, tpr_dl, _ = roc_curve(y_test, probs)
    auc_dl = auc(fpr_dl, tpr_dl)
    plt.plot(fpr_dl, tpr_dl, label=f'Supervised DL (AUC = {auc_dl:.3f})', linewidth=2, color='royalblue')
    
    # ==========================
    # 3. RL Models
    # ==========================
    # Helper to plot RL ROC
    def plot_rl_roc(model_path, label, color, use_risk_features=False):
        if not os.path.exists(model_path):
            print(f"Skipping {label} (Model not found)")
            return
            
        print(f"Evaluating {label}...")
        
        # Determine Input Shape
        if use_risk_features:
            if not has_risk_aware:
                print(f"Skipping {label} (Missing X_risk_aware)")
                return
            current_X = X_risk_test
            obs_shape = (X_risk.shape[1],)
        else:
            current_X = X_test
            obs_shape = (X.shape[1],)
            
        # Reconstruct Model (Weights Only) using Dummy Fit
        try:
            cql = d3rlpy.algos.DiscreteCQLConfig(batch_size=256).create(device='cpu')
            
            # Dummy Fit to initialize network info
            # We need a tiny dataset that matches the observation shape
            dummy_obs = np.zeros((2, obs_shape[0]), dtype=np.float32)
            dummy_act = np.zeros((2, 1), dtype=np.int32).flatten() # Discrete actions
            dummy_rew = np.zeros(2, dtype=np.float32)
            dummy_term = np.zeros(2, dtype=np.float32)
            dummy_term[-1] = 1.0 # Requirement: At least one terminal
            
            # Note: MDPDataset expects action as (N, 1) or (N,) depending on discrete
            # Discrete: (N,)
            dataset = d3rlpy.dataset.MDPDataset(
                observations=dummy_obs,
                actions=dummy_act,
                rewards=dummy_rew,
                terminals=dummy_term
            )
            
            # Run 1 step to build impl
            cql.fit(dataset, n_steps=1)
            
            # Now load real weights
            cql.load_model(model_path)
            
            # Predict
            q_values = cql.predict_value(current_X)
            score_risk = q_values[:, 0] - q_values[:, 1] # Q(Deny) - Q(Approve)
            
            fpr, tpr, _ = roc_curve(y_test, score_risk)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})', linestyle='--', color=color)
            
        except Exception as e:
            print(f"Error loading {label}: {e}")

    # RL v1 (Neutral) - Uses standard X
    plot_rl_roc(os.path.join(MODEL_DIR, 'cql_agent.d3'), 'RL v1 (Neutral)', 'salmon', use_risk_features=False)
    
    # RL v2 / Grid - Uses Risk Aware
    if has_risk_aware:
        # Check specific grid model, or fallbacks
        grid_best_path = os.path.join(MODEL_DIR, 'grid_search/cql_p5.0_a10.0.d3')
        if os.path.exists(grid_best_path):
            plot_rl_roc(grid_best_path, 'RL Grid Best (Strict)', 'darkred', use_risk_features=True)
        else:
            # Try another one if that doesn't exist?
            pass

    # Plot formatting
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (False Alarm)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve Comparison: Supervised vs RL')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    save_path = 'roc_comparison_all.png'
    plt.savefig(save_path, dpi=300)
    print(f"Saved ROC comparison to {save_path}")

if __name__ == "__main__":
    generate_roc_comparison()
