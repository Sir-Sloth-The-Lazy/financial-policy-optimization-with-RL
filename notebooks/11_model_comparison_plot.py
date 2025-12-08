import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def plot_comparison():
    # Data from project findings
    data = {
        'Model': [
            'Baseline (Approve All)',
            'RL v2 (Risk Sensitive)',
            'RL v1 (CQL)', 
            'RL Grid (Best)',
            'Deep Learning (v3)'
        ],
        'Policy Value ($M)': [
            -26.0,
            -15.14,
            -11.11, 
            -12.0, 
            -1.66
        ],
        'Approval Rate (%)': [
            100.0,
            93.21,
            91.28, 
            90.0, 
            57.70
        ],
        'Type': [
            'Unsupervised (Baseline)',
            'Reinforcement Learning',
            'Reinforcement Learning',
            'Reinforcement Learning',
            'Supervised (Deep Learning)'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Setup plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Bar plot for Policy Value
    palette = {
        'Unsupervised (Baseline)': 'gray', 
        'Reinforcement Learning': 'salmon', 
        'Supervised (Deep Learning)': 'royalblue'
    }
    sns.barplot(
        data=df, 
        y='Model', 
        x='Policy Value ($M)', 
        hue='Type', 
        palette=palette, 
        ax=ax1,
        dodge=False
    )
    
    # Add value labels
    for i, v in enumerate(df['Policy Value ($M)']):
        ax1.text(v - 0.5, i, f"${v}M", va='center', ha='right', fontweight='bold')
        
    ax1.set_title('Model Performance Comparison: Policy Value (Profit/Loss)', fontsize=14)
    ax1.set_xlabel('Estimated Portfolio Value (Million $)', fontsize=12)
    ax1.axvline(0, color='black', linewidth=1)
    ax1.set_xlim(-30, 0)
    
    # Create output path
    base_dir = os.path.dirname(os.path.dirname(__file__))
    output_path = os.path.join(base_dir, 'model_comparison.png')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved comparison plot to {output_path}")

if __name__ == "__main__":
    plot_comparison()
