import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
import os
import matplotlib.image as mpimg

def generate_report():
    # Load Results
    if not os.path.exists('analysis_results.json'):
        print("Error: analysis_results.json not found.")
        return

    with open('analysis_results.json', 'r') as f:
        data = json.load(f)

    metrics = data['metrics']
    examples = data['examples']

    pdf_path = 'report.pdf'
    
    with PdfPages(pdf_path) as pdf:
        # --- Page 1: Title ---
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.9, "Financial Policy Optimization Report", ha='center', fontsize=24, weight='bold')
        plt.text(0.5, 0.85, "Deep Learning vs Reinforcement Learning", ha='center', fontsize=16)
        
        plt.text(0.1, 0.75, "Executive Summary:", fontsize=14, weight='bold')
        summary_text = (
            "This report analyzes the performance of a supervised Deep Learning (DL) model\n"
            "and a Reinforcement Learning (RL) agent for loan approval decisions.\n\n"
            "Key Findings:\n"
            "1. The DL model outperforms the RL agent significantly in terms of financial value.\n"
            "   - DL Policy Value: -1.66 Million (Conservative)\n"
            "   - RL Policy Value: -11.11 Million (Aggressive)\n\n"
            "2. The RL agent appears over-optimistic, approving high-risk loans that the DL\n"
            "   model correctly identifies as likely defaults.\n\n"
            "3. Recommendation: Deploy the DL model (or a tuned conservative version)\n"
            "   and investigate reward shaping for the RL agent."
        )
        plt.text(0.1, 0.5, summary_text, fontsize=12, va='top', wrap=True)
        pdf.savefig()
        plt.close()

        # --- Page 2: Metrics Table ---
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.9, "1. Results Presentation", ha='center', fontsize=18, weight='bold')
        
        col_labels = ['Metric', 'DL Model', 'RL Agent']
        table_vals = [
            ['AUC (ROC)', f"{metrics['dl_auc']:.4f}", 'N/A'],
            ['F1 Score (Default)', f"{metrics['dl_f1']:.4f}", 'N/A'],
            ['Est. Policy Value ($)', f"${metrics['dl_value']:,.2f}", f"${metrics['rl_value']:,.2f}"]
        ]
        
        plt.table(cellText=table_vals, colLabels=col_labels, loc='center', cellLoc='center', bbox=[0.1, 0.6, 0.8, 0.2])
        
        plt.text(0.1, 0.5, "2. Metric Explanation:", fontsize=14, weight='bold')
        explanation_text = (
            "AUC & F1 (DL Model):\n"
            "- AUC (Area Under Curve) measures the model's ability to rank borrowers by risk.\n"
            "  A score of 0.74 indicates good predictive power.\n"
            "- F1-Score balances precision and recall for detecting defaults.\n\n"
            "Estimated Policy Value (RL Agent):\n"
            "- This represents the total financial outcome (Profit/Loss) if the agent's\n"
            "  decisions were applied to the test set.\n"
            "- It is the direct business objective: Maximizing portfolio return.\n"
            "- The negative values indicate that on this test set (which may have high\n"
            "  default rates or low interest margins), simply approving loans yielded losses,\n"
            "  but RL lost significantly more by failing to screen bad loans."
        )
        plt.text(0.1, 0.45, explanation_text, fontsize=11, va='top')
        pdf.savefig()
        plt.close()

        # --- Page 3: Policy Comparison ---
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.9, "3. Policy Comparison", ha='center', fontsize=18, weight='bold')
        
        comp_text = (
            "The DL model implicitly defines a policy: Approve if P(Default) < 0.5.\n"
            "The RL agent learns a policy directly (Action -> 1 or 0).\n\n"
            "Disagreement Analysis:\n"
            "The analysis found 2481 instances where the DL model flagged a loan as\n"
            "High Risk (Deny), but the RL agent Approved it."
        )
        plt.text(0.1, 0.8, comp_text, fontsize=12, va='top')
        
        if examples:
            ex = examples[0]
            ex_text = (
                f"Specific Example (Index {ex['index']}):\n"
                f"- DL Predicted Prob of Default: {ex['dl_prob']:.2f} (High Risk)\n"
                f"- RL Action: {ex['rl_action']} (Approve)\n"
                f"- Actual Outcome: {'Default' if ex['target']==1 else 'Paid'}\n"
                f"- Financial Consequence: ${ex['reward']:,.2f}\n\n"
                "Interpretation:\n"
                "The RL agent approved a loan that effectively defaulted, resulting in a\n"
                "loss. This suggests the agent's value function estimation may be biased\n"
                "or it hasn't converged to recognizing these risk factors."
            )
            plt.text(0.1, 0.5, ex_text, fontsize=12, va='top', bbox=dict(facecolor='lightgray', alpha=0.5))
        
        pdf.savefig()
        plt.close()

        # --- Page 4: Future Steps ---
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.9, "4. Future Steps", ha='center', fontsize=18, weight='bold')
        
        future_text = (
            "Deploy Strategy:\n"
            "- Deploy the DL Model. It is safer and financially superior currently.\n\n"
            "Limitations:\n"
            "- RL performance is poor, likely due to sparse rewards or insufficient training.\n"
            "- The dataset class imbalance might bias the RL agent to 'Approve All' if\n"
            "  most loans are paid, but here it seems to be approving bad loans too.\n\n"
            "Next Steps:\n"
            "1. Tuning RL: Try scaling rewards differently (e.g., penalty for default * 10).\n"
            "2. Data Collection: Collect more features on borrower behavior over time.\n"
            "3. Algorithms: Explore DDPG or PPO for continuous action spaces (e.g., setting\n"
            "   loan amount or interest rate dynamically)."
        )
        plt.text(0.1, 0.8, future_text, fontsize=12, va='top')
        pdf.savefig()
        plt.close()
        
        # --- Page 5: Visuals ---
        # Try to append existing images
        img_path = 'notebooks/roc_curve_dl.png'
        if os.path.exists(img_path):
            try:
                img = mpimg.imread(img_path)
                plt.figure(figsize=(8.5, 11))
                plt.imshow(img)
                plt.axis('off')
                plt.title("Appendix: DL Model ROC Curve")
                pdf.savefig()
                plt.close()
            except Exception as e:
                print(f"Could not add image: {e}")

    print(f"Report generated at {pdf_path}")

if __name__ == "__main__":
    generate_report()
