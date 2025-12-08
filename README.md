# Financial Policy Optimization with Reinforcement Learning

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![d3rlpy](https://img.shields.io/badge/d3rlpy-Offline_RL-blue)
![Colab](https://img.shields.io/badge/Google_Colab-A100_Ready-orange?logo=googlecolab)
![Status](https://img.shields.io/badge/Status-Complete-green)

A comprehensive project exploring the transition from **Supervised Deep Learning** to **Offline Reinforcement Learning** for optimizing loan approval policies.

## 📌 Project Overview

The goal of this project is to build an automated decision system for loan approvals. We aim to minimize defaults while maximizing interest profit.
We compare two distinct approaches:

1.  **Deep Learning (Supervised)**: Predicting the _Probability of Default_ and thresholding it.
2.  **Offline Reinforcement Learning (CQL)**: Learning a policy directly from historical data to maximize _Profit_.

---

## 🚀 Key Results

| Metric            |    Deep Learning (v3)     | RL Agent (v3 - Best) | Baseline (Approve All) |
| :---------------- | :-----------------------: | :------------------: | :--------------------: |
| **Approach**      |     Weighted BCE Loss     |   Grid-Search CQL    |         Naive          |
| **Approval Rate** | **57.70%** (Conservative) |  ~90% (Aggressive)   |          100%          |
| **F1 Score**      |  **0.45** (High Recall)   |         N/A          |          0.31          |
| **Policy Value**  |        **-$1.66M**        |       -$12.0M        |         -$26M          |
| **Verdict**       |      🏆 **Champion**      |       ⚠️ Risky       |        ❌ Fail         |

> **Key Insight**: The Offline RL agent suffered from **"Yield Chasing"**. Because the training data only contained \*approved\_ loans, the agent learned to associate high interest rates with high rewards, ignoring the subtle risk signals that the Deep Learning model successfully captured.

---

## 🛠️ Technology Stack

- **Deep Learning**: `PyTorch` (MLP Classifier)
- **Reinforcement Learning**: `d3rlpy` (Discrete CQL), `Gymnasium`
- **Data Pipeline**: `Pandas`, `NumPy`, `Scikit-Learn` (Imputation, Scaling, Encoding)
- **Visualization**: `Matplotlib`, `Seaborn`

---

## 📈 Project Journey

### Phase 1: Unsupervised Analysis & EDA

Analysing `accepted_2007_to_2018.csv` (2.2M rows total).

- **Findings**:
  - Default Rate: ~19.9%.
  - **Interest Rate**: Strongest correlation with default (+0.31).
  - **Graded Risk**: Monotonic relationship (Grade A < G).

### Phase 2: Supervised Deep Learning

Built an MLP to predict `PW(Default | Features)`.

- **Optimization**: Used `pos_weight=4.0` to handle class imbalance (defaults are rare but costly).
- **Result**: F1 Score **0.45**. Effective at filtering high-risk borrowers.

### Phase 3: Offline Reinforcement Learning

Framed as an MDP (State: 149 features, Action: Approve/Deny, Reward: Profit/Loss).

- **CQL (Conservative Q-Learning)**: Tested various "Alpha" values.
- **Fail Mode**: The agent struggled to learn conservatism because the dataset lacked "Deny" examples (Behavioral Bias). Even with synthetic penalties, it prioritized high-yield loans.

---

## 📂 Repository Structure

```bash
.
├── notebooks/
│   ├── colab_runner.ipynb       # <--- START HERE for A100 Run
│   └── 10_detailed_analysis.py  # Divergence Study
├── src/
│   ├── models.py           # PyTorch Architecture
│   ├── train_dl.py         # Deep Learning Training
│   ├── train_rl_grid.py    # RL Grid Search
│   ├── preprocessing.py    # Scikit-learn Pipeline
│   └── ...
├── requirements.txt
└── README.md
```

---

## 🔧 How to Run

### Option A: Google Colab (Recommended for A100)

1.  Open [notebooks/colab_runner.ipynb](notebooks/colab_runner.ipynb) in Google Colab.
2.  Upload `accepted_2007_to_2018.csv` to the session storage.
3.  Run all cells. This will execute the full pipeline (Preprocessing -> DL Training -> RL Training).

### Option B: Local Execution

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run Pipeline (Full Dataset)**

   ```bash
   python src/preprocessing.py
   python src/rl_preprocessing.py
   ```

3. **Train Models**

   ```bash
   python src/train_dl.py            # Train Deep Learning Model
   python src/augment_with_dl.py     # Add Risk Scores to Data
   python src/train_rl_grid_search.py # Run RL Grid Search
   ```

4. **Run Analysis**
   ```bash
   python notebooks/10_detailed_analysis.py
   ```

---

_Created by Sir Sloth._
