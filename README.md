# Financial Policy Optimization with Reinforcement Learning

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

> **Key Insight**: The Offline RL agent suffered from **"Yield Chasing"**. Because the training data only contained _approved_ loans (Positive-Unlabeled bias), the agent learned to associate high interest rates with high rewards, ignoring the subtle risk signals that the Deep Learning model successfully captured.

---

## 🛠️ Technology Stack

- **Languages**: Python 3.10+
- **Deep Learning**: PyTorch
- **Reinforcement Learning**: `d3rlpy` (Discrete CQL), `Gymnasium`
- **Data Processing**: Pandas, NumPy, Scikit-Learn
- **Visualization**: Matplotlib, Seaborn

---

## 📈 Project Journey (Step-by-Step)

### Phase 1: Unsupervised Analysis & EDA

Before modeling, we deeply analyzed the dataset (`accepted_2007_to_2018.csv`) to understand the risk factors.

- **Findings**:
  - Default Rate: ~19.9%.
  - **Interest Rate**: Strongest correlation with default (+0.31).
  - **Income Skew**: Annual income was highly right-skewed (log-transform required).
  - **Grade**: Monotonic relationship (Grade A defaults < Grade G).

### Phase 2: Supervised Deep Learning (The Classifier)

We built a Multi-Layer Perceptron (MLP) to predict `PW(Default | Features)`.

- **v1 (Baseline)**: Standard BCE Loss. Good AUC (0.75) but low Recall (F1 0.31).
- **v2 (Advanced Features)**: Added feature engineering (ratios, log-transforms). Performance remained similar.
- **v3 (Targeted Refinement)**:
  - **Class Imbalance**: Applied `pos_weight=4.0` to the loss function to penalize missing a default 4x more than a false alarm.
  - **Outlier Clipping**: Capped income and revolving balance.
  - **Result**: F1 Score jumped to **0.45**. This model effectively filters out risky borrowers.

### Phase 3: Offline Reinforcement Learning (The Agent)

We framed the problem as a Markov Decision Process (MDP):

- **State**: 143 Preprocessed Features.
- **Action**: 0 (Deny), 1 (Approve).
- **Reward**: `Interest` (if Paid) vs `-Principal` (if Default).

#### Iteration 1: Risk Neutral (CQL)

- **Settings**: Standard Conservative Q-Learning.
- **Outcome**: Approval Rate 91%. The agent approved almost everyone, chasing the high interest rates of risky loans.
- **Value**: -$11M (Better than baseline, but worse than DL).

#### Iteration 2: Risk Sensitive (Reward Shaping)

- **Settings**: Augmented State with DL Probabilities + **5x Penalty** for Default.
- **Hypothesis**: A huge penalty would force conservatism.
- **Outcome**: Failed. Approval Rate increased to 93%. The agent ignored the penalty because it rarely saw "Deny" actions in the dataset (Behavioral Cloning bias).

#### Iteration 3: Hyperparameter Grid Search

- **Settings**: Tested 9 combinations of Penalty (1x, 2x, 5x) and Conservatism (Alpha 0.5, 2.0, 10.0).
- **Outcome**: Even the most paranoid agent (Alpha 10.0) approved ~90% of loans.
- **Conclusion**: Offline RL cannot easily unlearn the "Approve" bias from a dataset of only accepted applications.

---

## 📂 Repository Structure

```
.
├── data/                   # (Ignored) Raw and Processed data
├── logs/                   # Training logs
├── models/                 # Saved PyTorch and d3rlpy models
├── notebooks/              # Analysis and Visualization
│   └── 10_detailed_analysis.py  # Divergence Study
├── src/
│   ├── dataset.py          # PyTorch LoanDataset
│   ├── models.py           # MLP Architecture
│   ├── preprocessing.py    # Scikit-learn Pipelines
│   ├── rl_preprocessing.py # MDP Construction
│   ├── train_dl.py         # Deep Learning Training Loop
│   ├── train_rl_grid.py    # Grid Search Script
│   └── utils.py
├── requirements.txt
└── README.md
```

## 🔧 How to Run

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Preprocess Data**

   ```bash
   python src/preprocessing.py
   python src/rl_preprocessing.py
   ```

3. **Train Deep Learning Model**

   ```bash
   python src/train_dl.py
   ```

4. **Run RL Grid Search**

   ```bash
   python src/train_rl_grid_search.py
   ```

5. **Run Analysis**
   ```bash
   python notebooks/10_detailed_analysis.py
   ```


