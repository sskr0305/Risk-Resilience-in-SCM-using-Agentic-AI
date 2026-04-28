# Supply Chain Risk Resilience Using Agentic AI

This repository contains the notebook **Supply_Chain_Risk_AI.ipynb**, developed as part of the course **Industrial Management (IM41081)**.[file:1] It implements an AI-powered supply chain risk management system with an agentic AI risk monitor built on top of trained machine learning models.[file:1]

---

## 1. Project Overview

The goal of this project is to simulate a realistic supply chain, train AI models to classify **risk levels (Low / Medium / High)**, and then wrap the best model in a **Sense–Reason–Act** agent that monitors scenarios and recommends corrective actions.[file:1]

Main capabilities:[file:1]

- Generate a synthetic, realistic supply chain dataset.
- Train and evaluate multiple ML models for multi-class risk prediction.
- Analyze feature importance to identify key risk drivers.
- Run an **Agentic AI Risk Monitor** that:
  - Senses the current supply chain state.
  - Reasons about risk level using the trained model.
  - Acts by recommending playbook actions and logging decisions.

---

## 2. Project Pipeline

The notebook is organized into nine main steps:[file:1]

1. **Import Libraries**  
   Imports NumPy, pandas, Matplotlib, Seaborn, scikit-learn (RandomForestClassifier, GradientBoostingClassifier, LogisticRegression, train_test_split, pipelines, metrics), joblib, json, random, and datetime utilities.[file:1]

2. **Generate Synthetic Supply Chain Dataset**  
   - Creates a dataset with **2000 rows × 11 columns**.[file:1]  
   - Features:[file:1]  
     - `supplier_reliability` (0–1)  
     - `delivery_delay_days`  
     - `demand_variability` (0–1)  
     - `inventory_level` (0–1)  
     - `geopolitical_risk` (0–1)  
     - `transportation_cost`  
     - `quality_defect_rate` (0–1)  
     - `lead_time_days`  
     - `num_suppliers`  
     - `weather_risk` (0–1)  
   - Target: `risk_level` ∈ {0: Low, 1: Medium, 2: High}, derived from a composite risk score and percentile thresholds.[file:1]

3. **Exploratory Data Analysis (EDA)**  
   - Computes dataset shape and risk distribution (~34% Medium, ~33% Low, ~33% High).[file:1]  
   - Visualizes distributions and correlations, and saves an EDA figure as `edaanalysis.png`.[file:1]

4. **Prepare Data for Training**  
   - Splits data into train/test sets.[file:1]  
   - Defines feature matrix and target vector.  
   - Applies scaling / preprocessing and prepares pipelines for different models.[file:1]

5. **Train Multiple AI Models**  
   Trains the following classifiers on the synthetic dataset:[file:1]

   - Logistic Regression  
   - Random Forest Classifier  
   - Gradient Boosting Classifier  

6. **Evaluate Models**  
   - Computes accuracy, classification reports, and confusion matrices.[file:1]  
   - In the reported run, **Logistic Regression** achieves about **92% accuracy**, outperforming Random Forest (~79%) and Gradient Boosting (~80%).[file:1]  
   - Saves confusion matrix plots in `modelevaluation.png`.[file:1]

7. **Feature Importance Analysis**  
   - Extracts importance scores (from tree models or coefficients) and plots them.[file:1]  
   - Top contributing features include:[file:1]  
     - `delivery_delay_days`  
     - `lead_time_days`  
     - `num_suppliers`  
     - `demand_variability`  
     - `inventory_level`  
   - Saves the chart as `featureimportance.png`.[file:1]

8. **Save the Trained Model**  
   - Stores the best-performing model and feature list using `joblib.dump` to `bestsupplychainmodel.pkl`.[file:1]  
   - Shows how to reload it with `joblib.load("bestsupplychainmodel.pkl")`.[file:1]

9. **Agentic AI Supply Chain Risk Monitor**  
   Defines a class `SupplyChainRiskAgent` that implements a Sense–Reason–Act loop:[file:1]

   - **Sense:**  
     - Accepts a state as a Python dict, converts it into a 1-row DataFrame with the same feature columns used in training.[file:1]
   - **Reason:**  
     - Uses the trained model’s `.predict` and `.predict_proba` to compute risk level and class probabilities.[file:1]
   - **Act:**  
     - Applies a rule-based action playbook based on key risk signals such as high delay, low reliability, low inventory, high geopolitical risk, or high demand variability.[file:1]  
     - Generates recommended mitigation actions and an alert flag for high/medium risks.[file:1]

   The agent also:[file:1]

   - Logs every decision with timestamp, input state, predicted risk, confidence, actions, and alert status.  
   - Provides `print_decision` to nicely display results.  
   - Plots decision history (`plot_agent_history`), saving to `agenthistory.png`.[file:1]

---

## 3. Agent Scenarios

The notebook includes three illustrative scenarios used to test the agent:[file:1]

- **Scenario A – Crisis (High Risk)**  
  - Low supplier reliability, high delays, high geopolitical and weather risk, and high defect rates.[file:1]  
  - Predicted: **HIGH risk** with near 100% confidence.[file:1]  
  - Actions: emergency inventory, expedite shipments (e.g., air), escalate to management, onboard backup suppliers, performance review.[file:1]

- **Scenario B – Moderate Risk**  
  - Mixed metrics with moderate delay and variability.[file:1]  
  - Predicted: **MEDIUM risk** with significant confidence.[file:1]  
  - Actions: targeted buffer, supplier collaboration, closer demand monitoring.[file:1]

- **Scenario C – Stable (Low Risk)**  
  - High reliability, low delays, low geopolitical and weather risk.[file:1]  
  - Predicted: **LOW risk** with no alert.[file:1]  
  - Actions: routine monitoring, KPI review, maintaining playbooks.[file:1]

The agent history plot shows risk level over scenarios and highlights decisions that triggered alerts.[file:1]

---

## 4. Generated Artifacts

Running the notebook end-to-end will produce these files:[file:1]

- `edaanalysis.png` – EDA visualizations (distributions, correlations, class balance).[file:1]  
- `modelevaluation.png` – Confusion matrices for all trained models.[file:1]  
- `featureimportance.png` – Feature importance chart for the selected best model.[file:1]  
- `agenthistory.png` – Timeline of agent decisions and risk levels.[file:1]  
- `bestsupplychainmodel.pkl` – Serialized best model plus feature list/metadata.[file:1]

---

## 5. How to Run

### 5.1 Requirements

Install Python 3 and the following packages (example using pip):[file:1]

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

You also need a Jupyter environment (Jupyter Notebook, JupyterLab, VS Code, etc.).[file:1]

### 5.2 Steps

1. Open `Supply_Chain_Risk_AI.ipynb` in Jupyter.[file:1]  
2. Run all cells from top to bottom:[file:1]  
   - Data generation  
   - Exploratory Data Analysis  
   - Model training and evaluation  
   - Model saving  
   - Agent class definition and scenario runs  
3. Review generated plots (`edaanalysis.png`, `modelevaluation.png`, `featureimportance.png`, `agenthistory.png`).[file:1]  
4. Use `bestsupplychainmodel.pkl` or the `SupplyChainRiskAgent` class to score new scenarios.[file:1]

### 5.3 Example: Using the Saved Model

```python
import joblib
import pandas as pd

# Load model bundle
bundle = joblib.load("bestsupplychainmodel.pkl")
model = bundle["model"]
feature_cols = bundle["features"]

# Example scenario
state = {
    "supplier_reliability": 0.8,
    "delivery_delay_days": 2.0,
    "demand_variability": 0.4,
    "inventory_level": 0.6,
    "geopolitical_risk": 0.2,
    "transportation_cost": 50.0,
    "quality_defect_rate": 0.03,
    "lead_time_days": 25,
    "num_suppliers": 5,
    "weather_risk": 0.3,
}

X = pd.DataFrame([state], columns=feature_cols)
risk_level = model.predict(X)
probas = model.predict_proba(X)

print("Predicted risk level:", risk_level)
print("Probabilities:", probas)
```

To use the full agent, instantiate `SupplyChainRiskAgent` with the loaded model and its feature list as shown in the notebook.[file:1]

---

## 6. Key Results

- Best classifier: **Logistic Regression**, with about **92% accuracy** on the test set.[file:1]  
- The trained models and agent behavior align with supply-chain intuition: higher delays, lower reliability, fewer suppliers, and higher external risks drive higher predicted risk levels and more aggressive actions.[file:1]
