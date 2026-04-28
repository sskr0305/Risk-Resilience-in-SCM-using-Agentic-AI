# ============================================================
#   SUPPLY CHAIN RISK RESILIENCE USING AGENTIC AI
#   Course: Industrial Management (IM41081)
#   Project: AI-Powered Supply Chain Risk Management System
# ============================================================

# ─────────────────────────────────────────────────────────────
# STEP 1: IMPORT LIBRARIES
# ─────────────────────────────────────────────────────────────
# We import all necessary libraries for data handling,
# machine learning, visualization, and AI agent logic.

import numpy as np                          # Numerical computations
import pandas as pd                         # Data manipulation
import matplotlib.pyplot as plt             # Plotting graphs
import seaborn as sns                       # Beautiful statistical plots
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline

import joblib                               # Save/load trained models
import json                                 # For agent communication
import random                               # Random data generation
from datetime import datetime, timedelta    # Date handling


# ─────────────────────────────────────────────────────────────
# STEP 2: GENERATE SYNTHETIC SUPPLY CHAIN DATASET
# ─────────────────────────────────────────────────────────────
# PURPOSE: In a real project you'd use real company data.
# Here, we SIMULATE a realistic supply chain dataset with
# features like delivery delays, demand variability, etc.
# This teaches the model what "risky" scenarios look like.

def generate_supply_chain_data(n_samples=2000, random_state=42):
    """
    Generates a synthetic supply chain dataset.

    Features (inputs to the model):
    - supplier_reliability   : Score 0–1 (how dependable the supplier is)
    - delivery_delay_days    : Days delayed beyond expected delivery
    - demand_variability     : Fluctuation in customer demand (0–1)
    - inventory_level        : Stock available as % of required
    - geopolitical_risk      : Risk score of supplier's country (0–1)
    - transportation_cost    : Cost index (higher = more expensive)
    - quality_defect_rate    : % of defective items (0–1)
    - lead_time_days         : How many days from order to delivery
    - num_suppliers          : Number of alternative suppliers
    - weather_risk           : Environmental disruption risk (0–1)

    Target (output — what we want to predict):
    - risk_level: 0 = Low, 1 = Medium, 2 = High
    """
    np.random.seed(random_state)
    n = n_samples

    data = {
        'supplier_reliability':   np.random.beta(5, 2, n),           # Mostly high reliability
        'delivery_delay_days':    np.random.exponential(3, n),        # Right-skewed delays
        'demand_variability':     np.random.uniform(0, 1, n),
        'inventory_level':        np.random.normal(0.6, 0.2, n).clip(0, 1),
        'geopolitical_risk':      np.random.beta(2, 5, n),            # Mostly low risk
        'transportation_cost':    np.random.normal(50, 15, n).clip(10, 100),
        'quality_defect_rate':    np.random.beta(1, 10, n),           # Low defects usually
        'lead_time_days':         np.random.randint(5, 60, n),
        'num_suppliers':          np.random.randint(1, 10, n),
        'weather_risk':           np.random.uniform(0, 1, n),
    }

    df = pd.DataFrame(data)

    # ── RISK SCORING LOGIC ──────────────────────────────────
    # We compute a composite risk score based on domain knowledge.
    # Each factor contributes positively or negatively to risk.
    risk_score = (
          (1 - df['supplier_reliability'])   * 2.5   # Unreliable → High risk
        + df['delivery_delay_days']          * 0.3   # Delays → Risk
        + df['demand_variability']           * 1.5   # Volatile demand → Risk
        + (1 - df['inventory_level'])        * 2.0   # Low stock → Risk
        + df['geopolitical_risk']            * 2.0   # Unstable region → Risk
        + df['transportation_cost']          * 0.02  # Expensive → Moderate risk
        + df['quality_defect_rate']          * 3.0   # Defects → Risk
        + df['lead_time_days']               * 0.05  # Long lead time → Risk
        - df['num_suppliers']                * 0.3   # More suppliers → Lower risk
        + df['weather_risk']                 * 1.0   # Bad weather → Risk
        + np.random.normal(0, 0.3, n)                # Random noise
    )

    # Convert score into 3 categories: Low / Medium / High
    low_thresh    = np.percentile(risk_score, 33)
    high_thresh   = np.percentile(risk_score, 67)

    df['risk_level'] = pd.cut(
        risk_score,
        bins=[-np.inf, low_thresh, high_thresh, np.inf],
        labels=[0, 1, 2]   # 0=Low, 1=Medium, 2=High
    ).astype(int)

    return df


# ─────────────────────────────────────────────────────────────
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────
# PURPOSE: Understand the data before training — distributions,
# correlations, class balance. This step is crucial in any
# real machine learning project.

def perform_eda(df):
    """Visualize the dataset to understand patterns."""
    print("\n" + "="*60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("="*60)
    print(f"\n Dataset Shape: {df.shape}")
    print(f"\n Risk Level Distribution:")
    labels = {0: 'Low', 1: 'Medium', 2: 'High'}
    for k, v in df['risk_level'].value_counts().items():
        print(f"   {labels[k]} Risk: {v} samples ({v/len(df)*100:.1f}%)")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Supply Chain Risk – Exploratory Data Analysis',
                 fontsize=16, fontweight='bold', y=1.01)

    # Plot 1: Risk level distribution
    ax = axes[0, 0]
    counts = df['risk_level'].value_counts().sort_index()
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    bars = ax.bar(['Low', 'Medium', 'High'], counts.values, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_title('Risk Level Distribution', fontweight='bold')
    ax.set_ylabel('Count')
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(val), ha='center', fontweight='bold')

    # Plot 2: Correlation heatmap
    ax = axes[0, 1]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=ax, mask=mask, cmap='RdYlGn_r', center=0,
                annot=False, linewidths=0.5, square=False)
    ax.set_title('Feature Correlation Heatmap', fontweight='bold')

    # Plot 3: Supplier reliability vs risk
    ax = axes[1, 0]
    for level, color, label in zip([0, 1, 2], colors, ['Low', 'Medium', 'High']):
        subset = df[df['risk_level'] == level]
        ax.hist(subset['supplier_reliability'], alpha=0.6, color=color,
                label=f'{label} Risk', bins=20)
    ax.set_xlabel('Supplier Reliability')
    ax.set_ylabel('Frequency')
    ax.set_title('Supplier Reliability by Risk Level', fontweight='bold')
    ax.legend()

    # Plot 4: Delivery delay vs inventory level scatter
    ax = axes[1, 1]
    scatter_colors = [colors[r] for r in df['risk_level']]
    ax.scatter(df['delivery_delay_days'], df['inventory_level'],
               c=scatter_colors, alpha=0.4, s=10)
    ax.set_xlabel('Delivery Delay (days)')
    ax.set_ylabel('Inventory Level')
    ax.set_title('Delay vs Inventory (colored by Risk)', fontweight='bold')
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l)
                       for c, l in zip(colors, ['Low', 'Medium', 'High'])]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig('eda_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n [EDA chart saved as: eda_analysis.png]")


# ─────────────────────────────────────────────────────────────
# STEP 4: PREPARE DATA FOR TRAINING
# ─────────────────────────────────────────────────────────────
# PURPOSE: Split data into features (X) and target (y),
# then split again into train/test sets so we can evaluate
# how well the model performs on unseen data.

def prepare_data(df):
    """Split dataset into training and testing sets."""
    feature_cols = [
        'supplier_reliability', 'delivery_delay_days', 'demand_variability',
        'inventory_level', 'geopolitical_risk', 'transportation_cost',
        'quality_defect_rate', 'lead_time_days', 'num_suppliers', 'weather_risk'
    ]
    X = df[feature_cols]
    y = df['risk_level']

    # 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n Training samples : {len(X_train)}")
    print(f" Test samples     : {len(X_test)}")
    return X_train, X_test, y_train, y_test, feature_cols

# ================= EXTRA VISUALS - STEP 4 =================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Feature Mean Comparison (Train vs Test)
train_means = X_train.mean()
test_means = X_test.mean()

mean_df = pd.DataFrame({
    'Train Mean': train_means,
    'Test Mean': test_means
})

mean_df.plot(kind='bar', figsize=(12,6))
plt.title("Feature Mean Comparison (Train vs Test)")
plt.xticks(rotation=45)
plt.ylabel("Mean Value")
plt.show()


# 2. Feature Variance Comparison
train_var = X_train.var()
test_var = X_test.var()

var_df = pd.DataFrame({
    'Train Variance': train_var,
    'Test Variance': test_var
})

var_df.plot(kind='bar', figsize=(12,6))
plt.title("Feature Variance Comparison (Train vs Test)")
plt.xticks(rotation=45)
plt.ylabel("Variance")
plt.show()


# 3. Target Distribution (Normalized %)
train_dist = y_train.value_counts(normalize=True)
test_dist = y_test.value_counts(normalize=True)

dist_df = pd.DataFrame({
    'Train %': train_dist,
    'Test %': test_dist
}).fillna(0)

dist_df.plot(kind='bar', figsize=(6,4))
plt.title("Target Distribution (%) - Train vs Test")
plt.ylabel("Proportion")
plt.show()


# 4. KS-style Distribution Difference (Quick sanity)
for col in X_train.columns:
    plt.figure(figsize=(5,3))
    sns.histplot(X_train[col], color='blue', label='Train', kde=True, stat="density")
    sns.histplot(X_test[col], color='orange', label='Test', kde=True, stat="density")
    plt.title(f"Distribution Check: {col}")
    plt.legend()
    plt.show()


# ─────────────────────────────────────────────────────────────
# STEP 5: TRAIN MULTIPLE AI MODELS
# ─────────────────────────────────────────────────────────────
# PURPOSE: We train 3 different models and compare them.
# This is best practice — never rely on a single model!
#
# Models used:
# 1. Random Forest     – Ensemble of decision trees (robust)
# 2. Gradient Boosting – Sequential tree boosting (accurate)
# 3. Logistic Regression – Baseline linear classifier

def train_models(X_train, y_train):
    """Train multiple classifiers using sklearn Pipelines."""
    print("\n" + "="*60)
    print("  TRAINING AI MODELS")
    print("="*60)

    models = {
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=150,     # 150 decision trees
                max_depth=10,         # Limit tree depth to prevent overfitting
                min_samples_split=5,
                random_state=42,
                n_jobs=-1             # Use all CPU cores
            ))
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=120,     # 120 boosting stages
                learning_rate=0.1,    # How much each tree contributes
                max_depth=5,
                random_state=42
            ))
        ]),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=1000,        # Allow enough iterations to converge
                random_state=42       # multi_class removed (auto in sklearn 1.7+)
            ))
        ])
    }

    trained = {}
    for name, pipeline in models.items():
        print(f"\n Training {name}...", end=' ')
        pipeline.fit(X_train, y_train)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        print(f"Done! CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        trained[name] = pipeline

    return trained


# ─────────────────────────────────────────────────────────────
# STEP 6: EVALUATE MODELS
# ─────────────────────────────────────────────────────────────
# PURPOSE: Check how well each model performs on TEST data
# (data the model has NEVER seen before). This gives an honest
# measure of real-world performance.

def evaluate_models(trained_models, X_test, y_test):
    """Evaluate each model and visualize results."""
    print("\n" + "="*60)
    print("  MODEL EVALUATION")
    print("="*60)

    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Model Evaluation – Confusion Matrices', fontsize=14, fontweight='bold')

    for ax, (name, model) in zip(axes, trained_models.items()):
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {'accuracy': acc, 'model': model}

        # Confusion matrix: rows=actual, cols=predicted
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Low', 'Med', 'High'],
                    yticklabels=['Low', 'Med', 'High'])
        ax.set_title(f'{name}\nAccuracy: {acc:.3f}', fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        print(f"\n {name}:")
        print(f"   Accuracy : {acc:.4f} ({acc*100:.1f}%)")
        print(classification_report(y_test, y_pred,
              target_names=['Low Risk', 'Medium Risk', 'High Risk'],
              zero_division=0))

    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(" [Confusion matrix chart saved as: model_evaluation.png]")

    # Choose the best model by accuracy
    best_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model = results[best_name]['model']
    print(f"\n Best Model: {best_name} (Accuracy: {results[best_name]['accuracy']:.4f})")
    return best_model, best_name, results

# ================= EXTRA VISUALS - STEP 6 =================

from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

# 1. Confusion Matrix (Clean Heatmap)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# 2. Prediction Confidence Distribution (if probabilities exist)
if hasattr(model, "predict_proba"):
    y_probs = model.predict_proba(X_test)

    plt.figure(figsize=(6,4))
    for i in range(y_probs.shape[1]):
        sns.kdeplot(y_probs[:, i], label=f'Class {i}', fill=True)

    plt.title("Prediction Probability Distribution")
    plt.xlabel("Probability")
    plt.legend()
    plt.show()


# 3. ROC Curve (Binary OR One-vs-Rest)
if hasattr(model, "predict_proba"):
    try:
        # For binary classification
        fpr, tpr, _ = roc_curve(y_test, y_probs[:,1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0,1], [0,1], linestyle='--')
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.show()

    except:
        print("ROC skipped (likely multiclass).")


# 4. Feature Importance (if model supports it)
if hasattr(model, "feature_importances_"):
    importance = model.feature_importances_

    feat_imp = pd.Series(importance, index=feature_cols).sort_values()

    plt.figure(figsize=(8,5))
    feat_imp.plot(kind='barh')
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.show()


# 5. Prediction vs Actual Count Comparison
pred_counts = pd.Series(y_pred).value_counts()
actual_counts = y_test.value_counts()

compare_df = pd.DataFrame({
    'Actual': actual_counts,
    'Predicted': pred_counts
}).fillna(0)

compare_df.plot(kind='bar', figsize=(6,4))
plt.title("Actual vs Predicted Counts")
plt.ylabel("Count")
plt.show()


# ─────────────────────────────────────────────────────────────
# STEP 7: FEATURE IMPORTANCE ANALYSIS
# ─────────────────────────────────────────────────────────────
# PURPOSE: Understand WHICH features matter most for predicting
# risk. This is critical for supply chain managers — it tells
# them what to monitor closely.

def plot_feature_importance(best_model, best_name, feature_cols):
    """Plot which features are most important for risk prediction."""
    clf = best_model.named_steps['clf']

    if hasattr(clf, 'feature_importances_'):
        # Tree-based models (Random Forest, Gradient Boosting)
        importances = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        # Logistic Regression — use average absolute coefficient across classes
        importances = np.mean(np.abs(clf.coef_), axis=0)
    else:
        print("Feature importance not available for this model.")
        return

    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(feature_cols)))
    plt.barh([feature_cols[i] for i in indices[::-1]],
             importances[indices[::-1]],
             color=colors[::-1], edgecolor='white')
    plt.xlabel('Importance Score')
    plt.title(f'Feature Importance – {best_name}', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n Top Risk Factors (by importance):")
    for i in indices[:5]:
        print(f"   {feature_cols[i]:<30} {importances[i]:.4f}")
    print(" [Feature importance chart saved as: feature_importance.png]")


# ─────────────────────────────────────────────────────────────
# STEP 8: AGENTIC AI SYSTEM
# ─────────────────────────────────────────────────────────────
# PURPOSE: This is the "Agentic AI" part — an AI agent that:
#   1. SENSES the environment (reads supply chain data)
#   2. DECIDES what to do (predicts risk level)
#   3. ACTS (generates specific recommendations)
#   4. MONITORS (tracks decisions over time)
#
# Agentic AI = AI that takes autonomous actions, not just answers.

class SupplyChainRiskAgent:
    """
    An autonomous AI agent for supply chain risk management.

    The agent follows a Sense → Reason → Act loop:
    - SENSE : Receive supply chain metrics
    - REASON: Predict risk level using trained ML model
    - ACT   : Generate corrective recommendations
    """

    RISK_LABELS = {0: '[LOW]', 1: '[MEDIUM]', 2: '[HIGH]'}

    # Playbook: What actions to take for each risk scenario
    ACTION_PLAYBOOK = {
        'high_delay': [
            "[ACTION] Activate emergency inventory buffer (safety stock)",
            "[ACTION] Switch to air freight for critical components",
            "[ACTION] Escalate to senior procurement team immediately",
        ],
        'low_supplier_reliability': [
            "[ACTION] Dual-source: activate backup supplier contracts",
            "[ACTION] Trigger supplier performance review meeting",
            "[ACTION] Increase local/regional sourcing to reduce dependency",
        ],
        'low_inventory': [
            "[ACTION] Issue emergency purchase order",
            "[ACTION] Revise reorder point (ROP) and safety stock formula",
            "[ACTION] Implement demand rationing for non-critical clients",
        ],
        'high_geopolitical_risk': [
            "[ACTION] Diversify supply base across multiple geographies",
            "[ACTION] Monitor geopolitical news feeds daily",
            "[ACTION] Pre-purchase and stockpile critical materials",
        ],
        'high_demand_variability': [
            "[ACTION] Apply demand forecasting model (ARIMA/ML)",
            "[ACTION] Share POS data with suppliers (VMI program)",
            "[ACTION] Move to flexible manufacturing/postponement strategy",
        ],
        'default': [
            "[OK] Continue standard monitoring protocols",
            "[OK] Review KPIs in weekly supply chain meeting",
            "[OK] Update supplier scorecards",
        ]
    }

    def __init__(self, model, feature_cols):
        self.model = model
        self.feature_cols = feature_cols
        self.decision_log = []          # Memory: stores all past decisions
        self.alert_threshold = 1        # Trigger alert for Medium+ risk

    def sense(self, supply_chain_state: dict) -> pd.DataFrame:
        """Convert raw supply chain data into model-ready format."""
        return pd.DataFrame([supply_chain_state])[self.feature_cols]

    def reason(self, X: pd.DataFrame) -> tuple:
        """Predict risk level and get confidence probabilities."""
        risk_level = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        return int(risk_level), probabilities

    def act(self, state: dict, risk_level: int, probabilities: np.ndarray) -> dict:
        """Generate targeted recommendations based on risk factors."""
        recommendations = []

        # Rule-based condition checks (domain knowledge)
        if state.get('delivery_delay_days', 0) > 7:
            recommendations.extend(self.ACTION_PLAYBOOK['high_delay'])
        if state.get('supplier_reliability', 1) < 0.5:
            recommendations.extend(self.ACTION_PLAYBOOK['low_supplier_reliability'])
        if state.get('inventory_level', 1) < 0.3:
            recommendations.extend(self.ACTION_PLAYBOOK['low_inventory'])
        if state.get('geopolitical_risk', 0) > 0.6:
            recommendations.extend(self.ACTION_PLAYBOOK['high_geopolitical_risk'])
        if state.get('demand_variability', 0) > 0.7:
            recommendations.extend(self.ACTION_PLAYBOOK['high_demand_variability'])

        if not recommendations:
            recommendations = self.ACTION_PLAYBOOK['default']

        # Remove duplicate recommendations
        recommendations = list(dict.fromkeys(recommendations))

        decision = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'state': state,
            'predicted_risk': self.RISK_LABELS[risk_level],
            'risk_level_code': risk_level,
            'confidence': {
                'Low Risk':    f"{probabilities[0]*100:.1f}%",
                'Medium Risk': f"{probabilities[1]*100:.1f}%",
                'High Risk':   f"{probabilities[2]*100:.1f}%",
            },
            'recommendations': recommendations,
            'alert_triggered': risk_level >= self.alert_threshold
        }

        self.decision_log.append(decision)   # Store in agent memory
        return decision

    def run(self, supply_chain_state: dict) -> dict:
        """
        Full Sense → Reason → Act cycle.
        Call this with any new supply chain data point.
        """
        X = self.sense(supply_chain_state)
        risk_level, probabilities = self.reason(X)
        decision = self.act(supply_chain_state, risk_level, probabilities)
        return decision

    def print_decision(self, decision: dict):
        """Pretty-print the agent's decision to the console."""
        print("\n" + "-"*55)
        print(f"  AGENT DECISION REPORT | {decision['timestamp']}")
        print("-"*55)
        print(f"  Predicted Risk : {decision['predicted_risk']}")
        print(f"  Confidence     :")
        for label, conf in decision['confidence'].items():
            print(f"    |-- {label:<14}: {conf}")
        print(f"  Alert Triggered: {'[!] YES' if decision['alert_triggered'] else 'NO'}")
        print(f"\n  Recommended Actions:")
        for i, rec in enumerate(decision['recommendations'][:5], 1):
            print(f"  {i}. {rec}")
        print("-"*55)

    def plot_agent_history(self):
        """Visualize how the agent's risk assessments evolve over time."""
        if len(self.decision_log) < 2:
            print("Need at least 2 decisions to plot history.")
            return

        times = [d['timestamp'] for d in self.decision_log]
        risks = [d['risk_level_code'] for d in self.decision_log]
        alerts = [d['alert_triggered'] for d in self.decision_log]

        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ['#2ecc71' if r == 0 else '#f39c12' if r == 1 else '#e74c3c'
                  for r in risks]
        ax.plot(range(len(risks)), risks, 'k-', linewidth=1.5, alpha=0.4, zorder=1)
        ax.scatter(range(len(risks)), risks, c=colors, s=100, zorder=2, edgecolors='white')

        # Mark alert points
        for i, (r, alert) in enumerate(zip(risks, alerts)):
            if alert:
                ax.annotate('⚠', (i, r), textcoords='offset points',
                            xytext=(0, 12), ha='center', fontsize=12)

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Low', 'Medium', 'High'])
        ax.set_xlabel('Decision Number')
        ax.set_title('Agent Risk Assessment History', fontweight='bold', fontsize=13)
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('agent_history.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(" [Agent history chart saved as: agent_history.png]")


# ─────────────────────────────────────────────────────────────
# STEP 9: SAVE THE TRAINED MODEL
# ─────────────────────────────────────────────────────────────
# PURPOSE: Save the best model to disk so it can be reused
# without retraining every time (like saving a Word document).

def save_model(model, model_name, feature_cols):
    """Save trained model and feature metadata to disk."""
    filename = 'best_supply_chain_model.pkl'
    joblib.dump({'model': model, 'features': feature_cols}, filename)
    print(f"\n Model saved: {filename}")
    print(f" To reload: data = joblib.load('{filename}')")
    return filename


# ─────────────────────────────────────────────────────────────
# STEP 10: MAIN EXECUTION PIPELINE
# ─────────────────────────────────────────────────────────────
# PURPOSE: Orchestrates all steps from data generation to
# agent demonstration. This is the entry point of the program.

def main():
    print("\n" + "="*60)
    print("  SUPPLY CHAIN RISK RESILIENCE — AGENTIC AI SYSTEM")
    print("  Course: Industrial Management (IM41081)")
    print("="*60)

    # ── Step 2: Generate data ─────────────────────────────
    print("\n [1/7] Generating synthetic supply chain data...")
    df = generate_supply_chain_data(n_samples=2000)
    print(f" Generated {len(df)} supply chain records.")
    print(df.head(3).to_string())

    # ── Step 3: EDA ───────────────────────────────────────
    print("\n [2/7] Performing Exploratory Data Analysis...")
    perform_eda(df)

    # ── Step 4: Prepare data ──────────────────────────────
    print("\n [3/7] Preparing training and test data...")
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(df)

    # ── Step 5: Train models ──────────────────────────────
    print("\n [4/7] Training AI Models...")
    trained_models = train_models(X_train, y_train)

    # ── Step 6: Evaluate models ───────────────────────────
    print("\n [5/7] Evaluating model performance...")
    best_model, best_name, all_results = evaluate_models(trained_models, X_test, y_test)

    # ── Step 7: Feature importance ────────────────────────
    print("\n [6/7] Analyzing feature importance...")
    plot_feature_importance(best_model, best_name, feature_cols)

    # ── Step 8: Save model ────────────────────────────────
    save_model(best_model, best_name, feature_cols)

    # ── Step 9: Run Agentic AI Demo ───────────────────────
    print("\n [7/7] Launching Agentic AI Supply Chain Monitor...")
    print("="*60)

    agent = SupplyChainRiskAgent(best_model, feature_cols)

    # Simulate 3 real-world scenarios
    scenarios = [
        {   # Scenario A: High risk situation
            'name': 'SCENARIO A - Crisis (High Risk Expected)',
            'data': {
                'supplier_reliability': 0.25,
                'delivery_delay_days': 15.0,
                'demand_variability': 0.85,
                'inventory_level': 0.10,
                'geopolitical_risk': 0.80,
                'transportation_cost': 85.0,
                'quality_defect_rate': 0.12,
                'lead_time_days': 45,
                'num_suppliers': 1,
                'weather_risk': 0.75,
            }
        },
        {   # Scenario B: Medium risk
            'name': 'SCENARIO B - Moderate (Medium Risk Expected)',
            'data': {
                'supplier_reliability': 0.62,
                'delivery_delay_days': 4.0,
                'demand_variability': 0.50,
                'inventory_level': 0.45,
                'geopolitical_risk': 0.35,
                'transportation_cost': 55.0,
                'quality_defect_rate': 0.04,
                'lead_time_days': 20,
                'num_suppliers': 4,
                'weather_risk': 0.40,
            }
        },
        {   # Scenario C: Low risk
            'name': 'SCENARIO C - Stable (Low Risk Expected)',
            'data': {
                'supplier_reliability': 0.92,
                'delivery_delay_days': 0.5,
                'demand_variability': 0.15,
                'inventory_level': 0.88,
                'geopolitical_risk': 0.10,
                'transportation_cost': 35.0,
                'quality_defect_rate': 0.01,
                'lead_time_days': 8,
                'num_suppliers': 7,
                'weather_risk': 0.12,
            }
        },
    ]

    for scenario in scenarios:
        print(f"\n\n{'='*55}")
        print(f"  {scenario['name']}")
        print('='*55)
        decision = agent.run(scenario['data'])
        agent.print_decision(decision)

    # Plot agent decision history
    agent.plot_agent_history()

    print("\n" + "="*60)
    print("  PROJECT COMPLETE!")
    print("  Files generated:")
    print("   |-- eda_analysis.png       (EDA charts)")
    print("   |-- model_evaluation.png   (Confusion matrices)")
    print("   |-- feature_importance.png (Risk drivers)")
    print("   |-- agent_history.png      (Agent decisions)")
    print("   |-- best_supply_chain_model.pkl  (Saved model)")
    print("="*60)


if __name__ == '__main__':
    main()
