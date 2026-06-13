"""
RetireWise+ ML Models
=====================
1. Risk Profiler        — RandomForest Classifier
2. SIP Return Predictor — Gradient Boosting Regressor
3. Portfolio Recommender — Scoring-based Recommender with KMeans clustering

Author: RetireWise+ (ARN-330249)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import joblib
import json
import os

np.random.seed(42)
N = 2000

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATASET
# ─────────────────────────────────────────────

def generate_dataset(n=N):
    age           = np.random.randint(22, 60, n)
    income        = np.random.randint(30000, 300000, n)
    expenses      = income * np.random.uniform(0.3, 0.85, n)
    savings_rate  = (income - expenses) / income
    dependents    = np.random.randint(0, 5, n)
    horizon       = np.random.randint(1, 35, n)
    existing_inv  = np.random.randint(0, 5000000, n)
    crash_behavior= np.random.choice([0, 1, 2], n, p=[0.25, 0.45, 0.30])  # 0=panic,1=wait,2=invest
    exp_level     = np.random.choice([0, 1, 2], n, p=[0.35, 0.40, 0.25])  # 0=beginner,1=some,2=exp
    elss_pref     = np.random.choice([0, 1], n)

    # Risk label logic (realistic)
    risk_score = (
        (horizon > 10).astype(int) * 2 +
        (savings_rate > 0.3).astype(int) * 1 +
        crash_behavior * 1.5 +
        exp_level * 1 +
        (age < 35).astype(int) * 1 -
        (dependents > 2).astype(int) * 1
    )
    risk_label = np.where(risk_score < 3, 'conservative',
                 np.where(risk_score < 6, 'moderate', 'aggressive'))

    # SIP amount
    sip = (income - expenses) * np.random.uniform(0.3, 0.7, n)
    sip = np.clip(sip, 1000, 200000).astype(int)

    # SIP return (CAGR %) — realistic based on risk
    base_return = np.where(risk_label == 'conservative', 7.5,
                  np.where(risk_label == 'moderate', 11.5, 14.5))
    noise = np.random.normal(0, 1.5, n)
    horizon_bonus = np.where(horizon > 15, 0.5, 0)
    sip_return = base_return + noise + horizon_bonus
    sip_return = np.clip(sip_return, 4.0, 22.0)

    return pd.DataFrame({
        'age': age, 'income': income, 'expenses': expenses.astype(int),
        'savings_rate': savings_rate.round(3), 'dependents': dependents,
        'horizon': horizon, 'existing_inv': existing_inv,
        'crash_behavior': crash_behavior, 'exp_level': exp_level,
        'elss_pref': elss_pref, 'sip': sip,
        'sip_return': sip_return.round(2), 'risk_label': risk_label
    })

df = generate_dataset()
df.to_csv('data/synthetic_clients.csv', index=False)
print(f"[DATA] Generated {len(df)} synthetic client records")

FEATURES = ['age', 'income', 'expenses', 'savings_rate', 'dependents',
            'horizon', 'existing_inv', 'crash_behavior', 'exp_level', 'elss_pref']

X = df[FEATURES]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────────
# 2. MODEL 1: RISK PROFILER (Classification)
# ─────────────────────────────────────────────

le = LabelEncoder()
y_risk = le.fit_transform(df['risk_label'])

X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_risk, test_size=0.2, random_state=42)

risk_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_split=5,
    class_weight='balanced', random_state=42
)
risk_model.fit(X_tr, y_tr)
y_pred_risk = risk_model.predict(X_te)
print("\n[MODEL 1] Risk Profiler — Random Forest Classifier")
print(classification_report(y_te, y_pred_risk, target_names=le.classes_))

# ─────────────────────────────────────────────
# 3. MODEL 2: SIP RETURN PREDICTOR (Regression)
# ─────────────────────────────────────────────

FEAT_REG = FEATURES + ['sip']
X_reg = df[FEAT_REG]
X_reg_scaled = scaler.fit_transform(X_reg[FEATURES])
X_reg_full = np.hstack([X_reg_scaled, df[['sip']].values / 100000])

y_ret = df['sip_return']
X_rtr, X_rte, y_rtr, y_rte = train_test_split(X_reg_full, y_ret, test_size=0.2, random_state=42)

ret_model = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=5,
    subsample=0.8, random_state=42
)
ret_model.fit(X_rtr, y_rtr)
y_pred_ret = ret_model.predict(X_rte)
mae = mean_absolute_error(y_rte, y_pred_ret)
r2  = r2_score(y_rte, y_pred_ret)
print(f"\n[MODEL 2] SIP Return Predictor — Gradient Boosting Regressor")
print(f"  MAE : {mae:.3f}%  |  R²: {r2:.3f}")

# ─────────────────────────────────────────────
# 4. MODEL 3: PORTFOLIO RECOMMENDER (KMeans + Scoring)
# ─────────────────────────────────────────────

kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

FUND_DB = [
    {"name": "Mirae Asset Large Cap Fund", "category": "Large Cap", "amc": "Mirae Asset",
     "min_horizon": 5, "risk": "moderate", "expense_ratio": 0.54, "cagr_3yr": 13.2,
     "downside_protection": 8, "consistency": 9},
    {"name": "Parag Parikh Flexi Cap Fund", "category": "Flexi Cap", "amc": "PPFAS",
     "min_horizon": 5, "risk": "moderate", "expense_ratio": 0.63, "cagr_3yr": 18.4,
     "downside_protection": 9, "consistency": 10},
    {"name": "HDFC Small Cap Fund", "category": "Small Cap", "amc": "HDFC",
     "min_horizon": 7, "risk": "aggressive", "expense_ratio": 0.67, "cagr_3yr": 22.1,
     "downside_protection": 5, "consistency": 7},
    {"name": "Axis Midcap Fund", "category": "Mid Cap", "amc": "Axis",
     "min_horizon": 7, "risk": "aggressive", "expense_ratio": 0.52, "cagr_3yr": 19.8,
     "downside_protection": 6, "consistency": 8},
    {"name": "SBI Magnum ELSS Tax Saver", "category": "ELSS", "amc": "SBI",
     "min_horizon": 3, "risk": "moderate", "expense_ratio": 0.94, "cagr_3yr": 15.6,
     "downside_protection": 7, "consistency": 8},
    {"name": "Kotak Equity Hybrid Fund", "category": "Hybrid", "amc": "Kotak",
     "min_horizon": 3, "risk": "conservative", "expense_ratio": 0.49, "cagr_3yr": 12.3,
     "downside_protection": 9, "consistency": 9},
    {"name": "HDFC Short Term Debt Fund", "category": "Debt", "amc": "HDFC",
     "min_horizon": 1, "risk": "conservative", "expense_ratio": 0.21, "cagr_3yr": 7.1,
     "downside_protection": 10, "consistency": 10},
    {"name": "Nippon India Small Cap Fund", "category": "Small Cap", "amc": "Nippon",
     "min_horizon": 7, "risk": "aggressive", "expense_ratio": 0.74, "cagr_3yr": 24.3,
     "downside_protection": 4, "consistency": 7},
    {"name": "ICICI Pru Bluechip Fund", "category": "Large Cap", "amc": "ICICI Prudential",
     "min_horizon": 5, "risk": "moderate", "expense_ratio": 0.85, "cagr_3yr": 14.1,
     "downside_protection": 8, "consistency": 9},
    {"name": "UTI Nifty 50 Index Fund", "category": "Index", "amc": "UTI",
     "min_horizon": 5, "risk": "moderate", "expense_ratio": 0.20, "cagr_3yr": 13.5,
     "downside_protection": 7, "consistency": 9},
    {"name": "Aditya Birla SL Corporate Bond", "category": "Debt", "amc": "Aditya Birla",
     "min_horizon": 2, "risk": "conservative", "expense_ratio": 0.36, "cagr_3yr": 7.8,
     "downside_protection": 10, "consistency": 9},
    {"name": "DSP Midcap Fund", "category": "Mid Cap", "amc": "DSP",
     "min_horizon": 7, "risk": "aggressive", "expense_ratio": 0.79, "cagr_3yr": 18.9,
     "downside_protection": 6, "consistency": 8},
]

print(f"\n[MODEL 3] Portfolio Recommender — KMeans Clustering (6 clusters)")
print(f"  Cluster distribution: {dict(zip(*np.unique(df['cluster'], return_counts=True)))}")

# ─────────────────────────────────────────────
# 5. SAVE EVERYTHING
# ─────────────────────────────────────────────

os.makedirs('models', exist_ok=True)
joblib.dump(risk_model, 'models/risk_profiler.pkl')
joblib.dump(ret_model,  'models/sip_predictor.pkl')
joblib.dump(kmeans,     'models/portfolio_clusterer.pkl')
joblib.dump(scaler,     'models/scaler.pkl')
joblib.dump(le,         'models/label_encoder.pkl')

with open('models/fund_db.json', 'w') as f:
    json.dump(FUND_DB, f, indent=2)

meta = {
    "risk_classes": list(le.classes_),
    "features": FEATURES,
    "reg_features": FEAT_REG,
    "n_clusters": 6,
    "model_versions": {
        "risk_profiler": "RandomForest v1.0",
        "sip_predictor": "GradientBoosting v1.0",
        "portfolio_recommender": "KMeans+Scoring v1.0"
    }
}
with open('models/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("\n[DONE] All models saved to models/")
print("       Run `python app.py` to start the web app")
