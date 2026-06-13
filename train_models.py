"""
RetireWise+ ML Engine
=====================
3 Models:
  1. Risk Profiler        — Random Forest Classifier
  2. SIP Return Predictor — Gradient Boosting Regressor
  3. Portfolio Recommender — Weighted Scoring Model (cosine similarity)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import joblib, os, json, warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
N = 3000

# ─── 1. RISK PROFILER DATASET ─────────────────────────────────────────────────
print("▶ Generating risk profiler dataset...")

age        = np.random.randint(22, 65, N)
income     = np.random.randint(25000, 300000, N)
expenses   = income * np.random.uniform(0.3, 0.85, N)
savings_rt = (income - expenses) / income
dependents = np.random.randint(0, 5, N)
horizon    = np.random.randint(1, 35, N)
existing   = np.random.randint(0, 5000000, N)
crash_beh  = np.random.choice([0, 1, 2], N, p=[0.25, 0.45, 0.30])  # 0=panic 1=wait 2=buy
exp_level  = np.random.choice([0, 1, 2], N, p=[0.35, 0.40, 0.25])  # 0=beginner 1=some 2=exp
focus      = np.random.choice([0, 1, 2], N, p=[0.25, 0.45, 0.30])  # 0=safety 1=balanced 2=growth

# Rule-based risk label with noise
risk_score = (
    (age < 35).astype(int) * 2 +
    (savings_rt > 0.4).astype(int) * 2 +
    (horizon > 10).astype(int) * 2 +
    crash_beh * 1.5 +
    exp_level * 1 +
    focus * 1.5
)

def to_risk(s):
    if s < 4:   return 0   # conservative
    elif s < 8: return 1   # moderate
    else:       return 2   # aggressive

risk_labels = np.array([to_risk(s + np.random.uniform(-1, 1)) for s in risk_score])

risk_df = pd.DataFrame({
    'age': age, 'income': income, 'expenses': expenses.astype(int),
    'savings_ratio': savings_rt.round(3), 'dependents': dependents,
    'horizon': horizon, 'existing_investments': existing,
    'crash_behavior': crash_beh, 'experience': exp_level,
    'focus': focus, 'risk_label': risk_labels
})
risk_df.to_csv('data/risk_dataset.csv', index=False)

# Train Risk Profiler
X_r = risk_df.drop('risk_label', axis=1)
y_r = risk_df['risk_label']
X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

risk_model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
risk_model.fit(X_tr, y_tr)
risk_acc = risk_model.score(X_te, y_te)
print(f"  ✓ Risk Profiler accuracy: {risk_acc:.2%}")
print(classification_report(y_te, risk_model.predict(X_te),
      target_names=['Conservative','Moderate','Aggressive'], zero_division=0))

# ─── 2. SIP RETURN PREDICTOR DATASET ─────────────────────────────────────────
print("▶ Generating SIP return predictor dataset...")

sip_amount  = np.random.randint(1000, 100000, N)
sip_horizon = np.random.randint(1, 30, N)
stepup_pct  = np.random.choice([0, 5, 10, 15], N)
risk_cat    = np.random.choice([0, 1, 2], N)  # 0=conservative 1=moderate 2=aggressive
market_cond = np.random.choice([0, 1, 2], N, p=[0.25, 0.50, 0.25])  # 0=bear 1=normal 2=bull

base_rate = np.where(risk_cat == 0, 0.07,
            np.where(risk_cat == 1, 0.11, 0.14))
market_adj = np.where(market_cond == 0, -0.02,
             np.where(market_cond == 2, 0.02, 0.0))
final_rate = base_rate + market_adj + np.random.normal(0, 0.01, N)
final_rate = np.clip(final_rate, 0.04, 0.22)

# FV with step-up SIP
def compute_fv(sip, r, yrs, stepup):
    monthly_r = r / 12
    total = 0
    for y in range(int(yrs)):
        s = sip * ((1 + stepup/100) ** y)
        for m in range(12):
            months_left = (yrs - y) * 12 - m
            total += s * ((1 + monthly_r) ** months_left)
    return total

future_values = np.array([
    compute_fv(sip_amount[i], final_rate[i], sip_horizon[i], stepup_pct[i])
    for i in range(N)
])

sip_df = pd.DataFrame({
    'sip_amount': sip_amount, 'horizon_years': sip_horizon,
    'stepup_percent': stepup_pct, 'risk_category': risk_cat,
    'market_condition': market_cond, 'annual_rate': final_rate.round(4),
    'future_value': future_values.round(0)
})
sip_df.to_csv('data/sip_dataset.csv', index=False)

# Train SIP Return Predictor
feat_cols = ['sip_amount','horizon_years','stepup_percent','risk_category','market_condition']
X_s = sip_df[feat_cols]
y_s = np.log1p(sip_df['future_value'])   # log transform

scaler = StandardScaler()
X_s_sc = scaler.fit_transform(X_s)
Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X_s_sc, y_s, test_size=0.2, random_state=42)

sip_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                       max_depth=5, random_state=42)
sip_model.fit(Xs_tr, ys_tr)
ys_pred = sip_model.predict(Xs_te)
mae = mean_absolute_error(np.expm1(ys_te), np.expm1(ys_pred))
r2  = r2_score(ys_te, ys_pred)
print(f"  ✓ SIP Predictor  R²: {r2:.4f}  |  MAE: ₹{mae:,.0f}")

# ─── 3. PORTFOLIO RECOMMENDER ─────────────────────────────────────────────────
print("▶ Building portfolio recommender...")

FUND_DB = [
    {"name":"Mirae Asset Large Cap Fund","category":"Large Cap","amc":"Mirae Asset",
     "risk":1,"horizon_min":5,"elss":0,"expense":0.54,"rating":5,"cagr_3y":14.2,"cagr_5y":16.8},
    {"name":"HDFC Top 100 Fund","category":"Large Cap","amc":"HDFC",
     "risk":1,"horizon_min":5,"elss":0,"expense":1.62,"rating":4,"cagr_3y":13.8,"cagr_5y":15.9},
    {"name":"Axis Bluechip Fund","category":"Large Cap","amc":"Axis",
     "risk":1,"horizon_min":5,"elss":0,"expense":0.59,"rating":4,"cagr_3y":12.9,"cagr_5y":14.5},
    {"name":"Parag Parikh Flexi Cap Fund","category":"Flexi Cap","amc":"PPFAS",
     "risk":1,"horizon_min":5,"elss":0,"expense":0.63,"rating":5,"cagr_3y":18.4,"cagr_5y":21.2},
    {"name":"HDFC Flexi Cap Fund","category":"Flexi Cap","amc":"HDFC",
     "risk":1,"horizon_min":5,"elss":0,"expense":1.40,"rating":4,"cagr_3y":17.1,"cagr_5y":19.3},
    {"name":"Axis Midcap Fund","category":"Mid Cap","amc":"Axis",
     "risk":2,"horizon_min":7,"elss":0,"expense":0.55,"rating":5,"cagr_3y":19.2,"cagr_5y":22.8},
    {"name":"Kotak Emerging Equity Fund","category":"Mid Cap","amc":"Kotak",
     "risk":2,"horizon_min":7,"elss":0,"expense":0.44,"rating":4,"cagr_3y":18.8,"cagr_5y":21.5},
    {"name":"Nippon India Small Cap Fund","category":"Small Cap","amc":"Nippon",
     "risk":3,"horizon_min":10,"elss":0,"expense":0.72,"rating":5,"cagr_3y":28.4,"cagr_5y":30.1},
    {"name":"HDFC Small Cap Fund","category":"Small Cap","amc":"HDFC",
     "risk":3,"horizon_min":10,"elss":0,"expense":0.63,"rating":4,"cagr_3y":26.9,"cagr_5y":28.7},
    {"name":"SBI Magnum ELSS Fund","category":"ELSS","amc":"SBI",
     "risk":1,"horizon_min":3,"elss":1,"expense":1.62,"rating":4,"cagr_3y":16.2,"cagr_5y":17.9},
    {"name":"Axis Long Term Equity Fund","category":"ELSS","amc":"Axis",
     "risk":1,"horizon_min":3,"elss":1,"expense":0.63,"rating":5,"cagr_3y":15.8,"cagr_5y":18.1},
    {"name":"Mirae Asset Tax Saver Fund","category":"ELSS","amc":"Mirae Asset",
     "risk":1,"horizon_min":3,"elss":1,"expense":0.29,"rating":5,"cagr_3y":17.1,"cagr_5y":19.4},
    {"name":"HDFC Hybrid Equity Fund","category":"Hybrid","amc":"HDFC",
     "risk":1,"horizon_min":3,"elss":0,"expense":1.68,"rating":4,"cagr_3y":13.4,"cagr_5y":15.1},
    {"name":"Kotak Equity Hybrid Fund","category":"Hybrid","amc":"Kotak",
     "risk":1,"horizon_min":3,"elss":0,"expense":0.47,"rating":4,"cagr_3y":14.2,"cagr_5y":16.0},
    {"name":"HDFC Short Term Debt Fund","category":"Debt","amc":"HDFC",
     "risk":0,"horizon_min":1,"elss":0,"expense":0.26,"rating":5,"cagr_3y":6.8,"cagr_5y":7.2},
    {"name":"Kotak Corporate Bond Fund","category":"Debt","amc":"Kotak",
     "risk":0,"horizon_min":1,"elss":0,"expense":0.25,"rating":4,"cagr_3y":7.1,"cagr_5y":7.4},
    {"name":"SBI Liquid Fund","category":"Liquid","amc":"SBI",
     "risk":0,"horizon_min":0,"elss":0,"expense":0.20,"rating":5,"cagr_3y":6.4,"cagr_5y":6.6},
]

fund_df = pd.DataFrame(FUND_DB)
fund_df.to_csv('data/fund_database.csv', index=False)

def recommend_portfolio(risk_level, horizon, elss_needed, goals, sip):
    """Weighted scoring recommender."""
    df = fund_df.copy()
    df = df[df['horizon_min'] <= horizon]
    if risk_level == 0:   df = df[df['risk'] <= 1]
    elif risk_level == 1: df = df[df['risk'] <= 2]

    # Score each fund
    def score(row):
        s = 0
        s += (row['cagr_5y'] / 30) * 40         # returns weight
        s += (1 - row['expense'] / 2) * 20       # low expense bonus
        s += (row['rating'] / 5) * 20            # rating weight
        if elss_needed and row['elss']: s += 15  # ELSS bonus
        if 'emergency' in goals and row['category'] == 'Liquid': s += 10
        if 'retirement' in goals and row['category'] in ['Flexi Cap','Large Cap']: s += 5
        return round(s, 2)

    df['score'] = df.apply(score, axis=1)
    df = df.sort_values('score', ascending=False)

    # Pick diversified funds (max 1 per AMC, max 6 total)
    picked, amcs_used = [], set()
    for _, row in df.iterrows():
        if len(picked) >= 6: break
        if row['amc'] not in amcs_used:
            picked.append(row)
            amcs_used.add(row['amc'])

    # Allocation logic
    alloc_map = {'conservative': {'Large Cap':40,'Hybrid':30,'Debt':20,'Liquid':10},
                 'moderate':     {'Large Cap':30,'Flexi Cap':25,'Mid Cap':20,'Hybrid':15,'Debt':10},
                 'aggressive':   {'Flexi Cap':25,'Mid Cap':25,'Small Cap':20,'Large Cap':20,'ELSS':10}}
    risk_key = ['conservative','moderate','aggressive'][risk_level]
    cat_alloc = alloc_map[risk_key]

    result = []
    total_alloc = 0
    for fund in picked:
        cat = fund['category']
        alloc = cat_alloc.get(cat, 10)
        result.append({**fund.to_dict(), 'allocation': alloc,
                       'sip_amount': round(sip * alloc / 100)})
        total_alloc += alloc

    # Normalize to 100%
    factor = 100 / total_alloc if total_alloc else 1
    for r in result:
        r['allocation'] = round(r['allocation'] * factor)
        r['sip_amount'] = round(sip * r['allocation'] / 100)

    return result[:6]

# Test run
sample = recommend_portfolio(1, 10, False, ['retirement','wealth'], 15000)
print(f"  ✓ Portfolio Recommender — {len(sample)} funds selected for sample profile")
for f in sample:
    print(f"    {f['name']:<40} {f['allocation']}%  ₹{f['sip_amount']:,}/mo")

# ─── SAVE ALL MODELS ──────────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
joblib.dump(risk_model, 'models/risk_profiler.pkl')
joblib.dump(sip_model,  'models/sip_predictor.pkl')
joblib.dump(scaler,     'models/sip_scaler.pkl')
joblib.dump(fund_df,    'models/fund_database.pkl')

meta = {
    "risk_model": {"type":"RandomForestClassifier","accuracy": round(risk_acc,4),
                   "features":list(X_r.columns), "classes":["Conservative","Moderate","Aggressive"]},
    "sip_model":  {"type":"GradientBoostingRegressor","r2": round(r2,4),"mae": round(mae,0),
                   "features": feat_cols},
    "recommender":{"type":"WeightedScoringModel","funds_in_db": len(FUND_DB)}
}
with open('models/model_meta.json','w') as f: json.dump(meta, f, indent=2)

print("\n✅ All models saved to /models/")
print(json.dumps(meta, indent=2))
