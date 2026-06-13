"""
RetireWise+ ML Backend
Flask app serving 3 ML models + live MFAPI data
"""
from flask import Flask, request, jsonify, render_template
import joblib, json, numpy as np, requests
from functools import lru_cache

app = Flask(__name__)

# ── Load models ────────────────────────────────────────────────────────────────
risk_model = joblib.load('models/risk_profiler.pkl')
sip_model  = joblib.load('models/sip_predictor.pkl')
sip_scaler = joblib.load('models/sip_scaler.pkl')
fund_db    = joblib.load('models/fund_database.pkl')

with open('models/model_meta.json') as f:
    model_meta = json.load(f)

RISK_LABELS = ['Conservative', 'Moderate', 'Aggressive']
CAT_COLORS  = {'Large Cap':'blue','Mid Cap':'amber','Small Cap':'red',
               'ELSS':'green','Hybrid':'blue','Debt':'purple','Liquid':'gray','Flexi Cap':'teal'}

# ── MFAPI helpers ──────────────────────────────────────────────────────────────
FUND_SCHEME_MAP = {
    "Mirae Asset Large Cap Fund":       "118989",
    "Parag Parikh Flexi Cap Fund":      "122639",
    "HDFC Flexi Cap Fund":              "100033",
    "Axis Midcap Fund":                 "120505",
    "Kotak Emerging Equity Fund":       "120238",
    "Nippon India Small Cap Fund":      "118778",
    "HDFC Small Cap Fund":              "100229",
    "SBI Magnum ELSS Fund":             "102885",
    "Axis Long Term Equity Fund":       "120503",
    "Mirae Asset Tax Saver Fund":       "126878",
    "HDFC Hybrid Equity Fund":          "100030",
    "Kotak Equity Hybrid Fund":         "120262",
    "HDFC Short Term Debt Fund":        "100026",
    "Kotak Corporate Bond Fund":        "120259",
    "SBI Liquid Fund":                  "102837",
    "HDFC Top 100 Fund":                "100025",
    "Axis Bluechip Fund":               "120465",
}

@lru_cache(maxsize=32)
def fetch_nav(scheme_code):
    try:
        r = requests.get(f"https://api.mfapi.in/mf/{scheme_code}", timeout=5)
        d = r.json()
        latest = d['data'][0]
        week52 = d['data'][min(364, len(d['data'])-1)]
        return {
            "nav":       round(float(latest['nav']), 2),
            "date":      latest['date'],
            "nav_52w_low":  round(float(week52['nav']), 2),
            "nav_52w_high": round(float(d['data'][0]['nav']), 2),
            "fund_name": d['meta']['scheme_name'],
            "scheme_code": scheme_code
        }
    except:
        return {"nav": None, "date": None, "error": "NAV unavailable"}

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', meta=model_meta)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json

    # 1. Risk Profiler
    crash_map = {'panic': 0, 'wait': 1, 'buy': 2}
    exp_map   = {'beginner': 0, 'some': 1, 'experienced': 2}
    focus_map = {'safety': 0, 'balanced': 1, 'growth': 2}

    income   = float(data['income'])
    expenses = float(data['expenses'])
    savings_ratio = (income - expenses) / income if income > 0 else 0

    risk_features = np.array([[
        int(data['age']),
        income,
        expenses,
        round(savings_ratio, 3),
        int(data.get('dependents', 0)),
        int(data['horizon']),
        float(data.get('existing', 0)),
        crash_map.get(data.get('crash_behavior', 'wait'), 1),
        exp_map.get(data.get('experience', 'some'), 1),
        focus_map.get(data.get('focus', 'balanced'), 1),
    ]])

    risk_pred   = int(risk_model.predict(risk_features)[0])
    risk_proba  = risk_model.predict_proba(risk_features)[0].tolist()
    risk_label  = RISK_LABELS[risk_pred]

    # 2. SIP Return Predictor (3 scenarios)
    sip    = float(data['sip'])
    horizon = int(data['horizon'])
    stepup  = int(data.get('stepup', 0))

    projections = {}
    for label, mkt in [('conservative', 0), ('realistic', 1), ('optimistic', 2)]:
        feat = np.array([[sip, horizon, stepup, risk_pred, mkt]])
        feat_sc = sip_scaler.transform(feat)
        fv = float(np.expm1(sip_model.predict(feat_sc)[0]))
        projections[label] = round(fv)

    # 3. Portfolio Recommender
    elss_needed = data.get('elss') == 'yes'
    goals       = data.get('goals', [])

    df = fund_db.copy()
    df = df[df['horizon_min'] <= horizon]
    if risk_pred == 0:   df = df[df['risk'] <= 1]
    elif risk_pred == 1: df = df[df['risk'] <= 2]

    def score_fund(row):
        s = (row['cagr_5y'] / 30) * 40 + (1 - row['expense'] / 2) * 20 + (row['rating'] / 5) * 20
        if elss_needed and row['elss']:  s += 15
        if 'emergency' in goals and row['category'] == 'Liquid': s += 10
        if 'retirement' in goals and row['category'] in ['Flexi Cap', 'Large Cap']: s += 5
        if 'wealth' in goals and row['category'] in ['Mid Cap', 'Small Cap']: s += 5
        return s

    df['score'] = df.apply(score_fund, axis=1)
    df = df.sort_values('score', ascending=False)

    alloc_map = {
        0: {'Large Cap': 40, 'Hybrid': 30, 'Debt': 20, 'Liquid': 10},
        1: {'Large Cap': 30, 'Flexi Cap': 25, 'Mid Cap': 20, 'Hybrid': 15, 'Debt': 10},
        2: {'Flexi Cap': 25, 'Mid Cap': 25, 'Small Cap': 20, 'Large Cap': 20, 'ELSS': 10}
    }
    cat_alloc = alloc_map[risk_pred]

    picked, amcs_used, funds_out = [], set(), []
    for _, row in df.iterrows():
        if len(picked) >= 6: break
        if row['amc'] not in amcs_used:
            picked.append(row)
            amcs_used.add(row['amc'])

    total_alloc = sum(cat_alloc.get(f['category'], 10) for f in picked)
    for fund in picked:
        raw_alloc = cat_alloc.get(fund['category'], 10)
        alloc = round(raw_alloc * 100 / total_alloc)
        sip_split = round(sip * alloc / 100)
        scheme_code = FUND_SCHEME_MAP.get(fund['name'])
        nav_data = fetch_nav(scheme_code) if scheme_code else {}
        funds_out.append({
            "name":       fund['name'],
            "category":   fund['category'],
            "amc":        fund['amc'],
            "allocation": alloc,
            "sip_amount": sip_split,
            "cagr_3y":    fund['cagr_3y'],
            "cagr_5y":    fund['cagr_5y'],
            "expense":    fund['expense'],
            "rating":     int(fund['rating']),
            "color":      CAT_COLORS.get(fund['category'], 'gray'),
            "nav":        nav_data.get('nav'),
            "nav_date":   nav_data.get('date'),
            "scheme_code": scheme_code,
        })

    # Goal structuring
    goal_priority = {
        'emergency': ('short', 'high'),
        'house':     ('medium', 'high'),
        'education': ('medium', 'high'),
        'marriage':  ('medium', 'medium'),
        'retirement':('long',   'high'),
        'wealth':    ('long',   'medium'),
    }
    structured_goals = []
    for g in goals:
        tf, pr = goal_priority.get(g, ('long', 'medium'))
        if tf == 'long' and horizon < 7:  tf = 'medium'
        if tf == 'medium' and horizon < 3: tf = 'short'
        structured_goals.append({"name": g.capitalize(), "timeframe": tf, "priority": pr})

    # SIP affordability
    savings = income - expenses
    sip_pct = (sip / income * 100) if income > 0 else 0
    affordable = sip <= savings * 0.8

    return jsonify({
        "risk": {
            "label":      risk_label,
            "index":      risk_pred,
            "confidence": round(max(risk_proba) * 100, 1),
            "probabilities": {RISK_LABELS[i]: round(risk_proba[i]*100,1) for i in range(3)}
        },
        "projections":    projections,
        "funds":          funds_out,
        "goals":          structured_goals,
        "sip_affordable": affordable,
        "sip_pct_income": round(sip_pct, 1),
        "monthly_savings": round(savings),
        "model_info":     model_meta
    })

@app.route('/api/nav/<scheme_code>')
def get_nav(scheme_code):
    return jsonify(fetch_nav(scheme_code))

@app.route('/api/models')
def get_models():
    return jsonify(model_meta)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
