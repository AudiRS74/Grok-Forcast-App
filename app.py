import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from statsmodels.tsa.ar_model import AutoReg

# ---------------------------
# Grok-like Assistant placeholder
# ---------------------------
class GrokLikeAssistant:
    def __init__(self):
        self.traits = (
            "You are Grok 4 built by xAI. You are helpful, witty, and truth-seeking.\n"
            "Estimate trend (mu), oscillator (a0, phi0), volatility, jumps.\n"
            "Provide tables and stochastic forecasts with step-by-step explanations."
        )
    def analyze(self, query, data=None):
        return f"Placeholder analysis for query: {query}"

# ---------------------------
# Data extraction
# ---------------------------
def extract_hourly_closes(df):
    df['UTC'] = df['UTC'].astype(str)
    df['datetime'] = pd.to_datetime(df['UTC'], errors='coerce')
    df = df.dropna(subset=['datetime','Close'])
    hourly_closes = {}
    for _, row in df.iterrows():
        dt = row['datetime']
        if dt.minute == 59:
            hour = (dt.hour + 1) % 24
            hour_label = f"{hour:02d}:00"
            hourly_closes[hour_label] = float(row['Close'])
    return hourly_closes

# ---------------------------
# Default parameters
# ---------------------------
def get_default_params(m0=1.0894):
    return {
        'm0': m0, 'a0': 0.0003, 'phi0': 0, 'omega0': 2*np.pi/24,
        'mu': 0.00005, 'sigma_m': 1e-5, 'sigma_a': 1e-4,
        'sigma_phi': 1e-3, 'sigma_omega': 1e-5, 'lambda_poisson':0.05,
        'sigma_kappa':0.003, 'alpha0':1e-8, 'alpha1':0.05, 'beta1':0.94, 'rho':0.9
    }

# ---------------------------
# Parameter estimation
# ---------------------------
def estimate_params(hourly_closes):
    if not hourly_closes:
        return get_default_params()
    hours = sorted(hourly_closes.keys())
    closes = np.array([hourly_closes[h] for h in hours], dtype=float)
    t = np.arange(len(closes))
    if len(t)<2:
        return get_default_params(closes[0] if len(closes) else 1.0894)
    
    slope, intercept, *_ = linregress(t, closes)
    mu = slope
    m0 = intercept
    trend = m0 + mu*t
    detrended = closes - trend
    
    omega = 2*np.pi/24
    X = np.column_stack([np.cos(omega*t), np.sin(omega*t)])
    b, *_ = np.linalg.lstsq(X, detrended, rcond=None)
    a0 = np.sqrt(b[0]**2 + b[1]**2)
    phi0 = np.arctan2(b[1], b[0])
    
    s_t = a0*np.cos(omega*t + phi0)
    res = detrended - s_t
    
    if len(res)>1:
        model = AutoReg(res, lags=1, old_names=False).fit()
        rho = float(model.params[1]) if len(model.params)>1 else 0.9
        epsilon = model.resid
    else:
        rho = 0.9
        epsilon = res
    
    var_h = float(np.var(epsilon)) if len(epsilon)>0 else 1e-8
    alpha1 = 0.05
    beta1 = 0.94
    alpha0 = var_h*(1-alpha1-beta1)
    sigma_m = np.std(res)/100 if len(res)>1 else 1e-5
    sigma_a = 1e-4
    sigma_phi = 1e-3
    sigma_omega = 1e-5
    lambda_poisson = 0.05
    sigma_kappa = np.sqrt(var_h) if var_h>0 else 0.003
    
    return {
        'm0': m0, 'a0': abs(a0), 'phi0': float(phi0), 'omega0': omega,
        'mu': mu, 'sigma_m': float(sigma_m), 'sigma_a': float(sigma_a),
        'sigma_phi': float(sigma_phi), 'sigma_omega': float(sigma_omega),
        'lambda_poisson': float(lambda_poisson), 'sigma_kappa': float(sigma_kappa),
        'alpha0': max(alpha0,1e-10), 'alpha1':alpha1, 'beta1':beta1, 'rho':float(rho)
    }

# ---------------------------
# Price simulation
# ---------------------------
def simulate_prices(params, num_hours=23, seed=42):
    rng = np.random.default_rng(seed)
    m = np.zeros(num_hours+1)
    a = np.zeros(num_hours+1)
    phi = np.zeros(num_hours+1)
    omega = np.zeros(num_hours+1)
    P_mid = np.zeros(num_hours)
    m[0] = params['m0']
    a[0] = params['a0']
    phi[0] = params['phi0']
    omega[0] = params['omega0']
    
    for t in range(1,num_hours+1):
        eta_m = rng.normal(0, params['sigma_m'])
        m[t] = m[t-1] + params['mu'] + eta_m
        eta_a = rng.normal(0, params['sigma_a'])
        a[t] = a[t-1]*np.exp(eta_a)
        eta_phi = rng.normal(0, params['sigma_phi'])
        phi[t] = phi[t-1]+eta_phi
        eta_omega = rng.normal(0, params['sigma_omega'])
        omega[t] = omega[t-1]+eta_omega
        
        s_t = a[t]*np.cos(omega[t]*t + phi[t])
        N_t = rng.poisson(params['lambda_poisson'])
        j_t = sum(rng.normal(0, params['sigma_kappa']) for _ in range(N_t))
        P_mid[t-1] = m[t]+s_t+j_t
    return P_mid

# ---------------------------
# Streamlit App
# ---------------------------
st.title("📈 Grok-Like Hourly Forecast App")

assistant = GrokLikeAssistant()
query = st.text_input("Enter analysis query (optional):")
if query:
    st.write(assistant.analyze(query))

uploaded_file = st.file_uploader("Upload CSV (or leave blank for default simulation)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, header=None, names=['UTC','Open','High','Low','Close','Volume'])
    df = df.dropna(subset=['Close'])
    hourly_live = extract_hourly_closes(df)
    params = estimate_params(hourly_live)
else:
    hourly_live = {}
    params = get_default_params()

P_mid = simulate_prices(params)
hours = [f"{h:02d}:00" for h in range(1,24)]
table = []
for i,hour in enumerate(hours):
    live = hourly_live.get(hour, None)
    predicted = float(P_mid[i]) if i<len(P_mid) else None
    diff = (live-predicted) if (live is not None and predicted is not None) else None
    table.append({
        'Hour': hour,
        'Live Price': round(live,4) if live is not None else '',
        'Predicted Price': round(predicted,6) if predicted is not None else '',
        'Difference': round(diff,6) if diff is not None else ''
    })
df_table = pd.DataFrame(table)
st.subheader("Hourly Prices Table (Live vs Predicted)")
st.dataframe(df_table)

if st.checkbox("Show chart"):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(hours,P_mid,label='Predicted Price')
    if hourly_live:
        live_hours = sorted(hourly_live.keys())
        live_values = [hourly_live[h] for h in live_hours]
        ax.plot(live_hours, live_values,label='Live Price',marker='o')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Price')
    ax.set_title('Hourly Prices: Live vs Predicted')
    ax.legend()
    st.pyplot(fig)