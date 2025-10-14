#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from statsmodels.tsa.ar_model import AutoReg

# Use the official OpenAI client if you want to call the API.
# Example for the modern client:
# from openai import OpenAI
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Utilities ----------
def ensure_no_nbsp(s):
    # Replace non-breaking spaces if present
    return s.replace("\u00A0", " ")

# ---------- Grok-like assistant (placeholder) ----------
class GrokLikeAssistant:
    def __init__(self, model="gpt-4"):
        self.model = model
        self.traits = (
            "You are Grok 4 built by xAI. You are helpful, witty, and truth-seeking.\n"
            "When analyzing data, estimate trend (mu), oscillator (a0, phi0), volatility, and jumps.\n"
            "Provide detailed explanations, tables, and stochastic forecasts. For math, explain step-by-step."
        )

    def analyze(self, query, data=None):
        prompt = self.traits + "\n\nUser query: " + query
        if data is not None:
            prompt += "\nData summary: " + str(data)[:2000]  # short summary
        prompt += "\n\nPlease analyze and forecast."
        # Placeholder: to call real API, uncomment and use your client
        # response = client.chat.completions.create(model=self.model, messages=[{"role":"system","content":prompt}])
        # return response.choices[0].message.content
        return "Placeholder analysis: trend estimated, oscillator fitted, residuals inspected."

# ---------- Data extraction ----------
def extract_hourly_closes(df):
    # Normalize column strings if necessary
    if 'UTC' not in df.columns:
        raise ValueError("CSV must include UTC column (first column).")
    # remove non-breaking spaces in UTC strings
    df['UTC'] = df['UTC'].astype(str).apply(ensure_no_nbsp)
    # Try several datetime formats
    possible_formats = ['%d.%m.%Y %H:%M:%S.000 UTC', '%Y-%m-%d %H:%M:%S', '%d.%m.%Y %H:%M:%S']
    parsed = None
    for fmt in possible_formats:
        try:
            parsed = pd.to_datetime(df['UTC'], format=fmt, errors='coerce')
            if parsed.notna().sum() > 0:
                break
        except Exception:
            parsed = pd.to_datetime(df['UTC'], errors='coerce')
            break
    df['datetime'] = parsed
    df = df.dropna(subset=['datetime', 'Close'])
    hourly_closes = {}
    for _, row in df.iterrows():
        dt = row['datetime']
        # treat rows at minute 59 as hour-close (convention used in your original script)
        if dt.minute == 59:
            hour = (dt.hour + 1) % 24
            hour_label = f"{hour:02d}:00"
            hourly_closes[hour_label] = float(row['Close'])
    return hourly_closes

# ---------- Parameter estimation ----------
def get_default_params(m0=1.1712):
    return {
        'm0': m0, 'a0': 0.0003, 'phi0': 0, 'omega0': 2 * np.pi / 24,
        'mu': -0.00028, 'sigma_m': np.sqrt(1e-10), 'sigma_a': np.sqrt(1e-8),
        'sigma_phi': np.sqrt(1e-6), 'sigma_omega': np.sqrt(1e-10),
        'lambda_poisson': 0.05, 'sigma_kappa': np.sqrt(4e-8),
        'alpha0': 1e-8, 'alpha1': 0.05, 'beta1': 0.94, 'rho': 0.9
    }

def estimate_params(hourly_closes):
    if not hourly_closes:
        return get_default_params()
    hours = sorted(hourly_closes.keys())
    closes = np.array([hourly_closes[h] for h in hours], dtype=float)
    t = np.arange(len(closes))
    if len(t) < 2:
        return get_default_params(closes[0] if len(closes) else 1.1712)

    slope, intercept, _, _, _ = linregress(t, closes)
    mu = slope
    m0 = intercept
    trend = m0 + mu * t
    detrended = closes - trend

    omega = 2 * np.pi / 24
    X = np.column_stack([np.cos(omega * t), np.sin(omega * t)])
    b, *_ = np.linalg.lstsq(X, detrended, rcond=None)
    a0 = np.sqrt(b[0]**2 + b[1]**2)
    phi0 = np.arctan2(b[1], b[0])

    s_t = a0 * np.cos(omega * t + phi0)
    res = detrended - s_t

    if len(res) > 1:
        model = AutoReg(res, lags=1, old_names=False).fit()
        rho = float(model.params[1]) if len(model.params) > 1 else 0.9
        epsilon = model.resid
    else:
        rho = 0.9
        epsilon = res

    var_h = float(np.var(epsilon)) if len(epsilon) > 0 else 1e-8
    alpha1 = 0.05
    beta1 = 0.94
    alpha0 = var_h * (1 - alpha1 - beta1)
    sigma_m = np.std(res) / 100 if len(res) > 1 else np.sqrt(1e-10)
    sigma_a = np.sqrt(1e-8)
    sigma_phi = np.sqrt(1e-6)
    sigma_omega = np.sqrt(1e-10)
    lambda_poisson = 0.05
    sigma_kappa = np.sqrt(var_h) if var_h > 0 else np.sqrt(4e-8)

    return {
        'm0': m0, 'a0': abs(a0), 'phi0': float(phi0), 'omega0': omega,
        'mu': mu, 'sigma_m': float(sigma_m), 'sigma_a': float(sigma_a),
        'sigma_phi': float(sigma_phi), 'sigma_omega': float(sigma_omega),
        'lambda_poisson': float(lambda_poisson), 'sigma_kappa': float(sigma_kappa),
        'alpha0': max(alpha0, 1e-10), 'alpha1': alpha1, 'beta1': beta1, 'rho': float(rho)
    }

# ---------- Simulation ----------
def simulate_prices(params, num_hours=23, seed=42):
    rng = np.random.default_rng(seed)
    m = np.zeros(num_hours + 1)
    a = np.zeros(num_hours + 1)
    phi = np.zeros(num_hours + 1)
    omega = np.zeros(num_hours + 1)
    P_mid = np.zeros(num_hours)
    m[0] = params['m0']
    a[0] = params['a0']
    phi[0] = params['phi0']
    omega[0] = params['omega0']
    for t in range(1, num_hours + 1):
        eta_m = rng.normal(0, params['sigma_m'])
        m[t] = m[t - 1] + params['mu'] + eta_m
        eta_a = rng.normal(0, params['sigma_a'])
        a[t] = a[t - 1] * np.exp(eta_a)
        eta_phi = rng.normal(0, params['sigma_phi'])
        phi[t] = phi[t - 1] + eta_phi
        eta_omega = rng.normal(0, params['sigma_omega'])
        omega[t] = omega[t - 1] + eta_omega
        s_t = a[t] * np.cos(omega[t] * t + phi[t])
        N_t = rng.poisson(params['lambda_poisson'])
        j_t = sum(rng.normal(0, params['sigma_kappa']) for _ in range(N_t))
        P_mid[t - 1] = m[t] + s_t + j_t
    return P_mid

# ---------- Main ----------
def main():
    # Obtain OpenAI key from environment (do NOT hardcode!)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        print("Warning: OPENAI_API_KEY not found in environment. API calls will be skipped.")
        # You can proceed without enabling the OpenAI API, or set the env var:
        # export OPENAI_API_KEY="sk-..."
        # Or use python-dotenv to load from a local .env file (not recommended for production)
    assistant = GrokLikeAssistant()

    # Optionally prompt user
    user_query = input("Enter your analysis query (or press Enter to skip): ").strip()
    if user_query:
        analysis = assistant.analyze(user_query)
        print("AI Assistant Response:\n", analysis)

    path_or_data = input("Enter path to CSV file (or paste CSV content): ").strip()
    if not path_or_data:
        print("No file provided. Using default parameters.")
        params = get_default_params()
    else:
        try:
            if os.path.exists(path_or_data):
                df = pd.read_csv(path_or_data, header=None, names=['UTC','Open','High','Low','Close','Volume'])
            else:
                from io import StringIO
                df = pd.read_csv(StringIO(path_or_data), header=None, names=['UTC','Open','High','Low','Close','Volume'])
            df = df.dropna(subset=['Close'])
            hourly_live = extract_hourly_closes(df)
            params = estimate_params(hourly_live)
            print("Estimated Parameters:")
            for k, v in params.items():
                print(f"  {k}: {v}")
        except Exception as e:
            print("Error reading/parsing CSV:", e)
            params = get_default_params()

    P_mid = simulate_prices(params)
    hours = [f"{h:02d}:00" for h in range(1, 24)]
    data = []
    live_map = locals().get('hourly_live', {})
    for i, hour in enumerate(hours):
        live = live_map.get(hour, None)
        predicted = float(P_mid[i]) if i < len(P_mid) else None
        difference = (live - predicted) if (live is not None and predicted is not None) else None
        data.append({
            'Hour': hour,
            'Live Price': round(live, 4) if live is not None else '',
            'Predicted Price': round(predicted, 6) if predicted is not None else '',
            'Difference': round(difference, 6) if difference is not None else ''
        })

    df_table = pd.DataFrame(data)
    print("\nCalculated Prices Table (Live vs. Predicted):")
    print(df_table.to_string(index=False))

    show_chart = input("Show chart? (y/n): ").lower() == 'y'
    if show_chart:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(hours, P_mid, label='Predicted Price')
        if live_map:
            live_hours = sorted(live_map.keys())
            live_values = [live_map[h] for h in live_hours]
            ax.plot(live_hours, live_values, label='Live Price', marker='o')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Price')
        ax.set_title('Grok-like Model — Hourly Price Forecast')
        ax.legend()
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
