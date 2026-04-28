import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────
# PATHS  — always resolve relative to this script (src/)
# ─────────────────────────────────────────────
DATA_DIR = Path(__file__).parent

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="WOW! Pricing Tool",
    page_icon="🏘️",
    layout="wide",
)

# ─────────────────────────────────────────────
# GLOBAL CSS  (Bishan Park sage-green theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f5f7f2;
}
h1, h2, h3, h4 { font-family: 'Playfair Display', serif; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #2d5a27 0%, #3d8a4a 50%, #5aa55e 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "";
    position: absolute;
    top: -40px; right: -40px;
    width: 220px; height: 220px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    line-height: 1.2;
}
.hero-sub {
    font-size: 1rem;
    opacity: 0.88;
    margin: 0 0 1.5rem 0;
}
.hero-stats {
    display: flex;
    gap: 2.5rem;
}
.hero-stat-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
}
.hero-stat-lbl {
    font-size: 0.78rem;
    opacity: 0.8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.4rem; }
.metric-card {
    background: white;
    border: 1.5px solid #b8d9bc;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    flex: 1;
    box-shadow: 0 2px 8px rgba(61,138,74,0.08);
}
.metric-card-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #5a8060;
    margin-bottom: 0.3rem;
}
.metric-card-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: #2d5a27;
}
.metric-card-sub {
    font-size: 0.78rem;
    color: #888;
    margin-top: 0.2rem;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 0.95rem;
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1.2rem;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background: linear-gradient(135deg, #2d5a27, #3d8a4a) !important;
    color: white !important;
    border-bottom: none !important;
}

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, #2d5a27, #3d8a4a);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    padding: 0.6rem 2rem;
    transition: transform 0.15s;
}
.stButton > button:hover { transform: translateY(-2px); }

/* ── Price result box ── */
.price-result {
    background: linear-gradient(135deg, #2d5a27, #3d8a4a);
    color: white;
    border-radius: 14px;
    padding: 1.8rem 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.price-result-label { font-size: 0.85rem; opacity: 0.85; text-transform: uppercase; letter-spacing: 0.06em; }
.price-result-value { font-family: 'Playfair Display', serif; font-size: 2.8rem; font-weight: 700; margin: 0.3rem 0; }
.price-result-psf { font-size: 0.9rem; opacity: 0.82; }

/* ── Section headers ── */
.section-header {
    color: #2d5a27;
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    margin: 1.5rem 0 0.8rem;
    border-bottom: 2px solid #b8d9bc;
    padding-bottom: 0.4rem;
}

/* ── Info box ── */
.info-box {
    background: #eef5ee;
    border-left: 4px solid #3d8a4a;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: #2d5a27;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def _clean(df):
    df.columns = df.columns.str.lower()
    for col in ["mall_nearest_distance", "hawker_nearest_distance",
                 "mrt_nearest_distance", "bus_stop_nearest_distance",
                 "pri_sch_nearest_distance", "sec_sch_nearest_dist"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["resale_price", "floor_area_sqm", "hdb_age", "mid"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Add town_encoded and flat_type_encoded (LabelEncoder — alphabetical order)
    towns_sorted = sorted(df["town"].dropna().unique().tolist())
    town_map = {t: i for i, t in enumerate(towns_sorted)}
    df["town_encoded"] = df["town"].map(town_map)

    flat_types_sorted = sorted(df["flat_type"].dropna().unique().tolist())
    flat_type_map = {f: i for i, f in enumerate(flat_types_sorted)}
    df["flat_type_encoded"] = df["flat_type"].map(flat_type_map)
    return df

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    train = _clean(pd.read_csv(DATA_DIR / "train_sample.csv", low_memory=False))
    return train

@st.cache_data(show_spinner="Loading full dataset for comparables…")
def load_all_data():
    train = _clean(pd.read_csv(DATA_DIR / "train_sample.csv", low_memory=False))
    test  = _clean(pd.read_csv(DATA_DIR / "test_sample.csv",  low_memory=False))
    return pd.concat([train, test], ignore_index=True)

@st.cache_resource(show_spinner="Loading prediction model…")
def load_model():
    with open(DATA_DIR / "model.pkl", "rb") as f:
        return pickle.load(f)


df     = load_data()       # train only — used for model training in tabs 1 & 2
df_all = load_all_data()   # train + test combined — used for comparables in tab 3
xgb_model = load_model()

TOWNS = sorted(df["town"].dropna().unique().tolist())
FLAT_TYPES = sorted(df["flat_type"].dropna().unique().tolist())

# Encoding maps (LabelEncoder — alphabetical order)
TOWN_ENCODING      = {t: i for i, t in enumerate(sorted(TOWNS))}
FLAT_TYPE_ENCODING = {f: i for i, f in enumerate(sorted(FLAT_TYPES))}

# New 9-feature set (flat_type_encoded replaces floor_area_sqm)
MODEL_9_FEATURES = [
    "flat_type_encoded",
    "town_encoded",
    "hdb_age",
    "mid",
    "mrt_nearest_distance",
    "pri_sch_nearest_distance",
    "hawker_within_1km",
    "mall_within_1km",
    "bus_stop_nearest_distance",
]

FEATURE_LABELS = {
    "flat_type_encoded":        "Flat Type",
    "town_encoded":             "Town",
    "hdb_age":                  "HDB Age (years)",
    "mid":                      "Storey Mid-Level",
    "mrt_nearest_distance":     "MRT Distance (m)",
    "pri_sch_nearest_distance": "Pri School Distance (m)",
    "hawker_within_1km":        "Hawker Within 1 km",
    "mall_within_1km":          "Mall Within 1 km",
    "bus_stop_nearest_distance":"Bus Stop Distance (m)",
}

AVAILABLE_FEATURES = [c for c in MODEL_9_FEATURES if c in df.columns]

# ─────────────────────────────────────────────
# HERO BANNER (inline SVG)
# ─────────────────────────────────────────────
n_rows = len(df)
towns_count = df["town"].nunique()
median_price = df["resale_price"].median()

st.markdown(f"""
<div class="hero-banner">
  <svg style="position:absolute;right:24px;top:50%;transform:translateY(-50%);opacity:0.18;"
       width="260" height="160" viewBox="0 0 260 160" fill="none" xmlns="http://www.w3.org/2000/svg">
    <!-- HDB blocks -->
    <rect x="10" y="40" width="35" height="110" rx="3" fill="white"/>
    <rect x="14" y="50" width="6" height="8" rx="1" fill="#cce8d0"/>
    <rect x="23" y="50" width="6" height="8" rx="1" fill="#cce8d0"/>
    <rect x="32" y="50" width="6" height="8" rx="1" fill="#cce8d0"/>
    <rect x="14" y="65" width="6" height="8" rx="1" fill="#cce8d0"/>
    <rect x="23" y="65" width="6" height="8" rx="1" fill="#cce8d0"/>
    <rect x="32" y="65" width="6" height="8" rx="1" fill="#cce8d0"/>
    <rect x="14" y="80" width="6" height="8" rx="1" fill="#cce8d0"/>
    <rect x="23" y="80" width="6" height="8" rx="1" fill="#cce8d0"/>
    <rect x="32" y="80" width="6" height="8" rx="1" fill="#cce8d0"/>
    <rect x="14" y="95" width="6" height="8" rx="1" fill="#cce8d0"/>
    <rect x="23" y="95" width="6" height="8" rx="1" fill="#cce8d0"/>
    <rect x="32" y="95" width="6" height="8" rx="1" fill="#cce8d0"/>
    <rect x="55" y="20" width="40" height="130" rx="3" fill="white"/>
    <rect x="60" y="30" width="7" height="9" rx="1" fill="#cce8d0"/>
    <rect x="71" y="30" width="7" height="9" rx="1" fill="#cce8d0"/>
    <rect x="82" y="30" width="7" height="9" rx="1" fill="#cce8d0"/>
    <rect x="60" y="47" width="7" height="9" rx="1" fill="#cce8d0"/>
    <rect x="71" y="47" width="7" height="9" rx="1" fill="#cce8d0"/>
    <rect x="82" y="47" width="7" height="9" rx="1" fill="#cce8d0"/>
    <rect x="60" y="64" width="7" height="9" rx="1" fill="#cce8d0"/>
    <rect x="71" y="64" width="7" height="9" rx="1" fill="#cce8d0"/>
    <rect x="82" y="64" width="7" height="9" rx="1" fill="#cce8d0"/>
    <rect x="60" y="81" width="7" height="9" rx="1" fill="#cce8d0"/>
    <rect x="71" y="81" width="7" height="9" rx="1" fill="#cce8d0"/>
    <rect x="82" y="81" width="7" height="9" rx="1" fill="#cce8d0"/>
    <rect x="105" y="55" width="30" height="95" rx="3" fill="white"/>
    <rect x="109" y="63" width="5" height="7" rx="1" fill="#cce8d0"/>
    <rect x="117" y="63" width="5" height="7" rx="1" fill="#cce8d0"/>
    <rect x="125" y="63" width="5" height="7" rx="1" fill="#cce8d0"/>
    <rect x="109" y="78" width="5" height="7" rx="1" fill="#cce8d0"/>
    <rect x="117" y="78" width="5" height="7" rx="1" fill="#cce8d0"/>
    <rect x="125" y="78" width="5" height="7" rx="1" fill="#cce8d0"/>
    <!-- Trees -->
    <ellipse cx="160" cy="125" rx="18" ry="20" fill="#5aa55e"/>
    <rect x="157" y="138" width="6" height="12" fill="#3d6b40"/>
    <ellipse cx="195" cy="130" rx="14" ry="16" fill="#4d9455"/>
    <rect x="192" y="142" width="5" height="8" fill="#3d6b40"/>
    <ellipse cx="225" cy="120" rx="20" ry="22" fill="#6bb86f"/>
    <rect x="222" y="135" width="6" height="15" fill="#3d6b40"/>
    <!-- Ground -->
    <rect x="0" y="148" width="260" height="12" rx="2" fill="#4d9455" opacity="0.5"/>
  </svg>
  <p class="hero-title">🏘️ WOW! HDB Pricing Tool</p>
  <p class="hero-sub">Data-backed resale price insights for WOW! Real Estate agents · The Model Citizens</p>
  <div class="hero-stats">
    <div>
      <div class="hero-stat-val">{n_rows:,}</div>
      <div class="hero-stat-lbl">Transactions</div>
    </div>
    <div>
      <div class="hero-stat-val">{towns_count}</div>
      <div class="hero-stat-lbl">Towns</div>
    </div>
    <div>
      <div class="hero-stat-val">S${median_price/1e3:.0f}K</div>
      <div class="hero-stat-lbl">Median Price</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊  Linear Regression", "🌲  Random Forest", "📈  XGBoost", "🔮  Live Prediction"])


# ══════════════════════════════════════════════
# TAB 1 — MODEL EVALUATION (Linear Regression)
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">Linear Regression — Model Evaluation</p>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">Evaluated on a fixed <strong>90% train / 10% test</strong> split using Linear Regression. '
                'Select which features to include in the model.</div>', unsafe_allow_html=True)

    # ── Feature multiselect (must not be removed) ──
    label_to_col = {FEATURE_LABELS.get(c, c): c for c in AVAILABLE_FEATURES}
    default_labels = [FEATURE_LABELS.get(c, c) for c in AVAILABLE_FEATURES]

    selected_labels = st.multiselect(
        "Select features (X) for the Linear Regression model:",
        options=list(label_to_col.keys()),
        default=default_labels,
        help="Choose which features to train the model on. At least one is required.",
    )
    selected_features = [label_to_col[l] for l in selected_labels]

    st.divider()

    if not selected_features:
        st.warning("⚠️ Please select at least one feature to run the model.")
    else:
        # Build model dataset
        needed = selected_features + ["resale_price"]
        df_model = df[needed].dropna()

        X = df_model[selected_features]
        y = df_model["resale_price"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.10, random_state=42
        )

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred_train = lr.predict(X_train)
        y_pred_test  = lr.predict(X_test)

        # Metrics
        r2_tr  = r2_score(y_train, y_pred_train)
        r2_te  = r2_score(y_test,  y_pred_test)
        mae_tr = mean_absolute_error(y_train, y_pred_train)
        mae_te = mean_absolute_error(y_test,  y_pred_test)
        rmse_tr = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_te = np.sqrt(mean_squared_error(y_test,  y_pred_test))

        # Null model RMSE
        null_preds = np.full(len(y_test), y_train.mean())
        null_rmse  = np.sqrt(mean_squared_error(y_test, null_preds))

        # ── Metric cards ──
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card">
            <div class="metric-card-label">Train R²</div>
            <div class="metric-card-value">{r2_tr:.3f}</div>
            <div class="metric-card-sub">Variance explained (train)</div>
          </div>
          <div class="metric-card">
            <div class="metric-card-label">Test R²</div>
            <div class="metric-card-value">{r2_te:.3f}</div>
            <div class="metric-card-sub">Variance explained (test)</div>
          </div>
          <div class="metric-card">
            <div class="metric-card-label">Test MAE</div>
            <div class="metric-card-value">S${mae_te:,.0f}</div>
            <div class="metric-card-sub">Mean absolute error</div>
          </div>
          <div class="metric-card">
            <div class="metric-card-label">Test RMSE</div>
            <div class="metric-card-value">S${rmse_te:,.0f}</div>
            <div class="metric-card-sub">vs null RMSE S${null_rmse:,.0f}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        # ── Actual vs Predicted scatter ──
        with col_a:
            st.markdown('<p class="section-header">Actual vs Predicted</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            ax.scatter(y_test / 1e3, y_pred_test / 1e3,
                       alpha=0.25, s=8, color="#3d8a4a")
            lims = [min(y_test.min(), y_pred_test.min()) / 1e3,
                    max(y_test.max(), y_pred_test.max()) / 1e3]
            ax.plot(lims, lims, color="#c0392b", lw=1.5, ls="--", label="Perfect fit")
            ax.set_xlabel("Actual Price (S$'000)", fontsize=10)
            ax.set_ylabel("Predicted Price (S$'000)", fontsize=10)
            ax.set_title(f"R² (test) = {r2_te:.3f}", fontsize=11)
            ax.legend(fontsize=9)
            fig.patch.set_facecolor("#f5f7f2")
            ax.set_facecolor("#f5f7f2")
            st.pyplot(fig)
            plt.close(fig)

        # ── Residuals distribution ──
        with col_b:
            st.markdown('<p class="section-header">Residuals Distribution</p>', unsafe_allow_html=True)
            residuals = (y_test - y_pred_test) / 1e3
            fig2, ax2 = plt.subplots(figsize=(5.5, 4.5))
            ax2.hist(residuals, bins=60, color="#3d8a4a", edgecolor="white", alpha=0.85)
            ax2.axvline(0, color="#c0392b", lw=1.8, ls="--", label="Zero error")
            ax2.set_xlabel("Residual (S$'000)", fontsize=10)
            ax2.set_ylabel("Count", fontsize=10)
            ax2.set_title("Distribution of Prediction Errors", fontsize=11)
            ax2.legend(fontsize=9)
            fig2.patch.set_facecolor("#f5f7f2")
            ax2.set_facecolor("#f5f7f2")
            st.pyplot(fig2)
            plt.close(fig2)

        # ── Coefficients table ──
        st.markdown('<p class="section-header">Model Coefficients</p>', unsafe_allow_html=True)
        coef_df = pd.DataFrame({
            "Feature": [FEATURE_LABELS.get(f, f) for f in selected_features],
            "Coefficient": lr.coef_,
        }).sort_values("Coefficient", key=abs, ascending=False)
        coef_df["Coefficient"] = coef_df["Coefficient"].apply(lambda x: f"{'+ ' if x > 0 else '− '}S${abs(x):,.2f}")

        st.dataframe(
            coef_df,
            use_container_width=True,
            hide_index=True,
        )

        # ── Sample predictions ──
        st.markdown('<p class="section-header">Sample Predictions (first 10 test rows)</p>', unsafe_allow_html=True)
        sample_df = X_test.head(10).copy()
        sample_df["Actual (S$)"]    = y_test.head(10).values
        sample_df["Predicted (S$)"] = y_pred_test[:10]
        sample_df["Error (S$)"]     = sample_df["Actual (S$)"] - sample_df["Predicted (S$)"]
        sample_df.index = range(1, 11)
        # Format money cols
        for col in ["Actual (S$)", "Predicted (S$)", "Error (S$)"]:
            sample_df[col] = sample_df[col].apply(lambda x: f"S${x:,.0f}")
        sample_df.columns = [FEATURE_LABELS.get(c, c) if c in FEATURE_LABELS else c
                              for c in sample_df.columns]
        st.dataframe(sample_df, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 — MODEL EVALUATION (Random Forest, live retrain)
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">Random Forest — Model Evaluation</p>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">Select features below — a fresh Random Forest model will be trained '
                'on your selection using a fixed <strong>90% train / 10% test</strong> split. '
                'Training may take a few seconds.</div>', unsafe_allow_html=True)

    # ── Feature multiselect ──
    rf_label_to_col  = {FEATURE_LABELS.get(c, c): c for c in AVAILABLE_FEATURES}
    rf_default_labels = [FEATURE_LABELS.get(c, c) for c in AVAILABLE_FEATURES]

    rf_selected_labels = st.multiselect(
        "Select features (X) for the Random Forest model:",
        options=list(rf_label_to_col.keys()),
        default=rf_default_labels,
        key="rf_features",
        help="The model will be retrained from scratch on whichever features you choose.",
    )
    rf_selected_features = [rf_label_to_col[l] for l in rf_selected_labels]

    st.divider()

    if not rf_selected_features:
        st.warning("⚠️ Please select at least one feature to train the model.")
    else:
        from sklearn.ensemble import RandomForestRegressor

        @st.cache_data(show_spinner=False)
        def train_rf(feature_tuple):
            features = list(feature_tuple)
            _df = df[features + ["resale_price"]].dropna()
            X_ = _df[features]
            y_ = _df["resale_price"]
            X_tr_, X_te_, y_tr_, y_te_ = train_test_split(X_, y_, test_size=0.10, random_state=42)
            model_ = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )
            with st.spinner("Training Random Forest on selected features… this may take ~10–20 seconds."):
                model_.fit(X_tr_, y_tr_)
            return model_, X_tr_, X_te_, y_tr_, y_te_

        live_rf, X_rf_tr, X_rf_te, y_rf_train, y_rf_test = train_rf(tuple(rf_selected_features))

        yp_rf_train = live_rf.predict(X_rf_tr)
        yp_rf_test  = live_rf.predict(X_rf_te)

        rf_r2_tr    = r2_score(y_rf_train, yp_rf_train)
        rf_r2_te    = r2_score(y_rf_test,  yp_rf_test)
        rf_mae_te   = mean_absolute_error(y_rf_test, yp_rf_test)
        rf_rmse_te  = np.sqrt(mean_squared_error(y_rf_test, yp_rf_test))
        rf_null     = np.full(len(y_rf_test), y_rf_train.mean())
        rf_null_rmse = np.sqrt(mean_squared_error(y_rf_test, rf_null))

        # ── Metric cards ──
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card">
            <div class="metric-card-label">Train R²</div>
            <div class="metric-card-value">{rf_r2_tr:.3f}</div>
            <div class="metric-card-sub">Variance explained (train)</div>
          </div>
          <div class="metric-card">
            <div class="metric-card-label">Test R²</div>
            <div class="metric-card-value">{rf_r2_te:.3f}</div>
            <div class="metric-card-sub">Variance explained (test)</div>
          </div>
          <div class="metric-card">
            <div class="metric-card-label">Test MAE</div>
            <div class="metric-card-value">S${rf_mae_te:,.0f}</div>
            <div class="metric-card-sub">Mean absolute error</div>
          </div>
          <div class="metric-card">
            <div class="metric-card-label">Test RMSE</div>
            <div class="metric-card-value">S${rf_rmse_te:,.0f}</div>
            <div class="metric-card-sub">vs null RMSE S${rf_null_rmse:,.0f}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_rfa, col_rfb = st.columns(2)

        with col_rfa:
            st.markdown('<p class="section-header">Actual vs Predicted</p>', unsafe_allow_html=True)
            fig_rfa, ax_rfa = plt.subplots(figsize=(5.5, 4.5))
            ax_rfa.scatter(y_rf_test / 1e3, yp_rf_test / 1e3, alpha=0.25, s=8, color="#3d8a4a")
            lims_rf = [min(y_rf_test.min(), yp_rf_test.min()) / 1e3,
                       max(y_rf_test.max(), yp_rf_test.max()) / 1e3]
            ax_rfa.plot(lims_rf, lims_rf, color="#c0392b", lw=1.5, ls="--", label="Perfect fit")
            ax_rfa.set_xlabel("Actual Price (S$'000)", fontsize=10)
            ax_rfa.set_ylabel("Predicted Price (S$'000)", fontsize=10)
            ax_rfa.set_title(f"R² (test) = {rf_r2_te:.3f}", fontsize=11)
            ax_rfa.legend(fontsize=9)
            fig_rfa.patch.set_facecolor("#f5f7f2")
            ax_rfa.set_facecolor("#f5f7f2")
            st.pyplot(fig_rfa)
            plt.close(fig_rfa)

        with col_rfb:
            st.markdown('<p class="section-header">Residuals Distribution</p>', unsafe_allow_html=True)
            rf_residuals = (y_rf_test.values - yp_rf_test) / 1e3
            fig_rfb, ax_rfb = plt.subplots(figsize=(5.5, 4.5))
            ax_rfb.hist(rf_residuals, bins=60, color="#3d8a4a", edgecolor="white", alpha=0.85)
            ax_rfb.axvline(0, color="#c0392b", lw=1.8, ls="--", label="Zero error")
            ax_rfb.set_xlabel("Residual (S$'000)", fontsize=10)
            ax_rfb.set_ylabel("Count", fontsize=10)
            ax_rfb.set_title("Distribution of Prediction Errors", fontsize=11)
            ax_rfb.legend(fontsize=9)
            fig_rfb.patch.set_facecolor("#f5f7f2")
            ax_rfb.set_facecolor("#f5f7f2")
            st.pyplot(fig_rfb)
            plt.close(fig_rfb)

        # ── Feature Importance ──
        st.markdown('<p class="section-header">Feature Importance</p>', unsafe_allow_html=True)
        fi_rf = pd.DataFrame({
            "Feature":    [FEATURE_LABELS.get(f, f) for f in rf_selected_features],
            "Importance": live_rf.feature_importances_,
        }).sort_values("Importance", ascending=True)

        fig_rffi, ax_rffi = plt.subplots(figsize=(8, max(3, len(fi_rf) * 0.55)))
        bars_rf = ax_rffi.barh(fi_rf["Feature"], fi_rf["Importance"],
                               color="#3d8a4a", edgecolor="white", alpha=0.88)
        for bar, val in zip(bars_rf, fi_rf["Importance"]):
            ax_rffi.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                         f"{val:.3f}", va="center", fontsize=9, color="#2d5a27")
        ax_rffi.set_xlabel("Importance Score", fontsize=10)
        ax_rffi.set_title("Random Forest Feature Importance", fontsize=11)
        ax_rffi.set_xlim(0, fi_rf["Importance"].max() * 1.18)
        fig_rffi.patch.set_facecolor("#f5f7f2")
        ax_rffi.set_facecolor("#f5f7f2")
        fig_rffi.tight_layout()
        st.pyplot(fig_rffi)
        plt.close(fig_rffi)

        # ── Sample predictions ──
        st.markdown('<p class="section-header">Sample Predictions (first 10 test rows)</p>', unsafe_allow_html=True)
        rf_sample = X_rf_te.head(10).copy()
        rf_sample.columns = [FEATURE_LABELS.get(c, c) for c in rf_sample.columns]
        rf_sample["Actual (S$)"]    = y_rf_test.head(10).values
        rf_sample["Predicted (S$)"] = yp_rf_test[:10]
        rf_sample["Error (S$)"]     = rf_sample["Actual (S$)"] - rf_sample["Predicted (S$)"]
        rf_sample.index = range(1, 11)
        for col in ["Actual (S$)", "Predicted (S$)", "Error (S$)"]:
            rf_sample[col] = rf_sample[col].apply(lambda x: f"S${x:,.0f}")
        st.dataframe(rf_sample, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — MODEL EVALUATION (XGBoost, live retrain)
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">XGBoost — Model Evaluation</p>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">Select features below — a fresh XGBoost model will be trained '
                'on your selection using a fixed <strong>90% train / 10% test</strong> split. '
                'Training may take a few seconds.</div>', unsafe_allow_html=True)

    # ── Feature multiselect ──
    xgb_label_to_col = {FEATURE_LABELS.get(c, c): c for c in AVAILABLE_FEATURES}
    xgb_default_labels = [FEATURE_LABELS.get(c, c) for c in AVAILABLE_FEATURES]

    xgb_selected_labels = st.multiselect(
        "Select features (X) for the XGBoost model:",
        options=list(xgb_label_to_col.keys()),
        default=xgb_default_labels,
        key="xgb_features",
        help="The model will be retrained from scratch on whichever features you choose.",
    )
    xgb_selected_features = [xgb_label_to_col[l] for l in xgb_selected_labels]

    st.divider()

    if not xgb_selected_features:
        st.warning("⚠️ Please select at least one feature to train the model.")
    else:
        # Build dataset
        needed_xgb = xgb_selected_features + ["resale_price"]
        df_xgb = df[needed_xgb].dropna()

        X_xgb = df_xgb[xgb_selected_features]
        y_xgb = df_xgb["resale_price"]

        X_xgb_train, X_xgb_test, y_xgb_train, y_xgb_test = train_test_split(
            X_xgb, y_xgb, test_size=0.10, random_state=42
        )

        # ── Train a fresh XGBoost on the selected features ──
        import xgboost as xgb

        @st.cache_data(show_spinner=False)
        def train_xgb(feature_tuple):
            features = list(feature_tuple)
            needed = features + ["resale_price"]
            _df = df[needed].dropna()
            X_ = _df[features]
            y_ = _df["resale_price"]
            X_tr_, X_te_, y_tr_, y_te_ = train_test_split(X_, y_, test_size=0.10, random_state=42)
            model_ = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                verbosity=0,
            )
            with st.spinner("Training XGBoost on selected features… this may take ~5–10 seconds."):
                model_.fit(X_tr_, y_tr_)
            return model_, X_tr_, X_te_, y_tr_, y_te_

        live_xgb, X_tr, X_te, y_xgb_train, y_xgb_test = train_xgb(
            tuple(xgb_selected_features)
        )

        yp_xgb_train = live_xgb.predict(X_tr)
        yp_xgb_test  = live_xgb.predict(X_te)

        # Metrics
        xr2_tr    = r2_score(y_xgb_train, yp_xgb_train)
        xr2_te    = r2_score(y_xgb_test,  yp_xgb_test)
        xmae_te   = mean_absolute_error(y_xgb_test, yp_xgb_test)
        xrmse_te  = np.sqrt(mean_squared_error(y_xgb_test, yp_xgb_test))
        null_xgb  = np.full(len(y_xgb_test), y_xgb_train.mean())
        xnull_rmse = np.sqrt(mean_squared_error(y_xgb_test, null_xgb))

        # ── Metric cards ──
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card">
            <div class="metric-card-label">Train R²</div>
            <div class="metric-card-value">{xr2_tr:.3f}</div>
            <div class="metric-card-sub">Variance explained (train)</div>
          </div>
          <div class="metric-card">
            <div class="metric-card-label">Test R²</div>
            <div class="metric-card-value">{xr2_te:.3f}</div>
            <div class="metric-card-sub">Variance explained (test)</div>
          </div>
          <div class="metric-card">
            <div class="metric-card-label">Test MAE</div>
            <div class="metric-card-value">S${xmae_te:,.0f}</div>
            <div class="metric-card-sub">Mean absolute error</div>
          </div>
          <div class="metric-card">
            <div class="metric-card-label">Test RMSE</div>
            <div class="metric-card-value">S${xrmse_te:,.0f}</div>
            <div class="metric-card-sub">vs null RMSE S${xnull_rmse:,.0f}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_xa, col_xb = st.columns(2)

        # ── Actual vs Predicted scatter ──
        with col_xa:
            st.markdown('<p class="section-header">Actual vs Predicted</p>', unsafe_allow_html=True)
            fig_xa, ax_xa = plt.subplots(figsize=(5.5, 4.5))
            ax_xa.scatter(y_xgb_test / 1e3, yp_xgb_test / 1e3,
                          alpha=0.25, s=8, color="#3d8a4a")
            lims_x = [min(y_xgb_test.min(), yp_xgb_test.min()) / 1e3,
                      max(y_xgb_test.max(), yp_xgb_test.max()) / 1e3]
            ax_xa.plot(lims_x, lims_x, color="#c0392b", lw=1.5, ls="--", label="Perfect fit")
            ax_xa.set_xlabel("Actual Price (S$'000)", fontsize=10)
            ax_xa.set_ylabel("Predicted Price (S$'000)", fontsize=10)
            ax_xa.set_title(f"R² (test) = {xr2_te:.3f}", fontsize=11)
            ax_xa.legend(fontsize=9)
            fig_xa.patch.set_facecolor("#f5f7f2")
            ax_xa.set_facecolor("#f5f7f2")
            st.pyplot(fig_xa)
            plt.close(fig_xa)

        # ── Residuals distribution ──
        with col_xb:
            st.markdown('<p class="section-header">Residuals Distribution</p>', unsafe_allow_html=True)
            xresiduals = (y_xgb_test.values - yp_xgb_test) / 1e3
            fig_xb, ax_xb = plt.subplots(figsize=(5.5, 4.5))
            ax_xb.hist(xresiduals, bins=60, color="#3d8a4a", edgecolor="white", alpha=0.85)
            ax_xb.axvline(0, color="#c0392b", lw=1.8, ls="--", label="Zero error")
            ax_xb.set_xlabel("Residual (S$'000)", fontsize=10)
            ax_xb.set_ylabel("Count", fontsize=10)
            ax_xb.set_title("Distribution of Prediction Errors", fontsize=11)
            ax_xb.legend(fontsize=9)
            fig_xb.patch.set_facecolor("#f5f7f2")
            ax_xb.set_facecolor("#f5f7f2")
            st.pyplot(fig_xb)
            plt.close(fig_xb)

        # ── Feature Importance chart ──
        st.markdown('<p class="section-header">Feature Importance</p>', unsafe_allow_html=True)

        fi_df = pd.DataFrame({
            "Feature":    [FEATURE_LABELS.get(f, f) for f in xgb_selected_features],
            "Importance": live_xgb.feature_importances_,
        }).sort_values("Importance", ascending=True)

        fig_fi, ax_fi = plt.subplots(figsize=(8, max(3, len(fi_df) * 0.55)))
        bars = ax_fi.barh(fi_df["Feature"], fi_df["Importance"],
                          color="#3d8a4a", edgecolor="white", alpha=0.88)
        for bar, val in zip(bars, fi_df["Importance"]):
            ax_fi.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                       f"{val:.3f}", va="center", fontsize=9, color="#2d5a27")
        ax_fi.set_xlabel("Importance Score", fontsize=10)
        ax_fi.set_title("XGBoost Feature Importance (F-score)", fontsize=11)
        ax_fi.set_xlim(0, fi_df["Importance"].max() * 1.18)
        fig_fi.patch.set_facecolor("#f5f7f2")
        ax_fi.set_facecolor("#f5f7f2")
        fig_fi.tight_layout()
        st.pyplot(fig_fi)
        plt.close(fig_fi)

        # ── Sample predictions ──
        st.markdown('<p class="section-header">Sample Predictions (first 10 test rows)</p>',
                    unsafe_allow_html=True)
        xsample = X_te.head(10).copy()
        xsample.columns = [FEATURE_LABELS.get(c, c) for c in xsample.columns]
        xsample["Actual (S$)"]    = y_xgb_test.head(10).values
        xsample["Predicted (S$)"] = yp_xgb_test[:10]
        xsample["Error (S$)"]     = xsample["Actual (S$)"] - xsample["Predicted (S$)"]
        xsample.index = range(1, 11)
        for col in ["Actual (S$)", "Predicted (S$)", "Error (S$)"]:
            xsample[col] = xsample[col].apply(lambda x: f"S${x:,.0f}")
        st.dataframe(xsample, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 4 — LIVE PREDICTION
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">Get an Instant Resale Price Estimate</p>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">Enter flat details below and click <strong>Estimate Price</strong> '
                'to get a data-backed valuation.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Flat Details**")
        town       = st.selectbox("Town", TOWNS, key="p_town")
        flat_type  = st.selectbox("Flat Type", FLAT_TYPES,
                                   index=FLAT_TYPES.index("4 ROOM") if "4 ROOM" in FLAT_TYPES else 0,
                                   key="p_flat_type")
        hdb_age    = st.slider("HDB Age (years)", 0, 60, 20, key="p_hdb_age")
        storey_mid = st.slider("Storey (mid-level)", 1, 50, 8, key="p_storey")

    with col2:
        st.markdown("**Nearby Amenities (count)**")
        hawker_1km = st.slider("Hawker Centres Within 1 km", 0, 5, 1, key="p_hawker")
        mall_1km   = st.slider("Malls Within 1 km", 0, 5, 2, key="p_mall")

    with col3:
        st.markdown("**Distance to Amenities**")
        mrt_dist  = st.slider("MRT Distance (m)", 50, 3000, 500, step=50, key="p_mrt")
        pri_dist  = st.slider("Pri School Distance (m)", 50, 3000, 500, step=50, key="p_pri")
        bus_dist  = st.slider("Bus Stop Distance (m)", 10, 500, 100, step=10, key="p_bus")

    st.divider()

    if st.button("🔮  Estimate Price", use_container_width=False):

        town_enc      = TOWN_ENCODING.get(town, 0)
        flat_type_enc = FLAT_TYPE_ENCODING.get(flat_type, 0)

        pkl_features = list(xgb_model.feature_names_in_)
        input_aligned = pd.DataFrame([{
            "flat_type_encoded":        flat_type_enc,
            "town_encoded":             town_enc,
            "hdb_age":                  hdb_age,
            "mid":                      storey_mid,
            "mrt_nearest_distance":     mrt_dist,
            "pri_sch_nearest_distance": pri_dist,
            "Hawker_Within_1km":        hawker_1km,
            "Mall_Within_1km":          mall_1km,
            "bus_stop_nearest_distance":bus_dist,
        }])[pkl_features]

        predicted_price = float(xgb_model.predict(input_aligned)[0])

        st.markdown(f"""
        <div class="price-result">
          <div class="price-result-label">Estimated Resale Price</div>
          <div class="price-result-value">S${predicted_price:,.0f}</div>
          <div class="price-result-psf">{flat_type} · {town}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Comparable transactions (train + test combined) ──
        st.markdown('<p class="section-header">Comparable Transactions</p>', unsafe_allow_html=True)

        COMP_COLS = ["town", "flat_type", "mid", "hdb_age",
                     "mrt_nearest_distance", "pri_sch_nearest_distance",
                     "bus_stop_nearest_distance", "hawker_within_1km",
                     "mall_within_1km", "resale_price"]

        # Filters applied and their ranges
        FILTER_RANGES = {
            "Town":          f"= {town}",
            "Flat Type":     f"= {flat_type}",
            "HDB Age (years)": f"{hdb_age - 5} – {hdb_age + 5}",
        }

        comparables = df_all[
            (df_all["town"] == town) &
            (df_all["flat_type"] == flat_type) &
            (df_all["hdb_age"].between(hdb_age - 5, hdb_age + 5))
        ][COMP_COLS].dropna()

        hdb_age_relaxed = False
        if len(comparables) == 0:
            hdb_age_relaxed = True
            comparables = df_all[
                (df_all["town"] == town) &
                (df_all["flat_type"] == flat_type)
            ][COMP_COLS].dropna()
            note = f"(HDB age filter relaxed — {len(comparables)} transactions found in {town})"
            FILTER_RANGES.pop("HDB Age (years)")
        else:
            note = f"{len(comparables)} transactions found — {flat_type} in {town} with similar HDB age"

        # ── Filter range pills ──
        pills_html = "".join([
            f'<span style="background:#2d5a27;color:white;border-radius:6px;'
            f'padding:0.3rem 0.7rem;margin:0.2rem;display:inline-block;font-size:0.82rem;">'
            f'<strong>{k}</strong>: {v}</span>'
            for k, v in FILTER_RANGES.items()
        ])
        st.markdown(f'<div style="margin-bottom:0.8rem;">{pills_html}</div>', unsafe_allow_html=True)
        st.caption(note)

        if len(comparables) > 0:
            med = comparables["resale_price"].median()
            avg = comparables["resale_price"].mean()

            cA, cB, cC = st.columns(3)
            cA.metric("Comparable Median", f"S${med:,.0f}")
            cB.metric("Comparable Mean",   f"S${avg:,.0f}")
            cC.metric("Your Estimate vs Median",
                      f"S${predicted_price - med:+,.0f}",
                      delta_color="normal")

            fig3, ax3 = plt.subplots(figsize=(7, 3.5))
            ax3.hist(comparables["resale_price"] / 1e3, bins=40,
                     color="#3d8a4a", edgecolor="white", alpha=0.8)
            ax3.axvline(predicted_price / 1e3, color="#c0392b",
                        lw=2, ls="--", label=f"Your estimate S${predicted_price/1e3:.0f}K")
            ax3.axvline(med / 1e3, color="#f39c12",
                        lw=1.5, ls=":", label=f"Median S${med/1e3:.0f}K")
            ax3.set_xlabel("Resale Price (S$'000)", fontsize=10)
            ax3.set_ylabel("Count", fontsize=10)
            ax3.set_title("Price Distribution — Similar Flats", fontsize=11)
            ax3.legend(fontsize=9)
            fig3.patch.set_facecolor("#f5f7f2")
            ax3.set_facecolor("#f5f7f2")
            st.pyplot(fig3)
            plt.close(fig3)

            with st.expander("View comparable transactions (sample of 20)"):
                show = comparables.sample(min(20, len(comparables)), random_state=42).copy()
                show.columns = [
                    "Town", "Flat Type", "Storey Mid", "HDB Age",
                    "MRT Distance (m)", "Pri School Distance (m)",
                    "Bus Stop Distance (m)", "Hawker Within 1km",
                    "Mall Within 1km", "Resale Price (S$)"
                ]
                show["Resale Price (S$)"] = show["Resale Price (S$)"].apply(lambda x: f"S${x:,.0f}")
                show = show.sort_values("Resale Price (S$)", ascending=False).reset_index(drop=True)

                # Columns used for filtering — always first
                highlight_cols = ["Town", "Flat Type"]
                if not hdb_age_relaxed:
                    highlight_cols.append("HDB Age")

                # Reorder: filtered cols first, then the rest, resale price last
                other_cols = [c for c in show.columns
                              if c not in highlight_cols and c != "Resale Price (S$)"]
                show = show[highlight_cols + other_cols + ["Resale Price (S$)"]]

                def highlight_filter_cols(df):
                    styles = pd.DataFrame("", index=df.index, columns=df.columns)
                    for col in highlight_cols:
                        if col in df.columns:
                            styles[col] = "background-color: #d4edda; font-weight: 600;"
                    return styles

                st.dataframe(
                    show.style.apply(highlight_filter_cols, axis=None),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("No comparable transactions found for the selected inputs.")

# ── Footer ──
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.8rem;'>"
    "The Model Citizens · WOW! Real Estate Agency Data Sprint · Singapore HDB Resale Price Prediction"
    "</div>",
    unsafe_allow_html=True,
)
