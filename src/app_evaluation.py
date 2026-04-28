import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# ─────────────────────────────────────────────
# PATHS  — always resolve relative to this script (src/)
# ─────────────────────────────────────────────
DATA_DIR = Path(__file__).parent

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="WOW! Model Evaluation",
    page_icon="📊",
    layout="wide",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #f5f7f2; }
h1, h2, h3, h4 { font-family: 'Playfair Display', serif; }
.hero-banner {
    background: linear-gradient(135deg, #2d5a27 0%, #3d8a4a 50%, #5aa55e 100%);
    border-radius: 16px; padding: 2.5rem 3rem; margin-bottom: 2rem;
    color: white; position: relative; overflow: hidden;
}
.hero-banner::before {
    content: ""; position: absolute; top: -40px; right: -40px;
    width: 220px; height: 220px; background: rgba(255,255,255,0.06); border-radius: 50%;
}
.hero-title { font-family: 'Playfair Display', serif; font-size: 2.4rem; font-weight: 700; margin: 0 0 0.4rem 0; line-height: 1.2; }
.hero-sub { font-size: 1rem; opacity: 0.88; margin: 0 0 1.5rem 0; }
.hero-stats { display: flex; gap: 2.5rem; }
.hero-stat-val { font-family: 'Playfair Display', serif; font-size: 1.8rem; font-weight: 700; }
.hero-stat-lbl { font-size: 0.78rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.4rem; }
.metric-card {
    background: white; border: 1.5px solid #b8d9bc; border-radius: 12px;
    padding: 1rem 1.4rem; flex: 1; box-shadow: 0 2px 8px rgba(61,138,74,0.08);
}
.metric-card-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: #5a8060; margin-bottom: 0.3rem; }
.metric-card-value { font-family: 'Playfair Display', serif; font-size: 1.7rem; font-weight: 700; color: #2d5a27; }
.metric-card-sub { font-size: 0.78rem; color: #888; margin-top: 0.2rem; }
[data-testid="stTabs"] button { font-family: 'Inter', sans-serif; font-weight: 500; font-size: 0.95rem; border-radius: 8px 8px 0 0; padding: 0.5rem 1.2rem; }
[data-testid="stTabs"] button[aria-selected="true"] { background: linear-gradient(135deg, #2d5a27, #3d8a4a) !important; color: white !important; border-bottom: none !important; }
.section-header { color: #2d5a27; font-family: 'Playfair Display', serif; font-size: 1.3rem; margin: 1.5rem 0 0.8rem; border-bottom: 2px solid #b8d9bc; padding-bottom: 0.4rem; }
.info-box { background: #eef5ee; border-left: 4px solid #3d8a4a; border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; font-size: 0.88rem; color: #2d5a27; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def _clean(data):
    data.columns = data.columns.str.lower()
    for col in ["mall_nearest_distance", "hawker_nearest_distance",
                 "mrt_nearest_distance", "bus_stop_nearest_distance",
                 "pri_sch_nearest_distance", "sec_sch_nearest_dist"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in ["resale_price", "floor_area_sqm", "hdb_age", "mid"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    towns_sorted = sorted(data["town"].dropna().unique().tolist())
    data["town_encoded"] = data["town"].map({t: i for i, t in enumerate(towns_sorted)})
    flat_types_sorted = sorted(data["flat_type"].dropna().unique().tolist())
    data["flat_type_encoded"] = data["flat_type"].map({f: i for i, f in enumerate(flat_types_sorted)})
    return data

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    return _clean(pd.read_csv(DATA_DIR / "train_sample.csv", low_memory=False))

df = load_data()

MODEL_9_FEATURES = [
    "flat_type_encoded", "town_encoded", "hdb_age", "mid",
    "mrt_nearest_distance", "pri_sch_nearest_distance",
    "hawker_within_1km", "mall_within_1km", "bus_stop_nearest_distance",
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
# HERO BANNER
# ─────────────────────────────────────────────
n_rows       = len(df)
towns_count  = df["town"].nunique()
median_price = df["resale_price"].median()

st.markdown(f"""
<div class="hero-banner">
  <svg style="position:absolute;right:24px;top:50%;transform:translateY(-50%);opacity:0.18;"
       width="260" height="160" viewBox="0 0 260 160" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="10" y="40" width="35" height="110" rx="3" fill="white"/>
    <rect x="55" y="20" width="40" height="130" rx="3" fill="white"/>
    <rect x="105" y="55" width="30" height="95" rx="3" fill="white"/>
    <ellipse cx="160" cy="125" rx="18" ry="20" fill="#5aa55e"/><rect x="157" y="138" width="6" height="12" fill="#3d6b40"/>
    <ellipse cx="195" cy="130" rx="14" ry="16" fill="#4d9455"/><rect x="192" y="142" width="5" height="8" fill="#3d6b40"/>
    <ellipse cx="225" cy="120" rx="20" ry="22" fill="#6bb86f"/><rect x="222" y="135" width="6" height="15" fill="#3d6b40"/>
    <rect x="0" y="148" width="260" height="12" rx="2" fill="#4d9455" opacity="0.5"/>
  </svg>
  <p class="hero-title">📊 WOW! Model Evaluation</p>
  <p class="hero-sub">Compare Linear Regression, Random Forest and XGBoost · The Model Citizens</p>
  <div class="hero-stats">
    <div><div class="hero-stat-val">{n_rows:,}</div><div class="hero-stat-lbl">Transactions</div></div>
    <div><div class="hero-stat-val">{towns_count}</div><div class="hero-stat-lbl">Towns</div></div>
    <div><div class="hero-stat-val">S${median_price/1e3:.0f}K</div><div class="hero-stat-lbl">Median Price</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPER — reusable plot functions
# ─────────────────────────────────────────────
def plot_actual_vs_predicted(y_true, y_pred, r2):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(y_true / 1e3, y_pred / 1e3, alpha=0.25, s=8, color="#3d8a4a")
    lims = [min(y_true.min(), y_pred.min()) / 1e3, max(y_true.max(), y_pred.max()) / 1e3]
    ax.plot(lims, lims, color="#c0392b", lw=1.5, ls="--", label="Perfect fit")
    ax.set_xlabel("Actual Price (S$'000)", fontsize=10)
    ax.set_ylabel("Predicted Price (S$'000)", fontsize=10)
    ax.set_title(f"R² (test) = {r2:.3f}", fontsize=11)
    ax.legend(fontsize=9)
    fig.patch.set_facecolor("#f5f7f2")
    ax.set_facecolor("#f5f7f2")
    return fig

def plot_residuals(y_true, y_pred):
    residuals = (np.array(y_true) - np.array(y_pred)) / 1e3
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.hist(residuals, bins=60, color="#3d8a4a", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="#c0392b", lw=1.8, ls="--", label="Zero error")
    ax.set_xlabel("Residual (S$'000)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Distribution of Prediction Errors", fontsize=11)
    ax.legend(fontsize=9)
    fig.patch.set_facecolor("#f5f7f2")
    ax.set_facecolor("#f5f7f2")
    return fig

def plot_feature_importance(features, importances, title):
    fi = pd.DataFrame({"Feature": [FEATURE_LABELS.get(f, f) for f in features],
                        "Importance": importances}).sort_values("Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(fi) * 0.55)))
    bars = ax.barh(fi["Feature"], fi["Importance"], color="#3d8a4a", edgecolor="white", alpha=0.88)
    for bar, val in zip(bars, fi["Importance"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color="#2d5a27")
    ax.set_xlabel("Importance Score", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, fi["Importance"].max() * 1.18)
    fig.patch.set_facecolor("#f5f7f2")
    ax.set_facecolor("#f5f7f2")
    fig.tight_layout()
    return fig

def show_metric_cards(r2_tr, r2_te, mae_te, rmse_te, null_rmse):
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

def show_sample_predictions(X_te, y_te, y_pred):
    sample = X_te.head(10).copy()
    sample.columns = [FEATURE_LABELS.get(c, c) for c in sample.columns]
    sample["Actual (S$)"]    = y_te.head(10).values
    sample["Predicted (S$)"] = y_pred[:10]
    sample["Error (S$)"]     = sample["Actual (S$)"] - sample["Predicted (S$)"]
    sample.index = range(1, 11)
    for col in ["Actual (S$)", "Predicted (S$)", "Error (S$)"]:
        sample[col] = sample[col].apply(lambda x: f"S${x:,.0f}")
    st.dataframe(sample, use_container_width=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊  Linear Regression", "🌲  Random Forest", "📈  XGBoost"])


# ══════════════════════════════════════════════
# TAB 1 — LINEAR REGRESSION
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">Linear Regression — Model Evaluation</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Evaluated on a fixed <strong>90% train / 10% test</strong> split. Select which features to include.</div>', unsafe_allow_html=True)

    label_to_col  = {FEATURE_LABELS.get(c, c): c for c in AVAILABLE_FEATURES}
    selected_labels = st.multiselect(
        "Select features (X) for the Linear Regression model:",
        options=list(label_to_col.keys()),
        default=list(label_to_col.keys()),
        help="Choose which features to train the model on. At least one is required.",
    )
    selected_features = [label_to_col[l] for l in selected_labels]

    st.divider()

    if not selected_features:
        st.warning("⚠️ Please select at least one feature to run the model.")
    else:
        _df = df[selected_features + ["resale_price"]].dropna()
        X_tr, X_te, y_tr, y_te = train_test_split(_df[selected_features], _df["resale_price"],
                                                    test_size=0.10, random_state=42)
        lr = LinearRegression()
        lr.fit(X_tr, y_tr)
        yp_tr = lr.predict(X_tr)
        yp_te = lr.predict(X_te)

        r2_tr   = r2_score(y_tr, yp_tr)
        r2_te   = r2_score(y_te, yp_te)
        mae_te  = mean_absolute_error(y_te, yp_te)
        rmse_te = np.sqrt(mean_squared_error(y_te, yp_te))
        null_rmse = np.sqrt(mean_squared_error(y_te, np.full(len(y_te), y_tr.mean())))

        show_metric_cards(r2_tr, r2_te, mae_te, rmse_te, null_rmse)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<p class="section-header">Actual vs Predicted</p>', unsafe_allow_html=True)
            fig = plot_actual_vs_predicted(y_te, yp_te, r2_te)
            st.pyplot(fig); plt.close(fig)
        with col_b:
            st.markdown('<p class="section-header">Residuals Distribution</p>', unsafe_allow_html=True)
            fig = plot_residuals(y_te, yp_te)
            st.pyplot(fig); plt.close(fig)

        st.markdown('<p class="section-header">Model Coefficients</p>', unsafe_allow_html=True)
        coef_df = pd.DataFrame({
            "Feature":     [FEATURE_LABELS.get(f, f) for f in selected_features],
            "Coefficient": lr.coef_,
        }).sort_values("Coefficient", key=abs, ascending=False)
        coef_df["Coefficient"] = coef_df["Coefficient"].apply(
            lambda x: f"{'+ ' if x > 0 else '− '}S${abs(x):,.2f}")
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

        st.markdown('<p class="section-header">Sample Predictions (first 10 test rows)</p>', unsafe_allow_html=True)
        show_sample_predictions(X_te, y_te, yp_te)


# ══════════════════════════════════════════════
# TAB 2 — RANDOM FOREST
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">Random Forest — Model Evaluation</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">A fresh Random Forest model will be trained on your selection using a fixed <strong>90% train / 10% test</strong> split. Training may take ~15–30 seconds.</div>', unsafe_allow_html=True)

    rf_label_to_col = {FEATURE_LABELS.get(c, c): c for c in AVAILABLE_FEATURES}
    rf_selected_labels = st.multiselect(
        "Select features (X) for the Random Forest model:",
        options=list(rf_label_to_col.keys()),
        default=list(rf_label_to_col.keys()),
        key="rf_features",
        help="The model will be retrained from scratch on whichever features you choose.",
    )
    rf_selected_features = [rf_label_to_col[l] for l in rf_selected_labels]

    st.divider()

    if not rf_selected_features:
        st.warning("⚠️ Please select at least one feature to train the model.")
    else:
        @st.cache_data(show_spinner=False)
        def train_rf(feature_tuple):
            feats = list(feature_tuple)
            _d = df[feats + ["resale_price"]].dropna()
            X_tr_, X_te_, y_tr_, y_te_ = train_test_split(_d[feats], _d["resale_price"],
                                                            test_size=0.10, random_state=42)
            m = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            with st.spinner("Training Random Forest… this may take ~10–20 seconds."):
                m.fit(X_tr_, y_tr_)
            return m, X_tr_, X_te_, y_tr_, y_te_

        live_rf, X_rf_tr, X_rf_te, y_rf_tr, y_rf_te = train_rf(tuple(rf_selected_features))
        yp_rf_tr = live_rf.predict(X_rf_tr)
        yp_rf_te = live_rf.predict(X_rf_te)

        show_metric_cards(
            r2_score(y_rf_tr, yp_rf_tr),
            r2_score(y_rf_te, yp_rf_te),
            mean_absolute_error(y_rf_te, yp_rf_te),
            np.sqrt(mean_squared_error(y_rf_te, yp_rf_te)),
            np.sqrt(mean_squared_error(y_rf_te, np.full(len(y_rf_te), y_rf_tr.mean()))),
        )

        col_rfa, col_rfb = st.columns(2)
        with col_rfa:
            st.markdown('<p class="section-header">Actual vs Predicted</p>', unsafe_allow_html=True)
            fig = plot_actual_vs_predicted(y_rf_te, yp_rf_te, r2_score(y_rf_te, yp_rf_te))
            st.pyplot(fig); plt.close(fig)
        with col_rfb:
            st.markdown('<p class="section-header">Residuals Distribution</p>', unsafe_allow_html=True)
            fig = plot_residuals(y_rf_te, yp_rf_te)
            st.pyplot(fig); plt.close(fig)

        st.markdown('<p class="section-header">Feature Importance</p>', unsafe_allow_html=True)
        fig = plot_feature_importance(rf_selected_features, live_rf.feature_importances_,
                                      "Random Forest Feature Importance")
        st.pyplot(fig); plt.close(fig)

        st.markdown('<p class="section-header">Sample Predictions (first 10 test rows)</p>', unsafe_allow_html=True)
        show_sample_predictions(X_rf_te, y_rf_te, yp_rf_te)


# ══════════════════════════════════════════════
# TAB 3 — XGBOOST
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">XGBoost — Model Evaluation</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">A fresh XGBoost model will be trained on your selection using a fixed <strong>90% train / 10% test</strong> split. Training may take ~10–20 seconds.</div>', unsafe_allow_html=True)

    xgb_label_to_col = {FEATURE_LABELS.get(c, c): c for c in AVAILABLE_FEATURES}
    xgb_selected_labels = st.multiselect(
        "Select features (X) for the XGBoost model:",
        options=list(xgb_label_to_col.keys()),
        default=list(xgb_label_to_col.keys()),
        key="xgb_features",
        help="The model will be retrained from scratch on whichever features you choose.",
    )
    xgb_selected_features = [xgb_label_to_col[l] for l in xgb_selected_labels]

    st.divider()

    if not xgb_selected_features:
        st.warning("⚠️ Please select at least one feature to train the model.")
    else:
        @st.cache_data(show_spinner=False)
        def train_xgb(feature_tuple):
            feats = list(feature_tuple)
            _d = df[feats + ["resale_price"]].dropna()
            X_tr_, X_te_, y_tr_, y_te_ = train_test_split(_d[feats], _d["resale_price"],
                                                            test_size=0.10, random_state=42)
            m = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6,
                                  subsample=0.8, colsample_bytree=0.8,
                                  objective="reg:squarederror", random_state=42, verbosity=0)
            with st.spinner("Training XGBoost… this may take ~5–10 seconds."):
                m.fit(X_tr_, y_tr_)
            return m, X_tr_, X_te_, y_tr_, y_te_

        live_xgb, X_xgb_tr, X_xgb_te, y_xgb_tr, y_xgb_te = train_xgb(tuple(xgb_selected_features))
        yp_xgb_tr = live_xgb.predict(X_xgb_tr)
        yp_xgb_te = live_xgb.predict(X_xgb_te)

        show_metric_cards(
            r2_score(y_xgb_tr, yp_xgb_tr),
            r2_score(y_xgb_te, yp_xgb_te),
            mean_absolute_error(y_xgb_te, yp_xgb_te),
            np.sqrt(mean_squared_error(y_xgb_te, yp_xgb_te)),
            np.sqrt(mean_squared_error(y_xgb_te, np.full(len(y_xgb_te), y_xgb_tr.mean()))),
        )

        col_xa, col_xb = st.columns(2)
        with col_xa:
            st.markdown('<p class="section-header">Actual vs Predicted</p>', unsafe_allow_html=True)
            fig = plot_actual_vs_predicted(y_xgb_te, yp_xgb_te, r2_score(y_xgb_te, yp_xgb_te))
            st.pyplot(fig); plt.close(fig)
        with col_xb:
            st.markdown('<p class="section-header">Residuals Distribution</p>', unsafe_allow_html=True)
            fig = plot_residuals(y_xgb_te, yp_xgb_te)
            st.pyplot(fig); plt.close(fig)

        st.markdown('<p class="section-header">Feature Importance</p>', unsafe_allow_html=True)
        fig = plot_feature_importance(xgb_selected_features, live_xgb.feature_importances_,
                                      "XGBoost Feature Importance (F-score)")
        st.pyplot(fig); plt.close(fig)

        st.markdown('<p class="section-header">Sample Predictions (first 10 test rows)</p>', unsafe_allow_html=True)
        show_sample_predictions(X_xgb_te, y_xgb_te, yp_xgb_te)


# ── Footer ──
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.8rem;'>"
    "The Model Citizens · WOW! Real Estate Agency Data Sprint · Singapore HDB Resale Price Prediction"
    "</div>",
    unsafe_allow_html=True,
)
