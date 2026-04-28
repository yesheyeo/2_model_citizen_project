import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# PATHS  — always resolve relative to this script (src/)
# ─────────────────────────────────────────────
DATA_DIR = Path(__file__).parent

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="WOW! Price Predictor",
    page_icon="🔮",
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
.stButton > button {
    background: linear-gradient(135deg, #2d5a27, #3d8a4a); color: white; border: none;
    border-radius: 8px; font-family: 'Inter', sans-serif; font-weight: 600;
    padding: 0.6rem 2rem; transition: transform 0.15s;
}
.stButton > button:hover { transform: translateY(-2px); }
.price-result {
    background: linear-gradient(135deg, #2d5a27, #3d8a4a); color: white;
    border-radius: 14px; padding: 1.8rem 2rem; text-align: center; margin-top: 1.5rem;
}
.price-result-label { font-size: 0.85rem; opacity: 0.85; text-transform: uppercase; letter-spacing: 0.06em; }
.price-result-value { font-family: 'Playfair Display', serif; font-size: 2.8rem; font-weight: 700; margin: 0.3rem 0; }
.price-result-psf { font-size: 0.9rem; opacity: 0.82; }
.section-header {
    color: #2d5a27; font-family: 'Playfair Display', serif; font-size: 1.3rem;
    margin: 1.5rem 0 0.8rem; border-bottom: 2px solid #b8d9bc; padding-bottom: 0.4rem;
}
.info-box {
    background: #eef5ee; border-left: 4px solid #3d8a4a; border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem; font-size: 0.88rem; color: #2d5a27; margin-bottom: 1rem;
}
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

@st.cache_data(show_spinner="Loading full dataset for comparables…")
def load_all_data():
    train = _clean(pd.read_csv(DATA_DIR / "train_sample.csv", low_memory=False))
    test  = _clean(pd.read_csv(DATA_DIR / "test_sample.csv",  low_memory=False))
    return pd.concat([train, test], ignore_index=True)

@st.cache_resource(show_spinner="Loading prediction model…")
def load_model():
    with open(DATA_DIR / "model.pkl", "rb") as f:
        return pickle.load(f)


df        = load_data()
df_all    = load_all_data()
xgb_model = load_model()

TOWNS      = sorted(df["town"].dropna().unique().tolist())
FLAT_TYPES = sorted(df["flat_type"].dropna().unique().tolist())
TOWN_ENCODING      = {t: i for i, t in enumerate(sorted(TOWNS))}
FLAT_TYPE_ENCODING = {f: i for i, f in enumerate(sorted(FLAT_TYPES))}

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
    <rect x="14" y="50" width="6" height="8" rx="1" fill="#cce8d0"/><rect x="23" y="50" width="6" height="8" rx="1" fill="#cce8d0"/><rect x="32" y="50" width="6" height="8" rx="1" fill="#cce8d0"/>
    <rect x="14" y="65" width="6" height="8" rx="1" fill="#cce8d0"/><rect x="23" y="65" width="6" height="8" rx="1" fill="#cce8d0"/><rect x="32" y="65" width="6" height="8" rx="1" fill="#cce8d0"/>
    <rect x="55" y="20" width="40" height="130" rx="3" fill="white"/>
    <rect x="60" y="30" width="7" height="9" rx="1" fill="#cce8d0"/><rect x="71" y="30" width="7" height="9" rx="1" fill="#cce8d0"/><rect x="82" y="30" width="7" height="9" rx="1" fill="#cce8d0"/>
    <rect x="105" y="55" width="30" height="95" rx="3" fill="white"/>
    <ellipse cx="160" cy="125" rx="18" ry="20" fill="#5aa55e"/><rect x="157" y="138" width="6" height="12" fill="#3d6b40"/>
    <ellipse cx="195" cy="130" rx="14" ry="16" fill="#4d9455"/><rect x="192" y="142" width="5" height="8" fill="#3d6b40"/>
    <ellipse cx="225" cy="120" rx="20" ry="22" fill="#6bb86f"/><rect x="222" y="135" width="6" height="15" fill="#3d6b40"/>
    <rect x="0" y="148" width="260" height="12" rx="2" fill="#4d9455" opacity="0.5"/>
  </svg>
  <p class="hero-title">🔮 WOW! Price Predictor</p>
  <p class="hero-sub">Instant HDB resale price estimates for WOW! Real Estate agents · The Model Citizens</p>
  <div class="hero-stats">
    <div><div class="hero-stat-val">{n_rows:,}</div><div class="hero-stat-lbl">Transactions</div></div>
    <div><div class="hero-stat-val">{towns_count}</div><div class="hero-stat-lbl">Towns</div></div>
    <div><div class="hero-stat-val">S${median_price/1e3:.0f}K</div><div class="hero-stat-lbl">Median Price</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUTS
# ─────────────────────────────────────────────
st.markdown('<div class="info-box">Enter flat details below and click <strong>Estimate Price</strong> to get a data-backed valuation.</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Flat Details**")
    town       = st.selectbox("Town", TOWNS)
    flat_type  = st.selectbox("Flat Type", FLAT_TYPES,
                               index=FLAT_TYPES.index("4 ROOM") if "4 ROOM" in FLAT_TYPES else 0)
    hdb_age    = st.slider("HDB Age (years)", 0, 60, 20)
    storey_mid = st.slider("Storey (mid-level)", 1, 50, 8)

with col2:
    st.markdown("**Nearby Amenities (count)**")
    hawker_1km = st.slider("Hawker Centres Within 1 km", 0, 5, 1)
    mall_1km   = st.slider("Malls Within 1 km", 0, 5, 2)

with col3:
    st.markdown("**Distance to Amenities**")
    mrt_dist = st.slider("MRT Distance (m)", 50, 3000, 500, step=50)
    pri_dist = st.slider("Pri School Distance (m)", 50, 3000, 500, step=50)
    bus_dist = st.slider("Bus Stop Distance (m)", 10, 500, 100, step=10)

st.divider()

if st.button("🔮  Estimate Price"):
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

    # ── Comparable transactions ──
    st.markdown('<p class="section-header">Comparable Transactions</p>', unsafe_allow_html=True)

    COMP_COLS = ["town", "flat_type", "mid", "hdb_age",
                 "mrt_nearest_distance", "pri_sch_nearest_distance",
                 "bus_stop_nearest_distance", "hawker_within_1km",
                 "mall_within_1km", "resale_price"]

    FILTER_RANGES = {
        "Town":            f"= {town}",
        "Flat Type":       f"= {flat_type}",
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
        cC.metric("Your Estimate vs Median", f"S${predicted_price - med:+,.0f}", delta_color="normal")

        fig3, ax3 = plt.subplots(figsize=(7, 3.5))
        ax3.hist(comparables["resale_price"] / 1e3, bins=40, color="#3d8a4a", edgecolor="white", alpha=0.8)
        ax3.axvline(predicted_price / 1e3, color="#c0392b", lw=2, ls="--", label=f"Your estimate S${predicted_price/1e3:.0f}K")
        ax3.axvline(med / 1e3, color="#f39c12", lw=1.5, ls=":", label=f"Median S${med/1e3:.0f}K")
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
            show.columns = ["Town", "Flat Type", "Storey Mid", "HDB Age",
                            "MRT Distance (m)", "Pri School Distance (m)",
                            "Bus Stop Distance (m)", "Hawker Within 1km",
                            "Mall Within 1km", "Resale Price (S$)"]
            show["Resale Price (S$)"] = show["Resale Price (S$)"].apply(lambda x: f"S${x:,.0f}")
            show = show.sort_values("Resale Price (S$)", ascending=False).reset_index(drop=True)

            highlight_cols = ["Town", "Flat Type"]
            if not hdb_age_relaxed:
                highlight_cols.append("HDB Age")

            other_cols = [c for c in show.columns if c not in highlight_cols and c != "Resale Price (S$)"]
            show = show[highlight_cols + other_cols + ["Resale Price (S$)"]]

            def highlight_filter_cols(df):
                styles = pd.DataFrame("", index=df.index, columns=df.columns)
                for col in highlight_cols:
                    if col in df.columns:
                        styles[col] = "background-color: #d4edda; font-weight: 600;"
                return styles

            st.dataframe(show.style.apply(highlight_filter_cols, axis=None),
                         use_container_width=True, hide_index=True)
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
