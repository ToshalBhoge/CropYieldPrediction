import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS — Farm / Earthy Theme
# =====================================================

st.markdown("""
<style>

    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Nunito', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ---------- Page background ---------- */

    .stApp {
        background: linear-gradient(180deg, #f9f6ee 0%, #f3ecd9 100%);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* ---------- Header banner ---------- */

    .farm-header {
        background: linear-gradient(120deg, #2f5233 0%, #4a7c3c 55%, #7a9c3f 100%);
        border-radius: 20px;
        padding: 34px 40px;
        color: #fdf9ec;
        margin-bottom: 24px;
        box-shadow: 0 10px 26px rgba(47, 82, 51, 0.25);
        position: relative;
        overflow: hidden;
    }

    .farm-eyebrow {
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        color: #d8e8b8;
        margin-bottom: 8px;
    }

    .farm-title {
        font-size: 32px;
        font-weight: 900;
        margin-bottom: 6px;
        letter-spacing: -0.3px;
    }

    .farm-subtitle {
        font-size: 15px;
        color: #eaf2d8;
        font-weight: 600;
    }

    /* ---------- Section headers ---------- */

    .farm-section {
        background: #fffdf6;
        padding: 16px 22px;
        border-radius: 14px;
        margin-top: 26px;
        margin-bottom: 14px;
        border: 1px solid #e8dfc2;
        border-left: 5px solid #7a9c3f;
    }

    .farm-section-title {
        font-size: 17px;
        font-weight: 800;
        color: #2f5233;
    }

    .farm-section-desc {
        font-size: 12.5px;
        color: #8a7d5c;
        margin-top: 2px;
    }

    /* ---------- Field cards / columns ---------- */

    div[data-testid="column"] {
        background: #fffdf6;
        border: 1px solid #ecE3c8;
        border-radius: 14px;
        padding: 6px 16px 14px 16px;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background-color: #f3ecd9;
        border-right: 1px solid #e2d6ad;
    }

    section[data-testid="stSidebar"] .stMetric {
        background: #fffdf6;
        border: 1px solid #e8dfc2;
        border-radius: 10px;
        padding: 8px 10px;
    }

    /* ---------- Labels & inputs ---------- */

    label {
        font-weight: 700 !important;
        color: #3f5b2e !important;
        font-size: 13px !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 9px !important;
        border-color: #d8cd9f !important;
    }

    /* ---------- Button ---------- */

    div.stButton > button {
        width: 100%;
        height: 52px;
        border-radius: 12px;
        border: none;
        background: linear-gradient(120deg, #4a7c3c, #7a9c3f);
        color: #fffdf6;
        font-size: 17px;
        font-weight: 800;
        letter-spacing: 0.3px;
        box-shadow: 0 6px 16px rgba(74, 124, 60, 0.28);
    }

    div.stButton > button:hover {
        opacity: 0.93;
        color: #fffdf6;
    }

    /* ---------- Prediction result ---------- */

    .yield-box {
        background: linear-gradient(135deg, #fffdf6 0%, #f5efd8 100%);
        border: 1.5px solid #d8cd9f;
        border-radius: 20px;
        padding: 32px;
        margin-top: 18px;
        text-align: center;
        box-shadow: 0 8px 22px rgba(122, 156, 63, 0.15);
    }

    .yield-label {
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #8a7d5c;
    }

    .yield-value {
        font-size: 54px;
        font-weight: 900;
        color: #2f5233;
        margin: 6px 0 2px 0;
        line-height: 1;
    }

    .yield-unit {
        font-size: 15px;
        font-weight: 700;
        color: #7a9c3f;
    }

    /* ---------- Info pills for summary ---------- */

    .field-pill {
        display: inline-block;
        background: #eaf2d8;
        color: #3f5b2e;
        font-weight: 700;
        font-size: 12.5px;
        padding: 5px 14px;
        border-radius: 20px;
        margin: 3px 4px 3px 0;
        border: 1px solid #d3e3ac;
    }

    /* ---------- Dataframe ---------- */

    div[data-testid="stDataFrame"] {
        border: 1px solid #e8dfc2;
        border-radius: 12px;
        overflow: hidden;
    }

    /* ---------- Footer ---------- */

    .farm-footer {
        text-align: center;
        color: #8a7d5c;
        font-size: 12.5px;
        padding: 26px 0 4px 0;
        margin-top: 36px;
        border-top: 1px solid #e2d6ad;
    }

</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding:6px 0 16px 0;">
            <div style="font-size:46px;">🌾</div>
            <h2 style="color:#2f5233; margin:4px 0 2px 0; font-size:20px;">
                Farm Analytics
            </h2>
            <p style="color:#8a7d5c; font-size:12.5px; margin:0;">
                Crop Yield Prediction Tool
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("##### 📌 About the Project")
    st.write(
        """
        This application predicts crop yield using a
        Machine Learning model trained on agricultural data.

        **Algorithm:** Random Forest Regressor
        **Dataset:** 500 records
        **Features:** 18
        **Target:** Crop Yield
        """
    )

    st.divider()

    st.markdown("##### 📊 Model Performance")

    m1, m2 = st.columns(2)
    with m1:
        st.metric("R² Score", "96.58%")
    with m2:
        st.metric("RMSE", "0.3067")

    st.divider()

    st.caption("Crop Yield Prediction · v1.0")
    st.caption("Python · Scikit-learn · Streamlit")

# =====================================================
# HEADER
# =====================================================

st.markdown(
    """
    <div class="farm-header">
        <div class="farm-eyebrow">Machine Learning · Agricultural Insights</div>
        <div class="farm-title">🌾 Crop Yield Prediction System</div>
        <div class="farm-subtitle">
            Estimate expected crop yield from soil, weather and farming input data.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# INPUT SECTION — FIELD DETAILS
# =====================================================

st.markdown(
    """
    <div class="farm-section">
        <div class="farm-section-title">🌱 Enter Agricultural Details</div>
        <div class="farm-section-desc">Fill in crop, soil, weather and input details for this field.</div>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

# =====================================================
# COLUMN 1
# =====================================================

with col1:

    st.markdown("**🌾 Crop & Field**")

    crop = st.selectbox(
        "Select Crop",
        [
            "Wheat",
            "Rice",
            "Maize",
            "Cotton",
            "Sugarcane",
            "Chickpea",
            "Soybean",
            "Groundnut"
        ]
    )

    area = st.number_input(
        "Area (acres)",
        min_value=0.0,
        value=10.0,
        step=1.0
    )

    st.markdown("**🌦️ Weather Conditions**")

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        value=1000.0,
        step=10.0
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        value=25.0,
        step=0.5
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=1.0
    )

# =====================================================
# COLUMN 2
# =====================================================

with col2:

    st.markdown("**🌱 Soil & Inputs**")

    soil_type = st.selectbox(
        "Select Soil Type",
        [
            "Red",
            "Black",
            "Sandy",
            "Loamy",
            "Clay"
        ]
    )

    fertilizer = st.number_input(
        "Fertilizer (kg/acre)",
        min_value=0.0,
        value=100.0,
        step=5.0
    )

    pesticide = st.number_input(
        "Pesticide (kg/acre)",
        min_value=0.0,
        value=30.0,
        step=5.0
    )

    st.markdown("**📅 Season**")

    year = st.number_input(
        "Year",
        min_value=2000,
        max_value=2035,
        value=2026,
        step=1
    )

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# PREDICTION
# =====================================================

if st.button("🌾  Predict Crop Yield", use_container_width=True):

    # Create empty input dataframe
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=feature_columns
    )

    # Numerical features
    input_data["Area"] = area
    input_data["Rainfall"] = rainfall
    input_data["Temperature"] = temperature
    input_data["Humidity"] = humidity
    input_data["Fertilizer"] = fertilizer
    input_data["Pesticide"] = pesticide
    input_data["Year"] = year

    # Crop encoding
    crop_column = "Crop_" + crop
    if crop_column in input_data.columns:
        input_data[crop_column] = 1

    # Soil encoding
    soil_column = "Soil_Type_" + soil_type
    if soil_column in input_data.columns:
        input_data[soil_column] = 1

    # Make prediction
    prediction = model.predict(input_data)[0]

    # =================================================
    # PREDICTION RESULT
    # =================================================

    st.success("✅ Prediction completed successfully!")

    st.markdown(
        f"""
        <div class="yield-box">
            <div class="yield-label">Predicted Crop Yield</div>
            <div class="yield-value">{prediction:.2f}</div>
            <div class="yield-unit">tonnes / unit area</div>
            <div style="margin-top:14px;">
                <span class="field-pill">🌾 {crop}</span>
                <span class="field-pill">🌱 {soil_type} Soil</span>
                <span class="field-pill">📅 {year}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =================================================
    # INPUT SUMMARY
    # =================================================

    st.markdown(
        """
        <div class="farm-section" style="margin-top:26px;">
            <div class="farm-section-title">📋 Prediction Details</div>
            <div class="farm-section-desc">Key parameters used to generate this prediction.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    result_col1, result_col2, result_col3, result_col4 = st.columns(4)

    with result_col1:
        st.metric("Selected Crop", crop)

    with result_col2:
        st.metric("Soil Type", soil_type)

    with result_col3:
        st.metric("Area", f"{area:.1f} acres")

    with result_col4:
        st.metric("Year", year)

    # =================================================
    # INPUT DATA TABLE
    # =================================================

    st.markdown(
        """
        <div class="farm-section" style="margin-top:10px;">
            <div class="farm-section-title">📊 Input Data</div>
            <div class="farm-section-desc">Full set of values submitted for this prediction. Search below or download as CSV.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    display_data = pd.DataFrame({
        "Parameter": [
            "Crop",
            "Area",
            "Rainfall",
            "Temperature",
            "Humidity",
            "Soil Type",
            "Fertilizer",
            "Pesticide",
            "Year",
            "Predicted Yield"
        ],
        "Value": [
            crop,
            area,
            rainfall,
            temperature,
            humidity,
            soil_type,
            fertilizer,
            pesticide,
            year,
            round(float(prediction), 2)
        ]
    })

    # -------- Search / filter --------

    search_col, download_col = st.columns([3, 1])

    with search_col:
        search_term = st.text_input(
            "🔍 Search parameter",
            placeholder="e.g. rainfall, soil, yield...",
            label_visibility="collapsed"
        )

    if search_term:
        filtered_data = display_data[
            display_data["Parameter"].str.contains(search_term, case=False)
            | display_data["Value"].astype(str).str.contains(search_term, case=False)
        ]
    else:
        filtered_data = display_data

    st.dataframe(
        filtered_data,
        use_container_width=True,
        hide_index=True
    )

    if search_term and filtered_data.empty:
        st.info("No matching parameters found.")

    # -------- Download options --------

    csv_bytes = display_data.to_csv(index=False).encode("utf-8")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with download_col:
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_bytes,
            file_name=f"crop_yield_prediction_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True
        )

# =====================================================
# FOOTER
# =====================================================

st.markdown(
    """
    <div class="farm-footer">
        🌾 Crop Yield Prediction System · Python · Scikit-learn · Streamlit
    </div>
    """,
    unsafe_allow_html=True
)