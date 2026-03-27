import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Liver Disease Detection", layout="wide")

st.title("🩺 Liver Disease Detection Web App")

# -------------------------
# Session State
# -------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "reports" not in st.session_state:
    st.session_state.reports = []

# -------------------------
# Load & Clean Data
# -------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    # Convert sex
    if "sex" in df.columns:
        df["sex"] = df["sex"].astype(str).str.lower().map({
            "m": 1, "male": 1,
            "f": 0, "female": 0
        })

    # Convert numeric safely
    for col in df.columns:
        if col != "category":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert category if exists
    if "category" in df.columns:
        df["category"] = df["category"].astype(str).str.lower().map({
            "no_disease": 0,
            "suspect_disease": 1,
            "hepatitis": 2,
            "fibrosis": 3,
            "cirrhosis": 4
        })

    df = df.dropna()
    return df

# -------------------------
# Upload Dataset
# -------------------------
st.header("📤 Upload Dataset")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.session_state.df = df
    st.success("Dataset loaded successfully ✔️")
    st.dataframe(df.head())

    # Train only if category exists
    if "category" in df.columns:
        X = df.drop("category", axis=1)
        y = df["category"]

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train_scaled, y_train)

        st.session_state.model = model
        st.session_state.scaler = scaler

        st.success("Model trained ✔️")
    else:
        st.warning("⚠️ No 'category' column → Prediction disabled")

# -------------------------
# Prediction Section
# -------------------------
st.header("🔍 Prediction & Health Score")

if st.session_state.df is None:
    st.warning("Upload dataset first")
    st.stop()

df = st.session_state.df

# STOP if no model
if st.session_state.model is None:
    st.warning("⚠️ Upload dataset WITH 'category' column for prediction")
    st.stop()

model = st.session_state.model
scaler = st.session_state.scaler

# -------------------------
# SAFE FEATURES (NO ERROR)
# -------------------------
features = [col for col in df.columns if col != "category"]

# -------------------------
# Healthy ranges
# -------------------------
healthy_ranges = {
    "age": (18, 80),
    "total_bilirubin": (0.1, 1.2),
    "direct_bilirubin": (0.0, 0.3),
    "alkphos": (44, 147),
    "sgpt": (7, 56),
    "sgot": (5, 40),
    "total_proteins": (6.0, 8.3),
    "albumin": (3.5, 5.0),
    "ag_ratio": (1.0, 2.5)
}

# -------------------------
# Inputs
# -------------------------
inputs = {}
cols = st.columns(2)

for i, col in enumerate(features):
    with cols[i % 2]:

        if col == "sex":
            val = st.selectbox("Sex", [0, 1],
                               format_func=lambda x: "Male" if x == 1 else "Female")

        elif col == "age":
            val = st.number_input("Age", min_value=0, max_value=120, value=30)

        else:
            default_val = float(df[col].median())
            val = st.number_input(col, value=default_val + 0.05*default_val)

        inputs[col] = val

        if col in healthy_ranges:
            low, high = healthy_ranges[col]
            st.caption(f"Normal: {low} – {high}")

# -------------------------
# HEALTH SCORE (FIXED)
# -------------------------
def compute_health_score(inputs, healthy_ranges):
    total_score = 0
    count = 0

    for col, val in inputs.items():
        if col == "sex":
            continue

        low, high = healthy_ranges.get(col, (None, None))
        if low is None:
            continue

        mid = (low + high) / 2
        width = (high - low)

        deviation = abs(val - mid)
        score = max(0, 1 - (deviation / width))

        total_score += score
        count += 1

    return int((total_score / count) * 100) if count else 0

health_score = compute_health_score(inputs, healthy_ranges)

# -------------------------
# Animated Progress Bar
# -------------------------
st.subheader("💚 Health Score")

bar = st.progress(0)
score_placeholder = st.empty()

for i in range(health_score + 1):
    time.sleep(0.01)
    bar.progress(i / 100)
    score_placeholder.metric("Score", f"{i}/100")

# -------------------------
# Prediction
# -------------------------
if st.button("Predict & Save Report"):
    input_df = pd.DataFrame([inputs])
    input_df = input_df[features]

    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]

    category_map = {
        0: "No Disease",
        1: "Suspect Disease",
        2: "Hepatitis",
        3: "Fibrosis",
        4: "Cirrhosis"
    }

    result = category_map[pred]

    if pred == 0:
        st.success(f"🟢 {result}")
    else:
        st.error(f"🔴 {result}")

    st.session_state.reports.append({
        **inputs,
        "prediction": result,
        "health_score": health_score
    })

# -------------------------
# Reports
# -------------------------
st.header("📄 Reports")

if st.session_state.reports:
    df_reports = pd.DataFrame(st.session_state.reports)
    df_reports.index = range(1, len(df_reports) + 1)
    st.dataframe(df_reports)

    csv = df_reports.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Reports", csv, "reports.csv")
else:
    st.info("No reports yet")
