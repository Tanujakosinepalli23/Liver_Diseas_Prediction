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
if "model" not in st.session_state:
    st.session_state.model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "reports" not in st.session_state:
    st.session_state.reports = []
if "df" not in st.session_state:
    st.session_state.df = None

# -------------------------
# Default Dataset
# -------------------------
def load_default_data():
    data = {
        "age": np.random.randint(20, 70, 200),
        "sex": np.random.choice([0, 1], 200),
        "total_bilirubin": np.random.uniform(0.1, 2.0, 200),
        "direct_bilirubin": np.random.uniform(0.0, 0.5, 200),
        "alkphos": np.random.uniform(50, 150, 200),
        "sgpt": np.random.uniform(10, 60, 200),
        "sgot": np.random.uniform(10, 50, 200),
        "total_proteins": np.random.uniform(5.5, 8.5, 200),
        "albumin": np.random.uniform(3.0, 5.5, 200),
        "ag_ratio": np.random.uniform(0.8, 2.5, 200),
        "category": np.random.choice([0,1,2,3,4], 200)
    }
    return pd.DataFrame(data)

# -------------------------
# Clean Data (SAFE)
# -------------------------
def clean_data(df):
    df.columns = df.columns.str.strip().str.lower()

    if "sex" in df.columns:
        df["sex"] = df["sex"].astype(str).str.lower().map({
            "m": 1, "male": 1,
            "f": 0, "female": 0,
            "1": 1, "0": 0
        })

    # Convert numeric safely
    for col in df.columns:
        if col != "category":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert category if exists
    if "category" in df.columns:
        if df["category"].dtype == object:
            df["category"] = df["category"].str.lower().map({
                "no_disease": 0,
                "suspect_disease": 1,
                "hepatitis": 2,
                "fibrosis": 3,
                "cirrhosis": 4
            })

    df = df.dropna()
    return df

# -------------------------
# Train Model (SAFE)
# -------------------------
def train_model(df):

    if "category" not in df.columns:
        st.error("❌ Dataset must contain 'category' column for training")
        st.stop()

    X = df.drop("category", axis=1)
    y = df["category"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler, X.columns

# -------------------------
# Dataset Selection
# -------------------------
st.header("📊 Dataset Selection")

option = st.radio("Choose Dataset", ["Use Default Dataset", "Upload Your Dataset"])

if option == "Use Default Dataset":
    df = load_default_data()
    st.session_state.df = df
    st.success("✅ Default dataset loaded")

elif option == "Upload Your Dataset":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = clean_data(df)
        st.session_state.df = df
        st.success("✅ Uploaded dataset loaded")

# -------------------------
# Train Model
# -------------------------
if st.session_state.df is not None:
    st.dataframe(st.session_state.df.head())

    if st.button("🚀 Train Model"):
        model, scaler, feature_cols = train_model(st.session_state.df)
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.feature_cols = feature_cols
        st.success("Model trained successfully ✔️")

# -------------------------
# Prediction Section
# -------------------------
st.header("🔍 Enter Patient Details")

if st.session_state.model is None:
    st.warning("⚠️ Train model first")
else:
    df = st.session_state.df
    model = st.session_state.model
    scaler = st.session_state.scaler
    feature_cols = st.session_state.feature_cols

    inputs = {}
    cols = st.columns(2)

    for i, col in enumerate(feature_cols):
        with cols[i % 2]:

            if col == "sex":
                val = st.selectbox("Sex", [0, 1],
                                   format_func=lambda x: "Male" if x == 1 else "Female")

            elif col == "age":
                val = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)

            else:
                val = st.number_input(col, min_value=0.0, max_value=1000.0,
                                      value=float(df[col].median()) if col in df.columns else 0.0,
                                      step=0.1)

            inputs[col] = val

    # -------------------------
    # Health Score
    # -------------------------
    def compute_health_score(inputs):
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

        penalty = 0

        for key, val in inputs.items():
            if key == "sex":
                continue

            low, high = healthy_ranges.get(key, (None, None))
            if low is None:
                continue

            if val < low:
                penalty += (low - val) * 4
            elif val > high:
                penalty += (val - high) * 4

        return max(0, 100 - int(penalty))

    health_score = compute_health_score(inputs)

    # Animated Bar
    st.subheader("💚 Health Score")

    score_placeholder = st.empty()
    bar = st.progress(0)

    for i in range(health_score + 1):
        time.sleep(0.01)
        bar.progress(i / 100)
        score_placeholder.metric("Score", f"{i}/100")

    # -------------------------
    # Prediction
    # -------------------------
    if st.button("Predict & Save Report"):
        input_df = pd.DataFrame([inputs])
        input_df = input_df[feature_cols]  # ensure correct order
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
    rep_df = pd.DataFrame(st.session_state.reports)
    rep_df.index = range(1, len(rep_df) + 1)
    st.dataframe(rep_df)

    csv = rep_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Reports", csv, "reports.csv")
else:
    st.info("No reports yet")
