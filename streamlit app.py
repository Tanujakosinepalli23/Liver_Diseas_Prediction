import streamlit as st
import pandas as pd
import numpy as np
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
# Data Cleaning
# -------------------------
def clean_data(df):
    df.columns = df.columns.str.strip()

    if "sex" in df.columns:
        df["sex"] = df["sex"].astype(str).str.lower().map({
            "m": 1, "male": 1,
            "f": 0, "female": 0,
            "1": 1, "0": 0
        })

    numeric_cols = df.columns.drop("category")
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

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
# Train Model
# -------------------------
def train_model(df):
    X = df.drop("category", axis=1)
    y = df["category"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler

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
# Train Model Button
# -------------------------
if st.session_state.df is not None:
    st.dataframe(st.session_state.df.head())

    if st.button("🚀 Train Model"):
        model, scaler = train_model(st.session_state.df)
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.success("Model trained successfully ✔️")

# -------------------------
# Prediction
# -------------------------
st.header("🔍 Enter Patient Details")

if st.session_state.model is None:
    st.warning("⚠️ Train model first")
else:
    df = st.session_state.df
    model = st.session_state.model
    scaler = st.session_state.scaler

    inputs = {}
    cols = st.columns(2)

    features = df.drop("category", axis=1).columns

    for i, col in enumerate(features):
        with cols[i % 2]:
            if col == "sex":
                val = st.selectbox("Sex", [0, 1],
                                   format_func=lambda x: "Male" if x == 1 else "Female")
            else:
                val = st.number_input(col, value=float(df[col].median()))
            inputs[col] = val

    # Health Score
    def compute_health_score(inputs):
        score = 100
        for k, v in inputs.items():
            if k != "sex" and v <= 0:
                score -= 5
        return max(score, 0)

    health_score = compute_health_score(inputs)

    st.subheader("💚 Health Score")
    st.metric("Score", f"{health_score}/100")
    st.progress(health_score / 100)

    if st.button("Predict & Save Report"):
        input_df = pd.DataFrame([inputs])
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
    st.dataframe(rep_df)

    csv = rep_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Reports", csv, "reports.csv")
else:
    st.info("No reports yet")
