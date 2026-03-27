import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Page Config
# -------------------------
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
# Load Data
# -------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, sep=None, engine="python")
    df.columns = df.columns.str.strip()

    df["sex"] = df["sex"].str.lower().map({
        "m": 1, "male": 1,
        "f": 0, "female": 0
    })

    numeric_cols = df.columns.drop("category")
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

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
# Upload + Train
# -------------------------
st.header("📤 Upload Dataset")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.session_state.df = df

    st.success("Dataset uploaded successfully ✔️")
    st.dataframe(df.head())

    X = df.drop("category", axis=1)
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    st.session_state.model = model
    st.session_state.scaler = scaler

    st.success("✅ Model trained successfully!")

# -------------------------
# Prediction Section
# -------------------------
st.header("🔍 Enter Patient Details")

if st.session_state.model is None:
    st.warning("⚠️ Please upload dataset first")
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

    # -------------------------
    # Health Score
    # -------------------------
    def compute_health_score(inputs):
        score = 100
        for key, val in inputs.items():
            if key != "sex":
                if val <= 0:
                    score -= 5
        return max(score, 0)

    health_score = compute_health_score(inputs)

    st.subheader("💚 Health Score")
    st.metric("Score", f"{health_score}/100")
    st.progress(health_score / 100)

    # -------------------------
    # Prediction Button
    # -------------------------
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

        st.success("Report saved ✔️")

# -------------------------
# Reports Section
# -------------------------
st.header("📄 Reports History")

if st.session_state.reports:
    reports_df = pd.DataFrame(st.session_state.reports)
    st.dataframe(reports_df)

    csv = reports_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Reports", csv, "reports.csv")
else:
    st.info("No reports available yet.")
