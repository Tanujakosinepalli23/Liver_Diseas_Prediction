import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Liver Disease Detection", layout="wide")

# ---------------------------
# Helper: load & clean data
# ---------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, sep=None, engine="python")
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df["sex"] = df["sex"].str.lower().map({"m": 1, "male": 1, "f": 0, "female": 0})
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

# ---------------------------
# Session state
# ---------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "reports" not in st.session_state:
    st.session_state.reports = []

# ---------------------------
# Sidebar navigation
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Reports"])

# ---------------------------
# Home Page
# ---------------------------
if page == "Home":
    st.title("🩺 Liver Disease Detection — Home")
    st.markdown("Upload a CSV with the required liver dataset. After uploading, the model will train automatically.")

    uploaded_file = st.file_uploader("📤 Upload Liver_data.csv", type=["csv"])
    if uploaded_file:
        try:
            df = load_data(uploaded_file)
            st.session_state.df = df
            st.success("Dataset uploaded successfully ✔️")
            st.dataframe(df.head())

            X = df.drop("category", axis=1)
            y = df["category"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
            model.fit(X_train_scaled, y_train)

            st.session_state.model = model
            st.session_state.scaler = scaler

            st.info("Model trained and ready for predictions.")
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")

# ---------------------------
# Prediction Page
# ---------------------------
elif page == "Prediction":
    st.title("🔍 Prediction & Real‑time Liver Health Score")

    if st.session_state.df is None:
        st.warning("Upload a dataset first on the Home page.")
    else:
        df = st.session_state.df
        model = st.session_state.model
        scaler = st.session_state.scaler

        # Healthy ranges
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

        st.markdown("### Enter Patient Values")

        inputs = {}
        abnormal_flags = {}
        numeric_cols = [c for c in df.drop("category", axis=1).columns]

        cols = st.columns(2)

        for i, col in enumerate(numeric_cols):
            with cols[i % 2]:
                if col == "sex":
                    val = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
                    inputs[col] = val
                    abnormal_flags[col] = False
                else:
                    col_min = float(df[col].min())
                    col_max = float(df[col].max())
                    default_val = float(df[col].median())
                    low, high = healthy_ranges.get(col, (col_min, col_max))

                    val = st.number_input(col, value=default_val, format="%.2f")
                    inputs[col] = val

                    # Show defined healthy range
                    st.markdown(f"**Normal Range:** {low} – {high}")

                    # Validate value
                    if val < low or val > high:
                        st.error(f"Abnormal — Out of Range ({low} – {high})")
                        abnormal_flags[col] = True
                    else:
                        st.success("Healthy — Within Range")
                        abnormal_flags[col] = False

        # Health Score
        def compute_health_score(inputs, healthy_ranges):
            scores = []
            for col, val in inputs.items():
                if col == "sex":
                    continue
                low, high = healthy_ranges.get(col, (None, None))
                if low is None:
                    scores.append(1.0)
                    continue
                if low <= val <= high:
                    scores.append(1.0)
                else:
                    dist = abs(val - (low + high) / 2)
                    half = (high - low) / 2
                    score = max(0, 1 - dist / half)
                    scores.append(score)
            return int((sum(scores) / len(scores)) * 100)

        health_score = compute_health_score(inputs, healthy_ranges)

        st.subheader("Overall Liver Health Score")
        st.metric("Health Score", f"{health_score}/100")
        st.progress(health_score / 100)

        # Prediction
        input_df = pd.DataFrame([inputs])
        input_scaled = scaler.transform(input_df)

        if st.button("🔍 Predict Stage and Save Report"):
            pred = model.predict(input_scaled)[0]

            category_map = {0: "No Disease", 1: "Suspect Disease", 2: "Hepatitis", 3: "Fibrosis", 4: "Cirrhosis"}
            severity_map = {0: "None", 1: "Mild", 2: "Moderate", 3: "High", 4: "Severe"}

            disease = category_map[pred]
            severity = severity_map[pred]

            if pred == 0:
                st.success(f"🟢 {disease} — {severity}")
            else:
                st.error(f"🔴 {disease} — {severity}")

            st.session_state.reports.append({**inputs, "predicted_stage": disease, "severity": severity, "health_score": health_score})
            st.success("Report saved!")

# ---------------------------
# Reports Page
# ---------------------------
elif page == "Reports":
    st.title("📄 Reports History")

    if not st.session_state.reports:
        st.info("No reports saved yet.")
    else:
        df_reports = pd.DataFrame(st.session_state.reports)
        st.dataframe(df_reports)

        csv = df_reports.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Reports CSV", csv, "liver_reports.csv", "text/csv")
