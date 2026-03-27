from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

model = None
scaler = None
reports = []

# ---------------------------
# Train Model
# ---------------------------
def train_model(file):
    global model, scaler

    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

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

    X = df.drop("category", axis=1)
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train_scaled, y_train)


# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    train_model(file)
    return jsonify({"message": "Model trained successfully"})


@app.route("/predict", methods=["POST"])
def predict():
    global model, scaler, reports

    data = request.json
    df = pd.DataFrame([data])

    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)[0]

    category_map = {
        0: "No Disease",
        1: "Suspect Disease",
        2: "Hepatitis",
        3: "Fibrosis",
        4: "Cirrhosis"
    }

    result = category_map[pred]

    reports.append({**data, "prediction": result})

    return jsonify({"prediction": result})


@app.route("/reports")
def get_reports():
    return jsonify(reports)


if __name__ == "__main__":
    app.run(debug=True)
