from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Road Accident Severity & Risk Prediction", page_icon="🚦", layout="wide")
DATA_PATH = Path(__file__).parent / "data" / "demo_accidents.csv"
FEATURES = ["State", "Year", "Month", "Hour", "Weather_Condition", "Road_Type", "Vehicle_Type", "Cause_of_Accident"]
TARGET = "Accident_Severity"
CLASSES = ["Slight", "Serious", "Fatal"]
RISK = {"Slight": "Low", "Serious": "Medium", "Fatal": "High"}

@st.cache_data
def load_demo():
    return pd.read_csv(DATA_PATH)

def clean_data(df):
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    data = df[required].copy()
    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
    data["Month"] = pd.to_numeric(data["Month"], errors="coerce")
    data["Hour"] = pd.to_numeric(data["Hour"], errors="coerce")
    for c in ["State", "Weather_Condition", "Road_Type", "Vehicle_Type", "Cause_of_Accident", TARGET]:
        data[c] = data[c].astype("string").str.strip()
        data[c] = data[c].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        data[c] = data[c].astype(object).where(data[c].notna(), np.nan)
    data = data.dropna(subset=[TARGET])
    data = data[data[TARGET].isin(CLASSES)]
    if len(data) < 30:
        raise ValueError("Please upload at least 30 valid rows with Slight/Serious/Fatal labels.")
    return data

def train_models(df):
    X, y = df[FEATURES], df[TARGET]
    counts = y.value_counts()
    stratify = y if len(counts) >= 2 and counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)
    numeric = ["Year", "Month", "Hour"]
    categorical = [c for c in FEATURES if c not in numeric]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)])
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=4, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=160, max_depth=14, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1),
    }
    results, models = [], {}
    for name, estimator in estimators.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append({"Model": name, "Accuracy": accuracy_score(y_test, pred), "Macro precision": precision_score(y_test, pred, average="macro", zero_division=0), "Macro recall": recall_score(y_test, pred, average="macro", zero_division=0), "Macro F1": f1_score(y_test, pred, average="macro", zero_division=0)})
        models[name] = {"pipeline": pipe, "pred": pred, "y_test": y_test}
    scores = pd.DataFrame(results).sort_values("Macro F1", ascending=False).reset_index(drop=True)
    best_name = scores.iloc[0]["Model"]
    return scores, models, best_name

st.title("🚦 Road Accident Severity and Risk Prediction System")
st.caption("BCA Minor Project • Machine Learning prototype • Educational use only")
st.warning("Dataset notice: the bundled CSV is synthetic demonstration data, not official accident records. Metrics and predictions from it are not evidence of real-world accuracy.")
with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Optionally upload a CSV", type=["csv"], help="Required columns are listed in README.md. Target labels must be Slight, Serious or Fatal.")
    st.caption("No upload? The app uses the included demo dataset.")

try:
    raw = pd.read_csv(uploaded) if uploaded is not None else load_demo()
    df = clean_data(raw)
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

with st.spinner("Training and evaluating three models…"):
    try:
        scores, models, best_name = train_models(df)
    except Exception as e:
        st.error(f"Model training failed. Check the dataset has enough examples of each class. Details: {e}")
        st.stop()

page = st.radio("Choose a module", ["Overview & EDA", "Predict Severity", "Model Evaluation", "Dataset Explorer"], horizontal=True)
if page == "Overview & EDA":
    st.subheader("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Input features", len(FEATURES))
    c3.metric("Severity classes", df[TARGET].nunique())
    c4.metric("Selected model", best_name)
    st.markdown("### Accident severity distribution")
    counts = df[TARGET].value_counts().reindex(CLASSES, fill_value=0).rename_axis("Severity").reset_index(name="Records")
    left, right = st.columns(2)
    left.plotly_chart(px.bar(counts, x="Severity", y="Records", color="Severity", title="Records by severity"), use_container_width=True)
    hourly = df.groupby("Hour", as_index=False).size().rename(columns={"size": "Records"})
    right.plotly_chart(px.line(hourly, x="Hour", y="Records", markers=True, title="Records by hour of day"), use_container_width=True)
    left, right = st.columns(2)
    weather = df.groupby(["Weather_Condition", TARGET]).size().reset_index(name="Records")
    left.plotly_chart(px.bar(weather, x="Weather_Condition", y="Records", color=TARGET, barmode="group", title="Severity by weather"), use_container_width=True)
    road = df.groupby(["Road_Type", TARGET]).size().reset_index(name="Records")
    right.plotly_chart(px.bar(road, x="Road_Type", y="Records", color=TARGET, barmode="group", title="Severity by road type"), use_container_width=True)
    st.info("Interpret charts as patterns within the selected dataset only. With synthetic data, these are illustrative patterns rather than real-world findings.")
elif page == "Predict Severity":
    st.subheader("Try a hypothetical accident scenario")
    st.write("Choose contextual details, then the trained model will return a severity class and the project's simple risk mapping.")
    with st.form("prediction_form"):
        c1, c2, c3, c4 = st.columns(4)
        states = sorted(df["State"].dropna().astype(str).unique().tolist()) or ["Rajasthan"]
        weather_values = sorted(df["Weather_Condition"].dropna().astype(str).unique().tolist()) or ["Clear"]
        road_values = sorted(df["Road_Type"].dropna().astype(str).unique().tolist()) or ["Urban road"]
        vehicle_values = sorted(df["Vehicle_Type"].dropna().astype(str).unique().tolist()) or ["Car"]
        cause_values = sorted(df["Cause_of_Accident"].dropna().astype(str).unique().tolist()) or ["Other"]
        state = c1.selectbox("State", states)
        year = c2.selectbox("Year", list(range(2019, 2027)), index=7)
        month = c3.selectbox("Month", list(range(1, 13)), index=9)
        hour = c4.slider("Hour (0–23)", 0, 23, 18)
        weather = c1.selectbox("Weather", weather_values)
        road = c2.selectbox("Road type", road_values)
        vehicle = c3.selectbox("Vehicle type", vehicle_values)
        cause = c4.selectbox("Reported cause", cause_values)
        submitted = st.form_submit_button("Predict severity", type="primary")
    if submitted:
        model = models[best_name]["pipeline"]
        row = pd.DataFrame([{"State": state, "Year": year, "Month": month, "Hour": hour, "Weather_Condition": weather, "Road_Type": road, "Vehicle_Type": vehicle, "Cause_of_Accident": cause}])[FEATURES]
        prediction = str(model.predict(row)[0])
        st.success(f"Predicted severity: {prediction} | Project risk category: {RISK.get(prediction, 'Not mapped')}")
        if hasattr(model.named_steps["model"], "predict_proba"):
            probabilities = model.predict_proba(row)[0]
            prob_df = pd.DataFrame({"Class": model.named_steps["model"].classes_, "Model score (%)": probabilities * 100}).sort_values("Model score (%)", ascending=False)
            st.plotly_chart(px.bar(prob_df, x="Class", y="Model score (%)", title="Model class scores (not calibrated real-world probabilities)"), use_container_width=True)
        st.caption("This is a classroom prototype. Do not use it to make emergency, medical, insurance or real-time driving decisions.")
elif page == "Model Evaluation":
    st.subheader("Held-out test-set comparison")
    st.dataframe(scores.style.format({"Accuracy": "{:.3f}", "Macro precision": "{:.3f}", "Macro recall": "{:.3f}", "Macro F1": "{:.3f}"}), use_container_width=True, hide_index=True)
    st.caption("The test split is 25% of this dataset (random_state=42). These values are meaningful only to the extent that the dataset represents the real problem.")
    chosen = st.selectbox("Inspect model", list(models.keys()), index=list(models.keys()).index(best_name))
    result = models[chosen]
    labels = [c for c in CLASSES if c in set(result["y_test"].astype(str)) | set(pd.Series(result["pred"]).astype(str))]
    cm = confusion_matrix(result["y_test"], result["pred"], labels=labels)
    st.plotly_chart(px.imshow(cm, x=labels, y=labels, text_auto=True, labels={"x": "Predicted", "y": "Actual", "color": "Count"}, title=f"Confusion matrix — {chosen}"), use_container_width=True)
    report = classification_report(result["y_test"], result["pred"], labels=labels, zero_division=0, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
elif page == "Dataset Explorer":
    st.subheader("Inspect and export records")
    selected = st.multiselect("Filter by severity", CLASSES, default=CLASSES)
    view = df[df[TARGET].isin(selected)]
    st.write(f"Showing {len(view):,} of {len(df):,} records")
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("Download filtered CSV", data=view.to_csv(index=False).encode("utf-8"), file_name="accident_records_filtered.csv", mime="text/csv")

with st.expander("How this prototype works"):
    st.markdown("""
    1. **Preprocessing:** numeric fields use median imputation and scaling; categorical fields use most-frequent imputation and one-hot encoding.
    2. **Training:** Logistic Regression, Decision Tree and Random Forest are trained on 75% of the records.
    3. **Evaluation:** models are compared on the held-out 25% using accuracy, macro precision, macro recall and macro F1.
    4. **Prediction:** the model with the highest macro F1 is used for the interactive scenario form.
    5. **Risk mapping:** Slight → Low, Serious → Medium, Fatal → High. This is a project-defined mapping, not a separately validated risk model.
    """)
