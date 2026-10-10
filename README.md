# Road Accident Severity and Risk Prediction System Using Machine Learning

**BCA Minor Project | Manipal University Jaipur | 2026–27**

## Important dataset disclosure

The included `data/demo_accidents.csv` contains **3,500 synthetic/demo records** generated for demonstrating the full application flow. It is not a real-world accident dataset and must not be described as official Indian accident statistics. Its severity labels were generated using a heuristic, so evaluation scores demonstrate that the pipeline runs; they do not establish real-world predictive performance.

For a research-grade result, replace it with a properly licensed real dataset and re-run the same evaluation. The app supports CSV upload. Required columns and target labels are listed below.

## Features
- Dataset validation and preprocessing with missing-value imputation and one-hot encoding.
- Exploratory visualizations for severity, time, weather, and road type.
- Three classifiers: Logistic Regression, Decision Tree, and Random Forest.
- Held-out test evaluation: accuracy, macro precision, macro recall, macro F1, classification report, confusion matrix.
- Interactive severity prediction and model class scores.
- Dataset filtering and CSV export.

## Requirements
- Python 3.10 or newer (Python 3.11 recommended)
- Internet connection for the initial package installation
- Windows, macOS, or Linux

## Run on Windows (recommended steps)
1. Extract the ZIP to a folder, e.g. `Desktop\road_accident_project`.
2. Open that folder in File Explorer.
3. Click the address bar, type `cmd`, and press Enter.
4. Run these commands one at a time:

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

5. A browser tab should open at `http://localhost:8501`. If it does not, copy that address into Chrome/Edge.
6. Keep the command window open while demonstrating the app. Press `Ctrl+C` in it to stop the app.

### If `py` is not recognized
Install Python from https://www.python.org/downloads/ and tick **Add Python to PATH** during setup, then reopen Command Prompt and retry.

## CSV format for replacing demo data
The uploaded CSV must contain these exact columns:

- `State`
- `Year` (numeric)
- `Month` (numeric 1–12)
- `Hour` (numeric 0–23)
- `Weather_Condition`
- `Road_Type`
- `Vehicle_Type`
- `Cause_of_Accident`
- `Accident_Severity` (labels: `Slight`, `Serious`, `Fatal`)

Use the app sidebar's CSV uploader to try another dataset. Dataset categories are learned from the uploaded file, but the year selector in the demo form is fixed to 2019–2026; update it in `app.py` if your dataset covers a different period.

## Project structure

```text
road_accident_project/
├── app.py                      # Streamlit user interface, EDA, evaluation and prediction
├── requirements.txt            # Python dependencies
├── README.md                   # Setup and dataset notes
├── DEMONSTRATION_GUIDE.md      # Click-by-click demo and viva explanations
├── PROJECT_REPORT.md           # Implementation report draft
├── data/
│   └── demo_accidents.csv      # Synthetic demo dataset; explicitly disclosed
├── models/                     # Reserved for saved model artifacts
└── outputs/                    # Reserved for exported evaluation figures/results
```

## Model methodology
1. Select target `Accident_Severity` and the eight contextual input features.
2. Split into training and held-out test data (75/25, random seed 42; stratified where possible).
3. Impute missing categorical values with the most frequent value and numerical values with the median.
4. One-hot encode categorical features and scale numeric features inside a scikit-learn pipeline to avoid preprocessing leakage.
5. Train Logistic Regression, Decision Tree, and Random Forest.
6. Compare Accuracy, macro Precision, macro Recall and macro F1 on the held-out test set.
7. Select the model with the highest macro F1 for the interactive form.

## Limitations
- The bundled data is synthetic; results are not evidence of real-world accuracy.
- The class-score output is not a calibrated probability of a real accident outcome.
- The risk category is a project-defined mapping: Slight → Low, Serious → Medium, Fatal → High.
- The application is educational and must not be used for emergency response or real-time driving decisions.
