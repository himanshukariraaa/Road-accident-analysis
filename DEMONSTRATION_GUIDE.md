# Demonstration Guide (about 5–7 minutes)

## Before the evaluator arrives
1. Start the app using the README commands.
2. Open `http://localhost:8501`.
3. Check that the Overview page loads and charts are visible.
4. Keep this guide open separately for your explanation.

## Click-by-click walkthrough

### 1. Overview & Analytics (1 minute)
**Do:** Show the record count, number of features, most common severity and best model macro-F1. Scroll through severity distribution, hourly trends, weather vs severity and road type vs severity.

**Say:** “This module performs exploratory data analysis. It helps us inspect how the records are distributed across severity classes and compare patterns across time, weather and road types. The included records are synthetic demo data, so these charts demonstrate the software flow rather than real accident statistics.”

### 2. Predict Severity (1–2 minutes)
**Do:** Choose a state, year, month, hour, weather, road type, vehicle type and cause. Click **Predict severity**. Show the predicted class, project risk interpretation and class-score chart. Change one input and predict again.

**Say:** “The form creates one input row. The same preprocessing pipeline used during training transforms the row, and the selected classifier predicts one of three classes: Slight, Serious or Fatal. The displayed scores are model outputs, not guaranteed real-world probabilities. Risk labels are defined by the project.”

### 3. Model Evaluation (1–2 minutes)
**Do:** Show the model comparison table. Select a model and explain its confusion matrix. Point out the macro-F1 score.

**Say:** “We compare Logistic Regression, Decision Tree and Random Forest on a held-out test set. Macro-F1 averages F1 across the classes, so it is more informative than accuracy alone when class counts differ. The confusion matrix shows correct predictions on the diagonal and misclassifications elsewhere.”

### 4. Dataset Explorer (1 minute)
**Do:** Filter by state, severity or weather. Show the filtered rows and click **Download filtered CSV**.

**Say:** “This module allows the user to inspect and filter records and export the selected subset for further analysis.”

## Likely viva questions and answers

**Q1. What is the aim of this project?**
To analyze accident-related features and demonstrate a machine-learning system that classifies accident severity and provides an interactive interface.

**Q2. What is classification?**
Classification is a supervised machine-learning task that predicts a category from input features. Here the target categories are Slight, Serious and Fatal.

**Q3. Why do preprocessing?**
Real datasets can contain missing values and categorical text. Imputation handles missing values, one-hot encoding converts categories into numeric features, and scaling helps algorithms that are sensitive to feature scale.

**Q4. Why compare multiple models?**
Different algorithms learn different decision boundaries. Comparing them on the same held-out test set helps select a model based on measured performance rather than assumption.

**Q5. What is a confusion matrix?**
A table comparing actual and predicted classes. It shows which classes are correctly classified and which are confused with others.

**Q6. Why macro-F1?**
It calculates F1 separately for each class and averages the scores equally, preventing a frequent class from completely dominating the metric.

**Q7. Is this usable for real-time accident prevention?**
No. It is an educational prototype. The included data is synthetic and the output is not validated for real-world safety decisions.

**Q8. What are the limitations?**
Synthetic demo data, possible class imbalance, limited input features, no external validation, and uncalibrated class scores. Real licensed accident data is needed for credible real-world evaluation.

**Q9. What is the future scope?**
Use official validated data, add road-lighting and junction features, perform cross-validation and external testing, improve calibration and interpretability, and evaluate performance separately for each class and region.

## Honest note
Do not claim that the synthetic dataset is official, that the model is production-ready, or that the displayed metrics represent real-world accuracy. If asked, explain that the prototype demonstrates the end-to-end workflow and that real-data validation is the next step.
