# Short presentation script

“Good morning. My project is titled Road Accident Severity and Risk Prediction System Using Machine Learning. The objective is to demonstrate how accident-context features can be analyzed and passed to classification algorithms to estimate a severity class.

The system has four main screens. The Overview screen visualizes severity distribution and patterns by time, weather and road type. The Prediction screen takes conditions such as weather, road type, vehicle type, time and reported cause and returns a predicted class. The Model Evaluation screen compares Logistic Regression, Decision Tree and Random Forest using accuracy and macro-averaged precision, recall and F1, alongside a confusion matrix. The Dataset Explorer lets a user filter records and export them to CSV.

Preprocessing is implemented inside scikit-learn pipelines. Categorical variables are imputed and one-hot encoded, while numeric values are imputed and scaled. The data is split into training and testing subsets, and the best model is selected using macro-F1 rather than accuracy alone.

One limitation is important: the included dataset is synthetic demonstration data. It lets the application run end-to-end, but the metrics must not be interpreted as real-world accident prediction performance. A future version should use documented real accident data and independent validation. This is an educational prototype, not a tool for real-time safety decisions. Thank you.”
