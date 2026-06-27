import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load dataset
df = pd.read_csv("dataset/credit_risk_dataset.csv")

# Handle missing values
df = df.fillna(0)

# Encode categorical columns
categorical_columns = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file"
]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# -------- Feature Engineering --------

# Debt feature
df["debt_ratio"] = df["loan_amnt"] / (df["person_income"] + 1)

# Payment history feature
# default_on_file + credit history length
df["payment_history_score"] = (
    df["cb_person_cred_hist_length"] * 2
) - df["cb_person_default_on_file"]

# -----------------------------------

# Features and target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Metrics
print("===== MODEL PERFORMANCE =====")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred))
print("Recall :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC :", roc_auc_score(y_test, y_pred))