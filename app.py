import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Credit Scoring AI",
    page_icon="💳",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Background */
.stApp{
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

/* Title */
.main-title{
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:#38bdf8;
    animation: glow 2s infinite alternate;
}

/* Subtitle */
.sub{
    text-align:center;
    font-size:18px;
    color:#cbd5e1;
    margin-bottom:30px;
}

/* Button */
.stButton > button{
    width:100%;
    background: linear-gradient(90deg,#2563eb,#7c3aed);
    color:white;
    font-size:20px;
    border-radius:12px;
    height:55px;
    border:none;
    transition:0.3s;
}

.stButton > button:hover{
    transform: scale(1.03);
    box-shadow:0px 0px 20px #60a5fa;
}

/* Animation */
@keyframes glow{
    from{ text-shadow:0 0 10px #38bdf8; }
    to{ text-shadow:0 0 25px #7c3aed; }
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("dataset/credit_risk_dataset.csv")
df = df.fillna(0)

# ---------------- ENCODERS ----------------
home_encoder = LabelEncoder()
intent_encoder = LabelEncoder()
grade_encoder = LabelEncoder()
default_encoder = LabelEncoder()

# Encode dataset
df["person_home_ownership"] = home_encoder.fit_transform(df["person_home_ownership"])
df["loan_intent"] = intent_encoder.fit_transform(df["loan_intent"])
df["loan_grade"] = grade_encoder.fit_transform(df["loan_grade"])
df["cb_person_default_on_file"] = default_encoder.fit_transform(df["cb_person_default_on_file"])

# Feature engineering
df["debt_ratio"] = df["loan_amnt"] / (df["person_income"] + 1)
df["payment_history_score"] = (
    df["cb_person_cred_hist_length"] * 2
) - df["cb_person_default_on_file"]

# Train model
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

model = LogisticRegression(max_iter=3000)
model.fit(X, y)

# ---------------- UI ----------------
st.markdown(
    '<p class="main-title">💳 Credit Scoring AI</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub">Smart Loan Approval Prediction using Machine Learning</p>',
    unsafe_allow_html=True
)

# Two columns layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100)
    income = st.number_input("Annual Income")
    emp_length = st.number_input("Employment Length (Years)")
    loan_amount = st.number_input("Loan Amount")
    cred_hist = st.number_input("Credit History Length")

with col2:
    loan_int_rate = st.number_input("Interest Rate")
    loan_percent_income = st.number_input("Loan Percent Income")

    home = st.selectbox(
        "Home Ownership",
        ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )

    loan_intent = st.selectbox(
        "Loan Intent",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
         "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
    )

    loan_grade = st.selectbox(
        "Loan Grade",
        ["A", "B", "C", "D", "E", "F", "G"]
    )

    default = st.selectbox(
        "Previous Default",
        ["Y", "N"]
    )

# Encode input
home_encoded = home_encoder.transform([home])[0]
intent_encoded = intent_encoder.transform([loan_intent])[0]
grade_encoded = grade_encoder.transform([loan_grade])[0]
default_encoded = default_encoder.transform([default])[0]

# Engineered input
debt_ratio = loan_amount / (income + 1)
payment_history_score = (cred_hist * 2) - default_encoded

# Prediction button
if st.button("🔍 Predict Credit Score"):

    data = [[
        age,
        income,
        home_encoded,
        emp_length,
        intent_encoded,
        grade_encoded,
        loan_amount,
        loan_int_rate,
        loan_percent_income,
        default_encoded,
        cred_hist,
        debt_ratio,
        payment_history_score
    ]]

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.markdown("## ❌ High Credit Risk")
        st.warning("Loan Application Rejected")

    else:
        st.balloons()
        st.markdown("## ✅ Low Credit Risk")
        st.success("Loan Approved Successfully 🎉")