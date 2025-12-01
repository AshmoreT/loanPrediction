
import streamlit as st
import numpy as np
import pickle
from utils.preprocessing import preprocess_input

@st.cache_resource
def load_model():
    with open('Random_Forest.sav', 'rb') as f:
        return pickle.load(f)

def main():
    st.title("🏦 Loan Approval Prediction System")
    st.markdown("""
    Welcome to the **Loan Approval Prediction App**.
    This tool uses a machine learning model trained on financial data to estimate the likelihood of loan approval.
    Please enter applicant details in the sidebar.
    """)

    model = load_model()

    # Sidebar inputs
    st.sidebar.header("Applicant Information")
    Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    Married = st.sidebar.selectbox("Married", ["Yes", "No"])
    Dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
    ApplicantIncome = st.sidebar.number_input("Applicant Income", min_value=0, max_value=100000, value=5000)
    CoapplicantIncome = st.sidebar.number_input("Coapplicant Income", min_value=0, max_value=50000, value=0)
    LoanAmount = st.sidebar.number_input("Loan Amount (in thousands)", min_value=0, max_value=1000, value=150)
    Loan_Amount_Term = st.sidebar.selectbox("Loan Amount Term (months)", [360, 180, 120, 60])
    Credit_History = st.sidebar.selectbox("Credit History", [1, 0])
    Property_Area = st.sidebar.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    # Preprocess input
    input_df = preprocess_input(Gender, Married, Dependents, Education, Self_Employed,
                                ApplicantIncome, CoapplicantIncome, LoanAmount,
                                Loan_Amount_Term, Credit_History, Property_Area)

    if st.button("Predict Loan Approval"):
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][prediction]

        if prediction == 1:
            st.success(f" Congratulations! Loan likely to be approved with confidence {proba:.2%}.")
        else:
            st.error(f" Sorry, loan likely to be denied with confidence {proba:.2%}.")

if __name__ == "__main__":
    main()
