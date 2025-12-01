
import pandas as pd

def preprocess_input(Gender, Married, Dependents, Education, Self_Employed,
                     ApplicantIncome, CoapplicantIncome, LoanAmount,
                     Loan_Amount_Term, Credit_History, Property_Area):
    # Encoding categorical values
    gender_val = 1 if Gender == "Male" else 0
    married_val = 1 if Married == "Yes" else 0
    education_val = 1 if Education == "Graduate" else 0
    self_emp_val = 1 if Self_Employed == "Yes" else 0

    # Dependents one-hot encoding
    dependents_map = {
        "0": [1, 0, 0, 0],
        "1": [0, 1, 0, 0],
        "2": [0, 0, 1, 0],
        "3+": [0, 0, 0, 1]
    }
    dep_0, dep_1, dep_2, dep_3 = dependents_map[Dependents]

    # Property area one-hot encoding
    property_map = {
        "Urban": [1, 0, 0],
        "Rural": [0, 1, 0],
        "Semiurban": [0, 0, 1]
    }
    prop_urban, prop_rural, prop_semiurban = property_map[Property_Area]

    # Construct input dataframe for prediction
    input_dict = {
        "ApplicantIncome": [ApplicantIncome],
        "CoapplicantIncome": [CoapplicantIncome],
        "LoanAmount": [LoanAmount],
        "Loan_Amount_Term": [Loan_Amount_Term],
        "Credit_History": [Credit_History],
        "Gender": [gender_val],
        "Married": [married_val],
        "Education": [education_val],
        "Self_Employed": [self_emp_val],
        "dep_0": [dep_0],
        "dep_1": [dep_1],
        "dep_2": [dep_2],
        "dep_3": [dep_3],
        "Urban": [prop_urban],
        "Rural": [prop_rural],
        "Semiurban": [prop_semiurban]
    }

    df = pd.DataFrame(input_dict)
    return df
