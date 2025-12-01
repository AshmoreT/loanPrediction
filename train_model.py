
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("../data/loan_dataset.csv")

df.dropna(inplace=True)

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
df['Dependents'] = df['Dependents'].map({'0': [1, 0, 0, 0], '1': [0, 1, 0, 0], '2': [0, 0, 1, 0], '3+': [0, 0, 0, 1]})
df['Property_Area'] = df['Property_Area'].map({'Urban': [1, 0, 0], 'Rural': [0, 1, 0], 'Semiurban': [0, 0, 1]})

dependents_df = pd.DataFrame(df['Dependents'].tolist(), columns=['dep_0', 'dep_1', 'dep_2', 'dep_3'])
property_df = pd.DataFrame(df['Property_Area'].tolist(), columns=['Urban', 'Rural', 'Semiurban'])

X = pd.concat([
    df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
        'Gender', 'Married', 'Education', 'Self_Employed']],
    dependents_df,
    property_df
], axis=1)

y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open("../model/Random_Forest.sav", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved.")
