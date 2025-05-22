import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_excel("Copy of loan.xlsx")

# Fill missing values without chained assignment warnings
fill_values = {
    'Gender': df['Gender'].mode()[0],
    'Married': df['Married'].mode()[0],
    'Dependents': df['Dependents'].mode()[0],
    'Self_Employed': df['Self_Employed'].mode()[0],
    'LoanAmount': df['LoanAmount'].mean(),
    'Loan_Amount_Term': df['Loan_Amount_Term'].mode()[0],
    'Credit_History': df['Credit_History'].mode()[0]
}
df.fillna(fill_values, inplace=True)

# Log transformations for LoanAmount and TotalIncome
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])

# Encoding categorical features
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

for col in categorical_columns:
    df[col] = df[col].astype(str)
    df[col] = label_encoder.fit_transform(df[col])

# Features and target variable
X = df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
        'Loan_Amount_Term', 'Credit_History', 'Property_Area']].values  # Features

y = df['Loan_Status'].values  # Target variable

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scaling features (only after encoding)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy}")

print(y_pred)

# Visualizations

# 1. Distribution of Loan Amount (Log transformed)
plt.figure(figsize=(8, 6))
df['LoanAmount_log'].hist(bins=20)
plt.title('Log Transformed LoanAmount')
plt.xlabel('LoanAmount_log')
plt.ylabel('Frequency')
plt.show()

# 2. Distribution of Total Income (Log transformed)
plt.figure(figsize=(8, 6))
df['TotalIncome_log'].hist(bins=20)
plt.title('Log Transformed TotalIncome')
plt.xlabel('TotalIncome_log')
plt.ylabel('Frequency')
plt.show()

# 3. Gender Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', data=df)
plt.title('Distribution of Loan Applicants by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# 4. Married Status Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Married', data=df)
plt.title('Distribution of Loan Applicants by Married Status')
plt.xlabel('Married')
plt.ylabel('Count')
plt.show()

# 5. Dependents Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Dependents', data=df)
plt.title('Distribution of Loan Applicants by Dependents')
plt.xlabel('Dependents')
plt.ylabel('Count')
plt.show()

# 6. Self Employed Status Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Self_Employed', data=df)
plt.title('Distribution of Loan Applicants by Self Employed Status')
plt.xlabel('Self_Employed')
plt.ylabel('Count')
plt.show()

# 7. Loan Amount Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['LoanAmount'], kde=True, color='blue')
plt.title('Distribution of Loan Amount')
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.show()

# 8. Credit History Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Credit_History', data=df)
plt.title('Distribution of Loan Applicants by Credit History')
plt.xlabel('Credit History')
plt.ylabel('Count')
plt.show()

# 9. Loan Status Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Loan_Status', data=df)
plt.title('Distribution of Loan Status (Approved/Not Approved)')
plt.xlabel('Loan_Status')
plt.ylabel('Count')
plt.show()
