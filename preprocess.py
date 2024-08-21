import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Load data
data = pd.read_csv('train_data.csv')

# Drop off some redundant features
data.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours','MonthlyIncome', 'MonthlyRate'], axis=1, inplace=True)

# create a copy of the dataset for model building
df = data.copy()

# Preprocess data
scaler = MinMaxScaler()
encoded_data = pd.get_dummies(df)

scaled_data = pd.DataFrame(scaler.fit_transform(encoded_data), columns=encoded_data.columns)

scaled_data.to_csv('preprocessed_data.csv', index=False)
