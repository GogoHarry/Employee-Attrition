import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Load data
data = pd.read_csv('train_data.csv')

# Drop off some redundant feature
data.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours','MonthlyIncome', 'MonthlyRate'], axis=1, inplace=True)

# create a copy of the dataset for model building
df = data.copy()

# Separate numerical and categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns

# Scale numerical features
scaler = MinMaxScaler()
df_num_scaled = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)

# Encode categorical features
encoder = OneHotEncoder(drop='first', sparse=False)
df_cat_encoded = pd.DataFrame(encoder.fit_transform(df[cat_cols]), columns=encoder.get_feature_names_out(cat_cols))

# Combine scaled numerical and encoded categorical features
df_preprocessed = pd.concat([df_num_scaled, df_cat_encoded], axis=1)

# Save preprocessed data
df_preprocessed.to_csv('preprocessed_data.csv', index=False)
