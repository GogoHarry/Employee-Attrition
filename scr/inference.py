# load the unseen test dataset
test_data = pd.read_excel("test.xlsx")

# Apply transforms to the new data similar to the training dataset
test_data.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)

# # Separate numerical and categorical columns of the unseen test_data
num_var = test_data.select_dtypes(include=['int64', 'float64']).columns
cat_var = test_data.select_dtypes(include=['object', 'category']).columns

# Scale numerical features of the unseen test_data
scaler = MinMaxScaler()
num_var_scaled = pd.DataFrame(scaler.fit_transform(test_data[num_var]), columns=num_var)

# Encode categorical features of the unseen test_data
encoder = OneHotEncoder(drop='first', sparse=False)
cat_var_encoded = pd.DataFrame(encoder.fit_transform(test_data[cat_var]), columns=encoder.get_feature_names_out(cat_var))

# Combine scaled numerical and encoded categorical features of the unseen test_data
test_data_preprocessed = pd.concat([num_var_scaled, cat_var_encoded], axis=1)

# Save preprocessed unseen test_data
test_data_preprocessed.to_csv('test_data_preprocessed_data.csv', index=False)

# Load the trained model
model = joblib.load('logistic_model.pkl')

# Make predictions on the unseen data
predictions = model.predict(test_data_preprocessed)

test_data['Predicted_Attrition'] = predictions

# Display the DataFrame with predictions
print(test_data.head())
