import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load the preprocessed dataset
df = pd.read_csv('preprocessed_data.csv')

# Define the target variable and features
target = 'Attrition'  # Assuming 'Attrition' was one of the categorical columns
X = df.drop(columns=[target])
y = df[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'logistic_model.pkl')

# Save the test set for evaluation
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
