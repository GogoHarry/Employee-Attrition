from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold

# Load the test data
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').values.ravel()

# Load the trained model
model = joblib.load('logistic_model.pkl')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on Test Set: {accuracy:.2f}')

# Generate classification report
report = classification_report(y_test, y_pred)
print('Classification Report:')
print(report)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_test, y_test, cv=kf, scoring='accuracy')

print(f'K-Fold Cross-Validation Scores: {cv_scores}')
print(f'Mean CV Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}')

# Explanation of K-Fold Cross-Validation Score
print("""
K-Fold Cross-Validation is used to evaluate the robustness of the model by splitting the data into k subsets (folds).
The model is trained on k-1 folds and tested on the remaining fold. This process is repeated k times, and the 
performance metric (accuracy in this case) is calculated for each iteration. The mean and standard deviation 
of the accuracy scores provide a more reliable estimate of the model's performance and its ability to generalize 
to unseen data.
""")
