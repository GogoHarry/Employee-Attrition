# Employee Attrition Prediction

## Project Overview

![att](https://github.com/user-attachments/assets/08610e4d-38e2-4bad-8bb3-42a2cfeca8ae)

This project aims to build a predictive model for employee attrition using logistic regression. The model is designed to classify whether an employee is likely to leave the company based on various features such as job satisfaction, work-life balance, and employee demographics.

## Table of Contents
- [Installation](#installation)
- [Data](#data)
- [Modeling](#modeling)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/GogoHarry/employee-attrition-prediction.git
   ```
   
2. Navigate to the project directory:
  ```bash
  cd employee-attrition-prediction
  ```
3. Install the dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Data
The dataset used in this project is the IBM HR Analytics Employee Attrition & Performance dataset, which contains the following columns:
- Age
- BusinessTravel
- Department
- DistanceFromHome
- Education
- EmployeeCount
- EmployeeNumber
- EnvironmentSatisfaction
- Gender
- JobInvolvement
- JobLevel
- JobRole
- JobSatisfaction
- MaritalStatus
- MonthlyIncome
- OverTime
- PerformanceRating
- RelationshipSatisfaction
- StandardHours
- StockOptionLevel
- TotalWorkingYears
- TrainingTimesLastYear
- WorkLifeBalance
- YearsAtCompany
- YearsInCurrentRole
- YearsSinceLastPromotion
- YearsWithCurrManager

The target variable is Attrition.

## Modeling

The following steps outline the modeling process:

1. Data Preprocessing:
-  Redundant features were dropped
- Categorical variables are encoded using one-hot encoding.
- Numerical variables are scaled using MinMaxScaler.

2. Modeling:
- A Logistic Regression Model is trained on the dataset.

3. Evaluation:

The model was evaluated using the following metrics:
- Accuracy: The proportion of correctly classified instances.
- Classification Report: Includes precision, recall, F1-score, and support for each class.
- Confusion Matrix: A summary of prediction results showing the true positives, false positives, true negatives, and false negatives.

## Usage

To run the model and make predictions:

1. Preprocess the data:
  ```python
python preprocess.py
```
2. Train the model:
```python
python train.py
```
3. Run predictions on new data:
```python
 predict.py --input preprocessed_test_data.csv
```

## Results

The model's performance was evaluated using accuracy, cross-validation scores, classification reports, and confusion matrices.
- **Logistic Regression**
  - Accuracy: 88%
  - Cross-Validation Accuracy: 89%

![image](https://github.com/user-attachments/assets/df9bc066-65ea-44ec-b826-8f7a77d3be5e)


## Contributing

We welcome contributions to improve this project. To contribute:

1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Make your changes and commit them (git commit -m 'Add new feature').
4. Push to the branch (git push origin feature-branch).
5. Open a Pull Request.

## License
This project is licensed under the MIT License.

## Contact
For any inquiries or issues, please contact Gogo Harrison at gogoharrison66@gmail.com.

## Acknowledgments

Special thanks to the contributors and the open-source community for their invaluable support.
