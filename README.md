# Task 4: Classification with Logistic Regression - Elevate Labs AI & ML Internship

## Objective
Build and evaluate a **binary classification model** using Logistic Regression.  
We will use the **Breast Cancer Wisconsin Dataset** from Kaggle to classify tumors as *malignant* or *benign*.

## Tools & Libraries
- **Python**
- **Pandas** – Data loading and preprocessing
- **NumPy** – Numerical computations
- **Matplotlib** – Data visualization
- **Seaborn** – Enhanced plots
- **Scikit-learn** – Model training and evaluation

## Steps Followed
1. **Dataset Selection & Loading**  
   - Used the [Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) from Kaggle.  
   - Loaded data with Pandas and inspected for missing values.

2. **Preprocessing**  
   - Dropped irrelevant columns (e.g., ID).
   - Standardized features using `StandardScaler` for better model performance.
   - Encoded target labels (*M* → 1, *B* → 0).

3. **Train-Test Split**  
   - Split dataset into **80% training** and **20% testing** using `train_test_split`.

4. **Model Training**  
   - Fitted a `LogisticRegression` model from `sklearn.linear_model`.

5. **Model Evaluation**  
   - **Confusion Matrix** – to analyze predictions.
   - **Precision** and **Recall** – for measuring classification performance.
   - **ROC Curve** & **AUC Score** – to evaluate model discriminative ability.
   - **Threshold Tuning** – adjusted decision threshold for better recall/precision trade-off.

6. **Sigmoid Function Explanation**  
   - Explained how the sigmoid function maps linear combinations of features to probabilities.

## Results
- **Accuracy**: *value*
- **Precision**: *value*
- **Recall**: *value*
- **ROC-AUC**: *value*
- Model achieved high classification performance on the test set.



## Author
Abhishek Pandey
