# Credit Card Fraud Detection

This project demonstrates a simple implementation of credit card fraud detection using Logistic Regression. The dataset used for this project is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle.

## Dataset

The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly imbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.

## Project Workflow

1. **Load the dataset**: The dataset is loaded using Pandas.
2. **Exploratory Data Analysis (EDA)**: Initial exploration of the dataset including checking for null values, understanding class distribution, and descriptive statistics.
3. **Data Cleaning**: Handling any missing values and understanding the distribution of the target variable.
4. **Data Preparation**: Creating a balanced dataset by undersampling the legitimate transactions.
5. **Feature and Target Separation**: Separating the features and target variable.
6. **Train-Test Split**: Splitting the data into training and testing sets.
7. **Model Training**: Training a Logistic Regression model.
8. **Model Evaluation**: Evaluating the model's performance on training and test data.

## Installation

To run this project, you will need Python and the following libraries installed:

- pandas
- numpy
- scikit-learn

You can install the required libraries using the following command:

```bash
pip install pandas numpy scikit-learn
