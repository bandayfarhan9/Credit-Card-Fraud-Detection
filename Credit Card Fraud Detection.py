import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
credit_card_data = pd.read_csv(r'C:\Users\Downloads\creditcard.csv')

# Check the first 5 rows
print(credit_card_data.head())

# Check the last 5 rows
print(credit_card_data.tail())

# Check dataset information
print(credit_card_data.info())

# Drop columns with any missing value
cleaned_data = credit_card_data.dropna()

# Verify the result
print(cleaned_data.isnull().sum())

# Class distribution
print(credit_card_data['Class'].value_counts())

# Separate the legitimate and fraudulent transactions
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
print(legit.shape)
print(fraud.shape)

# Describe the amounts in legitimate and fraudulent transactions
print(legit.Amount.describe())
print(fraud.Amount.describe())

# Group by class and calculate the mean for each group
print(credit_card_data.groupby('Class').mean())

# Take a random sample of legitimate transactions
legit_sample = legit.sample(n=356)

# Concatenate the sample of legitimate transactions with all the fraudulent transactions
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Print the first 5 rows of the new dataset
print(new_dataset.head())

# Print the last 5 rows of the new dataset
print(new_dataset.tail())

# Class distribution in the new dataset
print(new_dataset['Class'].value_counts())

# Group by class and calculate the mean for each group in the new dataset
print(new_dataset.groupby('Class').mean())

# Separate the features and target variable
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

print(X.head())
print(Y.head())

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X_train.shape, X_test.shape)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, Y_train)

# Make predictions on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy on training data:", training_data_accuracy)

# Make predictions on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy on test data:", test_data_accuracy)
