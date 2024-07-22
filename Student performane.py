# Import necessary libraries
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the dataset (assuming it's in a file named 'Student_Performance.csv')
df = pd.read_csv('Student_Performance.csv')

# Display the first few rows of the dataset
print(df.head())

# Display summary statistics of the dataset
print(df.describe())

# Display information about the dataset (e.g., data types, non-null counts)
print(df.info())

# Check for missing values in the dataset
print(df.isnull().sum())

# Extract features (X) and target variable (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Convert categorical features to numerical using LabelEncoder
X = np.array(X)
le = LabelEncoder()
for i in range(X.shape[1]):
    if X[:, i].dtype == 'object':
        X[:, i] = le.fit_transform(X[:, i])

# Standardize features using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model and fit it to the training data
reg = LinearRegression()
reg.fit(x_train, y_train)

# Make predictions on the test data
y_pred = reg.predict(x_test)

# Calculate the mean squared error (MSE) between predicted and actual values
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


