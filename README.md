# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 


```
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: 
RegisterNumber:  

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load California housing dataset
data = fetch_california_housing()

# Features: take first 3 columns
X = data.data[:, :3]

# Targets: original target + column 6
Y = np.column_stack((data.target, data.data[:, 6]))

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Standardize features and targets
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

# Multi-output regression using SGD
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train, Y_train)

# Predictions
Y_pred = multi_output_sgd.predict(X_test)

# Inverse transform to original scale
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)

# Evaluate
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Square Error:", mse)
print("\nPredictions:\n", Y_pred[:5])

## Output:
Mean Square Error: 2.5820450143438927

Predictions:
 [[ 0.99283845 35.72104929]
 [ 1.49723021 35.69667941]
 [ 2.23676114 35.59869162]
 [ 2.72178967 35.44098348]
 [ 2.11089765 35.61969573]]
​


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
