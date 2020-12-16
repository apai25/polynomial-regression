import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Importing the full dataset and storing into Pandas DataFrame
dataset = pd.read_csv('Position_Salaries.csv')

# Splitting into x and y subsets
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# Creating training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_x.fit_transform(x_train)
sc_x.transform(x_test)

sc_y = StandardScaler()
sc_y.fit_transform(y_train)
sc_y.transform(y_test)

# Polynomial transformation of the dataset to 3rd degree
from sklearn.preprocessing import PolynomialFeatures
poly_transform = PolynomialFeatures(degree=3, include_bias=False)
x_poly_train = poly_transform.fit_transform(x_train)
x_poly_test = poly_transform.transform(x_test)

# Creating polynomial regressor
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_poly_train, y_train)

"""
# Non-smooth curve graphing with the training data
plt.scatter(x_train, y_train, color='red', label='Training Data Points')
plt.plot(x_train, regressor.predict(x_poly_train), color='blue', label='Model Curve')
plt.title('Job Level vs Salary (Training Dataset)')
plt.xlabel('Job Level (1 - 10)')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Non-smooth curve graphing with the test data
plt.scatter(x_test, y_test, color='red', label='Test Data Points')
plt.plot(x_test, regressor.predict(x_poly_test), color='blue', label='Model Curve')
plt.title('Job Level vs Salary (Test Dataset)')
plt.xlabel('Job Level (1 - 10)')
plt.ylabel('Salary')
plt.legend()
plt.show()
"""
# Creating x_grid datasets (have smaller increments) for smoother graphing
x_grid_train = np.arange(min(x_train), max(x_train) + .01, step=0.01)
x_grid_train = x_grid_train.reshape(len(x_grid_train), 1)

x_grid_test = np.arange(min(x_test), max(x_test) + .01, step=0.01)
x_grid_test = x_grid_test.reshape(len(x_grid_test), 1)

# Smooth curve graphing with the training data
plt.scatter(x_train, y_train, color='red', label='Training Data Points')
plt.plot(x_grid_train, regressor.predict(poly_transform.transform(x_grid_train)), color='blue', label='Model Curve')
plt.title('Job Level vs Salary (Training Dataset)')
plt.xlabel('Job Level (1 - 10)')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Smooth curve graphing with the test data
plt.scatter(x_test, y_test, color='red', label='Test Data Points')
plt.plot(x_grid_test, regressor.predict(poly_transform.transform(x_grid_test)), color='blue', label='Model Curve')
plt.title('Job Level vs Salary (Test Dataset)')
plt.xlabel('Job Level (1 - 10)')
plt.ylabel('Salary')
plt.legend()
plt.show()
