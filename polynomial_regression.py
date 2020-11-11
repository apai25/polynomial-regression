import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Importing the full dataset and storing into Pandas DataFrame
dataset = pd.read_csv('Position_Salaries.csv')

# Splitting into x and y subsets
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# No data preprocessing necessary

# Polynomial transformation of the dataset to 4th degree
from sklearn.preprocessing import PolynomialFeatures
poly_transform = PolynomialFeatures(degree=3, include_bias=False)
x_poly = poly_transform.fit_transform(x)

# Creating polynomial regressor
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_poly, y)

# Predicting on the dataset and storing values into 'prediction' variable
predictions = regressor.predict(x_poly)

# Creating x_grid variable (has smaller increments) for smoother graphing
x_grid = np.arange(min(x), max(x) + .01, step=0.01)
x_grid = x_grid.reshape(len(x_grid), 1)

# Graphing the model with the data
plt.scatter(x, y, color='red', label='Actual data')
plt.plot(x_grid, regressor.predict(poly_transform.transform(x_grid)), color='blue', label='Model representation')
plt.title('Job Level vs Salary')
plt.xlabel('Job Level (1 - 10)')
plt.ylabel('Salary')
plt.legend()
plt.show()
