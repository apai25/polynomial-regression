import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Importing the full dataset and storing into Pandas DataFrame
dataset = pd.read_csv('Position_Salaries.csv')

# Splitting into x and y subsets
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

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
plt.plot(x_grid, regressor.predict(poly_transform.transform(x_grid)), color='blue', label='Model Representation')
plt.title('Job Level vs Salary')
plt.xlabel('Job Level (1 - 10)')
plt.ylabel('Salary')
plt.legend()
plt.show()
