# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 17:51:23 2021

@author: Qalbe
"""

#imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



#reading file
file = pd.read_csv("homeprices.csv")
file


#diving into x and y
X=file.iloc[:,1].values.reshape(-1, 1)
y=file.iloc[:,:-1].values



# sckit-learn implementation

# Model initialization
regression_model = LinearRegression()

# Fit the data(train the model)
regression_model.fit(X, y)



# Predict
y_predicted = regression_model.predict(X)



# model evaluation
rms = mean_squared_error(y, y_predicted)
r2 = r2_score(y, y_predicted)


# printing values
print('Slope:' ,regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('Root mean squared error: ', rms)
print('R2 score: ', r2)


# plotting values
plt.plot(X, y, color = 'g')
plt.xlabel('x')
plt.ylabel('y')


# predicted values
plt.plot(X, y_predicted, color='r')
plt.show()


# predicted values, just checking difference through graph
plt.scatter(y,y_predicted, color = 'purple',  marker = '*')
plt.show()


