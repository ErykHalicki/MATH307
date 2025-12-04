import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Purpose of this file is to experiment / try out the normal equation for linear regression
# Final_data.csv is a random dataset I found on kaggle, which i will try to fit using numpy and the normal equation

data = pd.read_csv('Final_data.csv')

# There appears to be a fairly strong linear correlation between Fat percentage and BMI
# Good candidate for the model

# Normal equation states: (AtA)x^ = Atb
# Where x^ is the least squares solution to Ax=b
# We can also look at the equation as XtXB^ = XtY, where X are our inputs, B are our coefficients, and Y are our outputs

X = data['Fat_Percentage'].to_numpy()
Y = data['BMI'].to_numpy()

# Turning X and Y into column vectors
X = X.reshape((X.shape[0]),1)
Y = Y.reshape((Y.shape[0]),1) 

# Padding X with ones (for intercept / bias term)
X = np.hstack((np.ones((X.shape[0],1)),X))

# Now our data should be ready for fitting!

Bhat = np.linalg.solve(X.T@X, X.T@Y) # Solving for XtXB^ = XtY

# Now we plot the predicted line
x_line = np.linspace(X[:,1].min(), X[:,1].max(), 100)
x_actual = X[:,1]
x_line_matrix = x_line.reshape((x_line.shape[0],1))
x_line_matrix = np.hstack((np.ones((x_line_matrix.shape[0],1)),x_line_matrix))
y_line = (x_line_matrix@Bhat).flatten()
y_actual = Y.flatten()

plt.scatter(x_actual,y_actual, s=5)
plt.plot(x_line, y_line, 'r-', linewidth=2, label='Regression Line')
plt.xlabel('Fat Percentage')
plt.ylabel('BMI')
plt.show()

