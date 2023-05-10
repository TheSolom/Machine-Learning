import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from ML_Liner_Regression import LinearRegression

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#fig = plt.figure(figsize=(8,6))
#plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
#plt.show()

reg = LinearRegression(0.01)
reg.fit(X_train,y_train) #return right weigth and bias

predictions = reg.predict(X_test) #return predicted values from tests

def mse(y_test, predictions):
    return np.mean((y_test - predictions)**2) #calculate error between tests and predictions 
mse = mse(y_test, predictions)
print(mse)

y_pred_line = reg.predict(X)
#fig = plt.figure(figsize=(8,6))
plt.scatter(X_train, y_train, s=10)
plt.scatter(X_test, y_test, s=10)
plt.plot(X, y_pred_line, color='black', label='Prediction')
plt.show()