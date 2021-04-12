from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error

dataset = load_boston()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=57)
reg = LinearRegression().fit(X_train, y_train)
#y_pred = reg.predict(X_test) -- i don't need this as reg.score(X_test, y_test) means the same

train_accuracy = reg.score(X_train, y_train)
test_accuracy = reg.score(X_test, y_test)

print("Training Accuracy: ", train_accuracy)
print("Testing Accuracy: ", test_accuracy)

y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: ", mse)