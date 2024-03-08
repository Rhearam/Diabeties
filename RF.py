import pandas as pd
import numpy as np
import random
df = pd.read_csv("diabetes.csv")
df.to_csv("train.csv", header=False, index=False)
dataset = pd.read_csv("train.csv")
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values
l=pd.unique(dataset.iloc[:,8])
pred=random.choice(l)
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(y_pred)
print(pred)
from sklearn import metrics
errors=metrics.mean_absolute_error(y_test, y_pred)
print("errors",errors)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("ytest",np.mean(y_test))

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / np.mean(y_test))# Calculate and display accuracy

print("mape",mape)
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')

import matplotlib.pyplot as plt

x = [0, 1, 2]
y = [accuracy, 0, 0]
plt.title('Accuracy')
plt.bar(x, y)
plt.show()
