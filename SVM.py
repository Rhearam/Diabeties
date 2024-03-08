import pandas as pd
import random
df = pd.read_csv("diabetes.csv")
df.to_csv("train.csv", header=False, index=False)
dataset = pd.read_csv("train.csv")
X = dataset.iloc[:, 0:8].values

Y = dataset.iloc[:, 8].values
l=pd.unique(dataset.iloc[:,8])
pred=random.choice(l)

from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, Y_train)
pred=random.choice(l)

Y_pred = classifier.predict(X_test)
print(pred)


from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(Y_test, Y_pred)
print("\n",cm)

print(classification_report(Y_test,Y_pred))

iclf = SVC(kernel='linear', C=1).fit(X_train, Y_train)
#print(iclf)
accuracy2=((iclf.score(X_test, Y_test))*100)
print("accuracy=",accuracy2)

import matplotlib.pyplot as plt

x = [0, 1, 2]
y = [accuracy2, 0, 0]
plt.title('Accuracy2')
plt.bar(x, y)
plt.show()
