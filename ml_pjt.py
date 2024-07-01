#Project_NAME=FLower classification project

# import libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
Iris = load_iris()
data = pd.DataFrame(data= np.c_[Iris['data'], Iris['target']], columns= Iris['feature_names'] + ['target'])
print(data.head())
print(data.head())
print(data.head())
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=1)
Scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)
X_test = Scaler.transform(X_test)
Knn = KNeighborsClassifier(n_neighbors=5)
Knn.fit(X_train, y_train)
Y_pred = Knn.predict(X_test)
print(Y_pred)
Cm = confusion_matrix(y_test, Y_pred)
print(Cm)
# Classification Report
Cr = classification_report(y_test, Y_pred)
print(Cr)
# Accuracy Score
Accuracy = accuracy_score(y_test, Y_pred)
print(f"Accuracy: {Accuracy: .2f}")
#visualize result
plt.figure(figsize=(10,5))
sns.heatmap(Cm, annot=True, fmt="d", cmap='Blues')
plt.xlabel('predicted')
plt.ylabel('Actual')
plt.title('confusion matrix')
plt.show()



