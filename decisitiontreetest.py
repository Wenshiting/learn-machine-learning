from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,f1_score)
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
iris = load_iris()
iris_X = iris.data   #x有4个属性，共有600个样本点
iris_y = iris.target #y的取值有3个，分别是0,1,2
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(clf.predict(X_test))
print(y_test)
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred, average="micro"))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred, average="micro"))
print("\tF1: %1.3f\n" % f1_score(y_test, y_pred, average="micro"))
train_sizes, train_loss, test_loss = learning_curve( clf, X_train, y_train, cv=10,
    train_sizes=[0.1, 0.25, 0.5, 0.75, 1])
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)
plt.plot(train_sizes, train_loss_mean, 'o-', color="r",
         label="Training")
plt.plot(train_sizes, test_loss_mean, 'o-', color="g",
        label="Cross-validation")

plt.xlabel("Training examples")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()
