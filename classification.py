from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("./processed.cleveland.data", header=None)
for column in df.columns:
    df = df[~df[column].isin(['?'])]

df[1] = pd.to_numeric(df[1])
df[2] = pd.to_numeric(df[2])
df[11] = pd.to_numeric(df[11])
df[12] = pd.to_numeric(df[12])

df[df[13].isin([1, 2, 3, 4])] = 1
X, y = np.array(df.drop(13, axis=1), dtype=np.float32), np.array(df[13], dtype=np.uint8)

skf = StratifiedKFold(n_splits=5)

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    print(classification_report(y_test, clf.predict(X_test)))
    print(confusion_matrix(y_test, clf.predict(X_test)))
