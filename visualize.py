from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

df = pd.read_csv("./processed.cleveland.data", header=None)
for column in df.columns:
    df = df[~df[column].isin(['?'])]

df[1] = pd.to_numeric(df[1])
df[2] = pd.to_numeric(df[2])
df[11] = pd.to_numeric(df[11])
df[12] = pd.to_numeric(df[12])

# df[df[13].isin([1, 2, 3, 4])] = 1
X, y = np.array(df.drop(13, axis=1), dtype=np.float32), np.array(df[13], dtype=np.uint8)

first_column = 7
second_column = 8

for label in range(5):
    plt.scatter([point[first_column] for point in X[y == label]],
                [point[second_column] for point in X[y == label]])
plt.show()
