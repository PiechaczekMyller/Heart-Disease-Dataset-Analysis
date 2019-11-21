import numpy as np
import pandas as pd

LABELS_ID = 13

df = pd.read_csv("./processed.cleveland.data", header=None)
for column in df.columns:
    df = df[~df[column].isin(['?'])]
df = df.drop([1, 3, 11, 12], axis=1)
stats = []
dfs = []
for label in df[LABELS_ID].unique():
    sub_df = df[df[LABELS_ID] == label]
    count = len(sub_df)
    print(count)
    mean = sub_df.mean()
    median = sub_df.median()
    std = sub_df.std()
    min = sub_df.min()
    max = sub_df.max()
    stat_df = pd.concat([mean, median, std, min, max], axis=1,
                        keys=["mean", "median", "std", "min", "max"])
    dfs.append(stat_df)
