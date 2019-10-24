import pandas as pd


with open(r"C:\Users\MMyller\Downloads\processed.cleveland.data") as file:
    df = pd.read_csv(file, header=0)
labels = df.iloc[:, -1]
df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)
data_frame = df

