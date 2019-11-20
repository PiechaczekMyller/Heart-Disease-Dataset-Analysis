import numpy as np
import pandas as pd


def load_numpy(path_to_file: str):
    """
    Load the dataset into two container: one with the actual data and the second
    one with the respective labels
    :param path_to_file: Path to the .data file
    :return: numpy array with the data, numpy array with the labels
    """
    with open(path_to_file) as file:
        df = pd.read_csv(file, header=0, na_values='?')
    labels = df.iloc[:, -1]
    df = df.fillna(0)
    df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)
    return np.array(df, dtype=np.float32), np.array(labels, dtype=np.uint8)


def load_dataframe(path_to_file: str):
    """
    Load the dataset into two container: one with the actual data and the second
    one with the respective labels
    :param path_to_file: Path to the .data file
    :return: numpy array with the data, numpy array with the labels
    """
    with open(path_to_file) as file:
        df = pd.read_csv(file, header=0, na_values='?')
    labels = df.iloc[:, -1]
    df = df.fillna(0)
    df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)
    return df, labels