from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from itertools import product


from utils import load_numpy

FEATURE_1 = 0
FEATURE_2 = 4
colors = ['red', 'green', 'blue', 'yellow']

data, labels = load_numpy(r"processed.cleveland.data")

# Normalize data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
no_of_features = data.shape[1]

for feature_1, feature_2 in product(range(no_of_features), range(no_of_features)):
    fig, ax = plt.subplots()
    for label in [0, 1, 2, 3]:
        ax.scatter(data[labels == label, feature_1], data[labels == label, feature_2], label='Class {}'.format(label),
                   c=colors[label])
    ax.legend()
    plt.savefig(r'images\scatter_correlation\feature{}_feature{}.png'.format(feature_1, feature_2))
    plt.close(fig)

