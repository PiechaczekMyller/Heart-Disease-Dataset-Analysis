from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

from utils import load_numpy, load_dataframe

data, labels = load_dataframe(r"processed.cleveland.data")

# Normalize data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Visualize using PCA
data_after_pca = PCA(n_components=2).fit_transform(data)
fig, ax = plt.subplots()
ax.scatter(data_after_pca[labels == 0, 0], data_after_pca[labels == 0, 1], c='red', label='Class 1 (No disease)')
ax.scatter(data_after_pca[labels > 0, 0], data_after_pca[labels > 0, 1], c='blue', label='Class 2 (Some kind of disease)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
ax.legend()
plt.show()




