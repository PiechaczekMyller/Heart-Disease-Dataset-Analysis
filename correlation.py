import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

data = pd.read_csv("./only_ill.csv", header=None)
data.columns = ["age", "sex", "cp", "trestbps",
                "chol", "fbs", "restecg", "thalach",
                "exang", "oldpeak", "slope", "ca", "thal", "label"]
sns.pairplot(data)
plt.show()


corr = data.corr()
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200)
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.show()
data.hist()
plt.show()
sns.pairplot(data)
plt.show()
