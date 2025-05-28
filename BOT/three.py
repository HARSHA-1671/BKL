import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
iris = load_iris()
data = iris.data
labels = iris.target
label_names = iris.target_names
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)
colors = ['r', 'g', 'b']
for i, label_name in enumerate(label_names):
    plt.scatter(
        reduced_data[labels == i, 0],
        reduced_data[labels == i, 1],
        label=label_name,
        color=colors[i]
    )
plt.title('PCA on Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()
plt.show()