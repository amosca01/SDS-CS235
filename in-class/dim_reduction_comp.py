# dimension_reduction_comparison.py

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from prince import MCA
from sklearn.preprocessing import StandardScaler

# Load a sample dataset (you can replace this with your own dataset)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Standardize the data for methods that are sensitive to scale
X_std = StandardScaler().fit_transform(X)

# Initialize dimension reduction methods
methods = [
    ("PCA", PCA(n_components=2)),
    ("t-SNE", TSNE(n_components=2, random_state=42)),
    ("Isomap", Isomap(n_components=2)),
    ("LLE", LocallyLinearEmbedding(n_components=2)),
    ("LDA", LinearDiscriminantAnalysis(n_components=2)),
    ("Factor Analysis", FactorAnalysis(n_components=2)),
    ("MDS", MDS(n_components=2)),
]

# Create a 3x3 plot
plt.figure(figsize=(15, 15))

for i, (name, method) in enumerate(methods, 1):
    plt.subplot(3, 3, i)

    if name == "LDA":
        # For supervised dimensionality reduction techniques, pass both X and y
        X_transformed = method.fit_transform(X_std, y)
    else:
        X_transformed = method.fit_transform(X)

    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y, cmap='viridis', edgecolors='k')
    plt.title(name)

# Add your code to apply Autoencoder here
plt.subplot(3, 3, 8)  # Adjust the position in the grid
# X_autoencoder_transformed = ...

# Add your code to apply MCA here
plt.subplot(3, 3, 9)  # Adjust the position in the grid
# X_mca_transformed = ...

plt.tight_layout()
plt.show()