import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

import matplotlib.pyplot as plt

# Load sample data (digits dataset)
digits = load_digits()
X = digits.data
y = digits.target

# Optionally, use a subset for faster computation
n_samples = 1000
X_subset = X[:n_samples]
y_subset = y[:n_samples]

print("X shape:", X_subset.shape)
print("y shape:", y_subset.shape)
print("X sample:", X_subset[:5])
print("y sample:", y_subset[:5])

# Apply t-SNE
print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_subset)

# Visualize the results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_subset, cmap='tab10', alpha=0.6)
plt.colorbar(scatter, label='Digit Class')
plt.title('t-SNE Visualization of Digits Dataset')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tsne_visualization.png', dpi=300)
print("Visualization saved as 'tsne_visualization.png'")
plt.show()