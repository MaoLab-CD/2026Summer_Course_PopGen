import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(42)
n = 30
p = 200

p1 = np.random.beta(0.8, 0.8, p)
p2 = np.clip(p1 + np.random.normal(0.12, 0.05, p), 0.01, 0.99)
p3 = np.clip(p1 - np.random.normal(0.10, 0.05, p), 0.01, 0.99)

def sim(freqs):
    return np.random.binomial(2, freqs, size=(n, len(freqs)))

X1 = sim(p1)
X2 = sim(p2)
X3 = sim(p3)
X = np.vstack([X1, X2, X3]).astype(float)
labels = np.array(["Pop1"]*n + ["Pop2"]*n + ["Pop3"]*n)

X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
X = np.nan_to_num(X)
pcs = PCA(n_components=2).fit_transform(X)

for pop in np.unique(labels):
    idx = labels == pop
    plt.scatter(pcs[idx, 0], pcs[idx, 1], label=pop)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Toy Genotype Data")
plt.legend()
plt.tight_layout()
plt.savefig("pca_toy_genotype_plot.png", dpi=300)
plt.show()
