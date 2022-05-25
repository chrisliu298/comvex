import matplotlib.pyplot as plt
import torch
from sklearn import decomposition

plt.rcParams.update({"font.size": 14})

embedding_file = "cnnzoo1-cifar10gs-comvex-linear-embeddings"
data = torch.load(embedding_file + ".pt")
X = data["embeddings"].detach().numpy()
y = data["true_acc"].numpy()
y_pred = data["pred_acc"].detach().numpy()

# pca = decomposition.PCA(n_components=2)
# pca.fit(X)
# X_ = pca.transform(X)

# fig, ax = plt.subplots(figsize=(12, 8))
# plt.figure(figsize=(12, 8))
# scatter = plt.scatter(X_[:, 0], X_[:, 1], c=y, cmap="magma")
# cbar = plt.colorbar()
# cbar.set_label("test_acc")
# plt.title(embedding_file)
# plt.show()

pca = decomposition.PCA(n_components=3)
pca.fit(X)
X_ = pca.transform(X)

fig = plt.figure(1, figsize=(12, 8))
plt.clf()
ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
p = ax.scatter(X_[:, 0], X_[:, 1], X_[:, 2], c=y, cmap="magma")
ax.view_init(elev=10, azim=160)
fig.colorbar(p)
plt.title(embedding_file)
plt.show()
