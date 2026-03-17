import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

X = torch.tensor([[1,2],[2,3],[3,3],[6,5],[7,7],[8,6]], dtype=torch.float32).numpy()
y = torch.tensor([0,0,0,1,1,1], dtype=torch.int64).numpy()

xx, yy = np.meshgrid(np.linspace(0,10,200), np.linspace(0,10,200))
grid = np.c_[xx.ravel(), yy.ravel()]

# ---------------------- K=3 分类曲线图（已包含） ----------------------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
Z = knn.predict(grid).reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Spectral)
plt.scatter(X[:,0], X[:,1], c=y, s=100, edgecolors='k', cmap=plt.cm.Spectral)
plt.title("KNN 分类边界曲线（K=3）")
plt.xlabel("特征 1")
plt.ylabel("特征 2")
plt.grid(True)
plt.show()