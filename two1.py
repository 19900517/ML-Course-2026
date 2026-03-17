import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 构造数据
data = torch.tensor([1., 1., 0., 1., 0.])
N = len(data)
sum_x = torch.sum(data)

# 2. 计算 MLE
p_mle = torch.mean(data)

# 3. 计算 MAP (Beta(2,2) 先验)
alpha = 2
beta_p = 2
p_map = (sum_x + alpha - 1) / (N + alpha + beta_p - 2)

# 4. 输出结果
print("="*30, "MLE & MAP 计算结果", "="*30)
print(f"样本数量: {N}, 样本中1的个数: {sum_x.item()}")
print(f"MLE估计p值: {p_mle.item():.4f}")
print(f"MAP估计p值: {p_map.item():.4f}")

# 5. 可视化
p = np.linspace(0, 1, 100)
sum_x_np = sum_x.item()

likelihood = np.power(p, sum_x_np) * np.power(1-p, N - sum_x_np)
prior = beta.pdf(p, alpha, beta_p)
posterior = np.power(p, sum_x_np + alpha - 1) * np.power(1-p, N - sum_x_np + beta_p - 1)

plt.figure(figsize=(10, 6))
plt.plot(p, likelihood, label="Likelihood（似然）", color='blue', linewidth=2)
plt.plot(p, prior, label="Prior（Beta(2,2)先验）", color='orange', linewidth=2)
plt.plot(p, posterior, label="Posterior（后验）", color='green', linewidth=2)

plt.axvline(p_mle.item(), color='red', linestyle='--', label=f"MLE = {p_mle.item():.4f}")
plt.axvline(p_map.item(), color='purple', linestyle='--', label=f"MAP = {p_map.item():.4f}")

plt.title("MLE vs MAP (伯努利分布+Beta先验)", fontsize=14)
plt.xlabel("参数p", fontsize=12)
plt.ylabel("概率/密度值", fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show(block=True)