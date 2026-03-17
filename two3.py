import torch
import numpy as np
import matplotlib.pyplot as plt

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 基础梯度下降计算 =====================
x = torch.tensor([5.0], requires_grad=True)
lr = 0.1
epochs = 30
loss_history = []
x_path = []

for i in range(epochs):
    y = x**2 + 2*x + 1
    loss_history.append(y.item())
    x_path.append(x.item())
    
    y.backward()
    with torch.no_grad():
        x -= lr * x.grad
    x.grad.zero_()

print("===== 梯度下降最终结果 =====")
print(f"最优 x = {x.item():.4f}")
print(f"最小损失 = {y.item():.4f}")

# ===================== 图1：损失函数下降曲线 =====================
plt.figure(figsize=(10, 4))
plt.plot(range(epochs), loss_history, 'o-r', linewidth=2, label="损失曲线")
plt.title("图1：梯度下降 — 损失变化曲线", fontsize=14)
plt.xlabel("迭代次数")
plt.ylabel("Loss 值")
plt.grid(True)
plt.legend()

# ===================== 图2：目标函数曲线 + 优化路径 =====================
x_vals = np.linspace(-6, 6, 100)
y_vals = x_vals**2 + 2 * x_vals + 1

plt.figure(figsize=(10, 4))
plt.plot(x_vals, y_vals, 'b-', linewidth=2, label="目标函数 f(x) = x²+2x+1")
plt.scatter(x_path, [xx**2 + 2*xx +1 for xx in x_path], 
            c="red", s=100, label="优化路径")
plt.title("图2：梯度下降 — 函数曲线与优化路径", fontsize=14)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()

# ===================== 图3：不同学习率对比曲线 =====================
def run_gd(lr, epochs=30):
    x = torch.tensor([5.0], requires_grad=True)
    hist = []
    for _ in range(epochs):
        y = x**2 + 2*x +1
        hist.append(y.item())
        y.backward()
        with torch.no_grad():
            x -= lr * x.grad
        x.grad.zero_()
    return hist

lrs = [0.01, 0.1, 0.5]
plt.figure(figsize=(10,4))
for lr in lrs:
    hist = run_gd(lr)
    plt.plot(range(len(hist)), hist, 'o-', linewidth=2, label=f"lr={lr}")

plt.title("图3：梯度下降 — 不同学习率收敛对比", fontsize=14)
plt.xlabel("迭代次数")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.show()