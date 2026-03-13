import torch

# 定义两个 2x2 矩阵
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# 计算矩阵乘法
c = torch.matmul(a, b)
print(c)