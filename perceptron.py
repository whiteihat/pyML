# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: myml
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 线性分类算法：感知器实现正负样本分类

# %%
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 显示中文
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

from utils.path import project_root

# %% [markdown]
# ## 1. 加载数据

# %%
PROJECT_ROOT = project_root().as_posix()
POS_PATH = os.path.join(PROJECT_ROOT, "datasets", "positive.mat")
NEG_PATH = os.path.join(PROJECT_ROOT, "datasets", "negative.mat")

with h5py.File(POS_PATH, "r") as f:
    pos = np.array(f["positive"]).T
with h5py.File(NEG_PATH, "r") as f:
    neg = np.array(f["negtive"]).T

X = np.vstack([pos, neg])
y = np.hstack([np.ones(pos.shape[0]), -np.ones(neg.shape[0])])

print(f"正样本数: {pos.shape[0]}, 负样本数: {neg.shape[0]}")

# %% [markdown]
# ## 2. 感知器算法实现


# %%
def perceptron(X, y, lr=1.0, epochs=1000):
    X_ = np.c_[np.ones(X.shape[0]), X]  # 添加偏置
    w = np.zeros(X_.shape[1])
    for epoch in range(epochs):
        errors = 0
        for xi, yi in zip(X_, y):
            if yi * (w @ xi) <= 0:
                w += lr * yi * xi
                errors += 1
        if errors == 0:
            break
    return w


# %%
w = perceptron(X, y, lr=1.0, epochs=1000)
print(f"训练得到的权重: {w}")

# %% [markdown]
# ## 3. 分类结果与决策边界可视化

# %%
plt.figure(figsize=(7, 5))
plt.scatter(pos[:, 0], pos[:, 1], color="b", label="正样本")
plt.scatter(neg[:, 0], neg[:, 1], color="r", label="负样本")

# 决策边界: w0 + w1*x1 + w2*x2 = 0
x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
x2 = -(w[0] + w[1] * x1) / w[2]
plt.plot(x1, x2, "g--", label="决策边界")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("感知器分类与决策边界")
plt.legend()
plt.show()
