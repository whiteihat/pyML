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
# # 线性回归：梯度下降法预测学生人数

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
YEAR_PATH = os.path.join(PROJECT_ROOT, "datasets", "year.mat")
STU_NUM_PATH = os.path.join(PROJECT_ROOT, "datasets", "stu_num.mat")

with h5py.File(YEAR_PATH, "r") as f:
    years = np.array(f["year"]).T.ravel()
with h5py.File(STU_NUM_PATH, "r") as f:
    stu_num = np.array(f["stu_num"]).T.ravel()

print(f"年份数据: {years}")
print(f"学生人数数据: {stu_num}")

# %% [markdown]
# ## 2. 数据预处理

# %%
# 特征归一化（有助于梯度下降收敛）
years_mean = years.mean()
years_std = years.std()
years_norm = (years - years_mean) / years_std

# 添加偏置项
X = np.c_[np.ones_like(years_norm), years_norm]
y = stu_num

# %% [markdown]
# ## 3. 梯度下降法实现线性回归


# %%
def compute_loss(X, y, theta):
    y_pred = X @ theta
    return ((y_pred - y) ** 2).mean() / 2


def gradient_descent(X, y, lr=0.01, epochs=1000):
    theta = np.zeros(X.shape[1])
    loss_history = []
    for i in range(epochs):
        y_pred = X @ theta
        grad = X.T @ (y_pred - y) / len(y)
        theta -= lr * grad
        loss = compute_loss(X, y, theta)
        loss_history.append(loss)
    return theta, loss_history


# %%
# 训练模型
lr = 0.01
epochs = 2000
theta, loss_history = gradient_descent(X, y, lr, epochs)
print(f"训练完成，最终参数: {theta}")

# %% [markdown]
# ## 4. 结果分析与可视化

# %%
plt.figure(figsize=(8, 4))
plt.plot(loss_history)
plt.xlabel("迭代次数")
plt.ylabel("损失 (MSE)")
plt.title("损失函数收敛曲线")
plt.show()


# 还原预测函数
def predict(year):
    year_norm = (year - years_mean) / years_std
    x = np.array([1, year_norm])
    return x @ theta


# 预测 2018 和 2019 年学生人数
pred_2018 = predict(2018)
pred_2019 = predict(2019)
print(f"预测2018年学生人数: {pred_2018:.2f}（百人）")
print(f"预测2019年学生人数: {pred_2019:.2f}（百人）")

# 可视化拟合效果
plt.figure(figsize=(8, 5))
plt.scatter(years, y, color="b", label="真实数据")
years_plot = np.linspace(years.min(), years.max(), 100)
plt.plot(years_plot, [predict(y) for y in years_plot], color="r", label="拟合直线")
plt.xlabel("年份")
plt.ylabel("学生人数（百人）")
plt.title("线性回归拟合效果")
plt.legend()
plt.show()
