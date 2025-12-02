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
# ---

# %% [markdown]
# # 使用SVM对葡萄酒数据进行分类

# %%
import h5py
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import warnings

from utils.path import project_root

# 忽略一些版本更新可能产生的警告
warnings.filterwarnings("ignore")

# %% [markdown]
# ## 1. 加载数据
#
# 使用 `scipy.io.loadmat` 加载 `.mat` 文件。数据集位于 `datasets` 文件夹下。

# %%
# 定义数据集路径
# 获取项目根目录路径
PROJECT_ROOT = project_root().as_posix()
WINE_DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "wine_data.mat")
WINE_LABEL_PATH = os.path.join(PROJECT_ROOT, "datasets", "wine_label.mat")

# 从加载的数据中提取特征和标签
# 读取 wine_data.mat
with h5py.File(WINE_DATA_PATH, "r") as f:
    X = np.array(f["wine_data"]).T

# 读取 wine_label.mat
# 使用 .ravel() 将标签数组展平为一维数组，以符合 scikit-learn 的要求
with h5py.File(WINE_LABEL_PATH, "r") as f:
    y = np.array(f["wine_label"]).T.ravel()

print(f"数据加载成功。特征维度: {X.shape}, 标签维度: {y.shape}\n")

# %% [markdown]
# ## 2. 数据预处理
#
# - 将数据集划分为训练集和测试集。
# - 对特征进行标准化，因为SVM对特征缩放很敏感。

# %%
# 将数据集划分为训练集和测试集，测试集占30%
# random_state 保证每次划分结果一致，便于复现
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 特征标准化：SVM对特征缩放很敏感，标准化可以提升模型性能
# 创建一个标准化处理器
scaler = StandardScaler()
# 在训练集上计算均值和标准差，并对训练集进行转换
X_train_scaled = scaler.fit_transform(X_train)
# 使用相同的处理器（相同的均值和标准差）对测试集进行转换
X_test_scaled = scaler.transform(X_test)

print("数据预处理完成：已划分为训练集和测试集，并进行了标准化。\n")

# %% [markdown]
# ## 3. 模型训练与超参数调优
#
# 使用 `GridSearchCV` 进行网格搜索和交叉验证，以找到最优的 `C` 和 `gamma` 参数。

# %%
# 定义要搜索的参数网格
# C 是正则化参数，控制模型的复杂度
# gamma 是 'rbf' 核函数的系数
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [1, 0.1, 0.01, 0.001],
    "kernel": ["rbf"],  # 这里我们专注于RBF核，因为它通常性能最好
}

# 使用 GridSearchCV 进行网格搜索和交叉验证
# estimator=SVC() 是我们要优化的模型
# param_grid 是要搜索的参数
# cv=5 表示5折交叉验证
# verbose=2 会打印搜索过程
# n_jobs=-1 使用所有可用的CPU核心并行计算
grid_search = GridSearchCV(SVC(), param_grid, cv=5, verbose=2, n_jobs=-1)

print("开始进行网格搜索以寻找最优参数...")
# 在标准化的训练数据上进行拟合
grid_search.fit(X_train_scaled, y_train)

# 获取最优模型和最优参数
best_svm = grid_search.best_estimator_
print("\n网格搜索完成！")
print(f"找到的最优参数: {grid_search.best_params_}")
print(f"交叉验证最高准确率: {grid_search.best_score_:.4f}\n")

# %% [markdown]
# ## 4. 模型评估
#
# 在测试集上评估最优模型的性能，并打印分类报告和混淆矩阵。

# %%
print("在测试集上评估最优模型性能...")
# 使用最优模型对测试集进行预测
y_pred = best_svm.predict(X_test_scaled)

# 打印分类报告，包含精确率、召回率、F1分数等
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 打印混淆矩阵
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# %% [markdown]
# ## 5. 结果可视化 (使用 PCA 降维)
#
# 由于原始数据有13个特征（维度），我们无法直接在二维或三维空间中将其可视化。为了直观地展示SVM的分类效果，我们可以使用**主成分分析（Principal Component Analysis, PCA）**将数据降至二维，然后在这个二维空间上训练一个新的SVM模型并绘制其决策边界。
#
# 下面的代码将执行以下操作：
# 1.  使用PCA将标准化后的数据降至2个主成分。
# 2.  在降维后的数据上训练一个新的SVM分类器（使用之前找到的最佳参数）。
# 3.  绘制决策边界和数据点，以可视化分类结果。

# %%
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 显示中文
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
from sklearn.decomposition import PCA

# 1. 使用PCA将数据降至二维
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 2. 在降维后的数据上训练一个新的SVM模型
# 我们使用之前网格搜索找到的最佳参数来初始化模型
# best_svm 是在全维度数据上训练的最优模型，我们用它的参数
svm_pca = SVC(**grid_search.best_params_)
svm_pca.fit(X_train_pca, y_train)


# 3. 绘制决策边界的函数
def plot_decision_boundary(X, y, model, title):
    plt.figure(figsize=(10, 6))

    # 创建网格来绘制背景
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    # 预测整个网格的分类结果
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界和区域
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)

    # 绘制数据点
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k")

    plt.xlabel("第一主成分 (Principal Component 1)")
    plt.ylabel("第二主成分 (Principal Component 2)")
    plt.title(title)

    # 从散点图对象中获取图例元素和标签
    handles, labels = scatter.legend_elements()
    # 根据标签数量创建新的标签名称
    class_labels = [f"Class {l}" for l in np.unique(y)]
    plt.legend(handles, class_labels)

    plt.show()


# 可视化训练集上的决策边界
plot_decision_boundary(X_train_pca, y_train, svm_pca, "SVM 决策边界 (训练集, PCA降维)")

# 可视化测试集上的决策边界
plot_decision_boundary(X_test_pca, y_test, svm_pca, "SVM 决策边界 (测试集, PCA降维)")

# %%
