
import numpy as np

# 生成示例数据
# data = np.array([[10, 11, 8, 3, 2, 1],
#                  [6, 4, 5, 3, 2.8, 1],
#                  [5, 7, 6, 4, 3.5, 2]])
data = np.array([[10, 6 ,5],
                 [11, 4, 7],
                 [8, 5, 6],
                 [3, 3, 4],
                 [2, 2.8, 3.5],
                 [1, 1, 2]])
print(f'\nshape of array: {data.shape}')
print("the array is:\n", data, "\n")

# 步骤1：数据标准化
data_standardized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# 步骤2：计算协方差矩阵
cov_matrix = np.cov(data_standardized, rowvar=False)

# 步骤3：计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 步骤4：选择主成分
k = 3  # 选择前3个主成分
selected_eigenvectors = eigenvectors[:, :k]

# 步骤5：构建投影矩阵
projection_matrix = selected_eigenvectors

# 步骤6：降维
reduced_data = np.dot(data_standardized, projection_matrix)

# 步骤7：计算每个主成分的信息量
total_variance = np.sum(eigenvalues)
explained_variance = eigenvalues[:k] / total_variance

# 输出结果
print("步骤1：数据标准化\n", data_standardized)
print("\n步骤2：协方差矩阵\n", cov_matrix)
print("\n步骤3：特征值\n", eigenvalues)
print("\n步骤3：特征向量\n", eigenvectors)
print("\n步骤4：选择的主成分\n", selected_eigenvectors)
print("\n步骤6：降维后的数据\n", reduced_data)
print("\n步骤7：每个主成分的信息量\n", explained_variance)
