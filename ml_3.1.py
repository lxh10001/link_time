#要求:用热身题中的学习了解的Anaconda创建一个虚拟环境，在该virtual env中使用python，
#    自行创建一组数据集，尝试调用sklearn库中的线性回归模型进行训练，
#    并调用matplotlib库输出

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error#mean_squared_error is a loss function

# 创建数据集
np.random.seed(0)  # 设置随机种子,保证结果的可重复性
X = np.random.rand(100, 1)  # 创建100个随机数作为特征
y = 3 * X.ravel() + 2  # 创建目标变量，与X呈线性关系

# 添加一些噪声
y += np.random.normal(0, 0.1, y.shape)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 进行预测
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 评估模型
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print(f"Training MSE: {mse_train}")
print(f"Testing MSE: {mse_test}")

coef = model.coef_#模型的系数
intercept = model.intercept_#模型的截距

# 使用matplotlib绘制数据点和回归线
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='red', label='Test data')
plt.plot(X, model.predict(X), color='green', linewidth=2, label='Regression line')

# 添加图例和轴标签
plt.legend()
plt.xlabel('Feature')
plt.ylabel('Target')

# 显示图表
plt.show()

# 输出模型的系数和截距
print(f"Coefficients: {coef}")
print(f"Intercept: {intercept}")







