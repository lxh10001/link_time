#要求:使用最小二乘法自行编写算法，调用sklearn库中的经典的波士顿房价数据集
#     (py提示:由于数据集存在伦理问题，并且 scikit-learn 已经移除了对它的支持，
#     这里使用 fetch_california_housing 数据集作为 load_boston 的替代。)
#     将一部分数据进行训练，另一部分数据进行测试，输出拟合图像结果。

import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.datasets import fetch_california_housing  
from sklearn.model_selection import train_test_split  

# 加载加州房价数据集  
california_housing = fetch_california_housing()  
X = california_housing.data  
y = california_housing.target  

# 划分训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# 添加偏置项（截距）  
X_train = np.concatenate([np.ones((X_train.shape[0], 1)), X_train], axis=1)  
#np.conctenate函数用于两个或多个数组沿着指定的轴连接起来,
#这里实际上在X_test的最左边添加了一列全为1的数据,实现截距项
#可以将截距项作为模型的一个权重来优化，而不是手动设置它。
X_test = np.concatenate([np.ones((X_test.shape[0], 1)), X_test], axis=1)  

# 最小二乘法求解参数  
theta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)  

# 使用求解得到的参数进行预测  
y_pred = X_test.dot(theta)  

# 可视化实际房价与预测房价  
plt.scatter(y_test, y_pred, alpha=0.5)  
plt.xlabel('Actual Prices')  
plt.ylabel('Predicted Prices')  
plt.title('Actual vs Predicted Prices on California Housing Dataset')  
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)  # 绘制理想情况下的直线 y=x  
plt.show()