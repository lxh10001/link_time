#要求:自行推导一遍梯度下降法进行线性回归的过程，理解各个参数是如何变化的。
#    使用python，自主编写线性回归的代码，并将其中的参数的关系以可视化的形式展现出来，
#    如迭代次数n与损失函数的关系，不同学习率α对损失函数有什么影响（至少三种关系）

import numpy as np  
import matplotlib.pyplot as plt  

# 生成样本数据  
np.random.seed(0)  
X = 2 * np.random.rand(100, 1)  
y = 4 + 3 * X + np.random.randn(100, 1)  

# 定义线性回归模型  
def linear_regression(X, y, learning_rate=0.01, n_iterations=1000):  
    m = len(y)  
    X_b = np.c_[np.ones((m, 1)), X]
    #这一行做了两件事,一是创建了一个形状为(m,1)的二维数组,其中所有的元素都是1.
    #这个数组代表了线性回归模型中的偏截距.二是np.c_按列连接两个数组.这里将全为1的偏截距数组和原来的特征矩阵X连接起来,
    # 形成一个新的矩阵X_b,其中X_b第一列是偏置项,其余列是原始特征 X
    theta = np.random.randn(2, 1)
    #生成一个形状为 (2, 1) 的数组，其中的元素是从标准正态分布中随机抽取的。
    loss_history = []
    #用于存储每次迭代后的损失值

    for iteration in range(n_iterations):  
        predictions = X_b.dot(theta)  
        error = predictions - y  
        gradient = X_b.T.dot(error) / m  
        theta = theta - learning_rate * gradient  
        loss = np.sum(error ** 2) / (2 * m)  
        loss_history.append(loss)  
    return theta, loss_history  

# 绘制损失函数与迭代次数的关系  
def plot_loss_over_time(loss_history): 
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)  
    plt.xlabel('Iteration')  
    plt.ylabel('Loss')  
    plt.title(f"Loss over time:{lr}")  
    plt.show()  

# 尝试不同的学习率  
learning_rates = [0.01, 0.001, 0.1, 0.9]  

for lr in learning_rates:  
    theta, loss_history = linear_regression(X, y, learning_rate=lr, n_iterations=1000)  
    print(f"Learning rate: {lr}, Theta found by gradient descent: {theta.T}")  
    plot_loss_over_time(loss_history)  
    

#可视化所有学习率的损失函数(最后几种学习率的图像汇集在一张图上面)
# #指定了图形的宽度和高度 
plt.figure(figsize=(10, 6))

#不同学习率信息
for i, lr in enumerate(learning_rates):  
    theta, loss_history = linear_regression(X, y, learning_rate=lr, n_iterations=1000)  
    plt.plot(loss_history, label=f'lr={lr}')  
#显示
plt.xlabel('Iteration')  
plt.ylabel('Loss')  
plt.title('Loss over time for different learning rates')  
plt.legend()  
plt.show()
