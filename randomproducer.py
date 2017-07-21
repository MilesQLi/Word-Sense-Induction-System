#coding=utf-8
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


#print random.uniform(-20, 20)
#print random.random()

data = []
label = np.append(np.zeros(10), np.ones(15))
label = np.append(label, np.zeros(5))
for i in range(11):
    temp = []
    temp.append(5+random.random())
    temp.append(3+random.random())
    data.append(temp)
for i in range(16):
    temp = []
    temp.append(10+random.random())
    temp.append(7+random.random())
    data.append(temp)
for i in range(5):
    temp = []
    temp.append(4.5+random.random())
    temp.append(2.9+random.random())
    data.append(temp)

data=np.array(data)

clf = KMeans(init='k-means++', n_clusters=2, n_init=10)
clf.fit(data)

h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5

print x_min,x_max
print y_min,y_max
'''
详细解释：help meshgrid
meshgrid用于从数组a和b产生网格。生成的网格矩阵A和B大小是相同的。它也可以是更高维的。
[A,B]=Meshgrid(a,b)
生成size(b)Xsize(a)大小的矩阵A和B。它相当于a从一行重复增加到size(b)行，把b转置成一列再重复增加到size(a)列。因此命令等效于：
'''
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
#np.c_(x,y)是把两个向量组合成二维向量，ravel是把矩阵的每一行展开连接成一个向量，所以从meshgrid到c_是得到了所有以一定间隔分离后的数据的坐标的组合
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
#把Z的形状从向量变成X一样的矩阵是为了imgshow
Z = Z.reshape(xx.shape)
plt.figure(1)
#clear当前图
plt.clf()
#在坐标轴上展示一张图，根据数据的值设定相应部分的颜色
im=plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')
#显示colorbar
plt.colorbar(im)
#显示数据点
plt.plot(data[:, 0], data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = clf.cluster_centers_
#散列图，与plot比较像
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='b', zorder=10)
plt.title('K-means clustering on the digits dataset \n'
          'Centroids are marked with white cross')
#设置x轴范围
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
#plt.xticks(())
#plt.yticks(())
#显示图示plt.legend()
plt.show()


