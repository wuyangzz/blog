---
title: "SVD分解"
author: "wuyangzz"
tags: [""]
categories: [""]
date: 2021-03-19T16:23:19+08:00
---
# 1、特征值分解（EVD) $ a \ne 0 $
## 实对称矩阵
在理角奇异值分解之前，需要先回顾一下特征值分解，如果矩阵A是一个 $  m\times m $的实对称矩阵$ A = A^T $ 
那么它可以被分解成如下的形式
$$A = Q\Sigma Q^T=
Q\left[
\begin{matrix}
    \lambda_1 & \cdots & \cdots & \cdots\\\\
    \cdots & \lambda_2 & \cdots & \cdots\\\\
    \cdots & \cdots & \ddots & \cdots\\\\
    \cdots & \cdots & \cdots & \lambda_m\\\\
\end{matrix}
\right]Q^T
$$

# 2、奇异值分解（SVD）
## 2.1 奇异值分解定义
有一个m×n的实数矩阵A，我们想要把它分解成如下的形式:
$$
A  = U\Sigma V^T
$$
其中U和V均为单位正交阵，即有 $ A = A^T$ 和$VV^T=I$，U称为左奇异矩阵，V称为右奇异矩阵，Σ仅在主对角线上有值，我们称它为奇异值，其它元素均为0。上面矩阵的维度分别为 
$$ U \in R^{m\times m},\ \Sigma \in R^{m\times n},\ V \in R^{n\times n} $$

一般地Σ有如下形式
$$
\Sigma = 
\left[
    \begin{matrix}
    \sigma_1 & 0 & 0 & 0 & 0\\\\
    0 & \sigma_2 & 0 & 0 & 0\\\\
    0 & 0 & \ddots & 0 & 0\\\\
    0 & 0 & 0 & \ddots & 0\\\\
    \end{matrix}
\right]_{m\times n}
$$
![20210319163719](https://raw.githubusercontent.com/wuyangzz/blog_image/main/20210319163719.png)
对于奇异值分解，我们可以利用上面的图形象表示，图中方块的颜色表示值的大小，颜色越浅，值越大。对于奇异值矩阵Σ，只有其主对角线有奇异值，其余均为0。
## 2.2奇异值求解
利用一下性质进行求解
$$
AA^T=U\Sigma V^TV\Sigma^TU^T=U\Sigma \Sigma^TU^T
$$
$$
A^TA=V\Sigma^TU^TU\Sigma V^T=V\Sigma^T\Sigma V^T
$$
利用特征值分解，得到的特征矩阵即为UV；对$ΣΣ^T或Σ^TΣ$中的特征值开方，可以得到所有的奇异值。
## 2.3 奇异值求解Python应用

### 读取图片
![svd](https://raw.githubusercontent.com/wuyangzz/blog_image/main/svd.jpg)

```python
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
 
img_eg = mpimg.imread("svd.jpg")
x=img_eg.shape[0]
y=img_eg.shape[1]
print(img_eg.shape)
```

    (1200, 800, 3)
    

### 奇异值分解


```python
img_temp = img_eg.reshape(x, y * 3)
U,Sigma,VT = np.linalg.svd(img_temp)
print(Sigma.astype(np.int32))
```

    [247846  43859  31072 ...      2      2      2]
    

### 取前部分奇异值重构图片


```python
# 取前10个奇异值
sval_nums = 10
img_restruct0 = (U[:,0:sval_nums]).dot(np.diag(Sigma[0:sval_nums])).dot(VT[0:sval_nums,:])
img_restruct0 = img_restruct0.reshape(x,y,3)

# 取前60个奇异值
sval_nums = 60
img_restruct1 = (U[:,0:sval_nums]).dot(np.diag(Sigma[0:sval_nums])).dot(VT[0:sval_nums,:])
img_restruct1 = img_restruct1.reshape(x,y,3)
 
# 取前120个奇异值
sval_nums = 120
img_restruct2 = (U[:,0:sval_nums]).dot(np.diag(Sigma[0:sval_nums])).dot(VT[0:sval_nums,:])
img_restruct2 = img_restruct2.reshape(x,y,3)

# 取前200个奇异值
sval_nums = 200
img_restruct3 = (U[:,0:sval_nums]).dot(np.diag(Sigma[0:sval_nums])).dot(VT[0:sval_nums,:])
img_restruct3 = img_restruct3.reshape(x,y,3)

# 取前400个奇异值
sval_nums = 400
img_restruct4 = (U[:,0:sval_nums]).dot(np.diag(Sigma[0:sval_nums])).dot(VT[0:sval_nums,:])
img_restruct4 = img_restruct4.reshape(x,y,3)

```


```python
fig, ax = plt.subplots(2,3,figsize = (40,30))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax[0][0].imshow(img_eg)
ax[0][0].set(title = "src")
ax[0][1].imshow(img_restruct0.astype(np.uint8))
ax[0][1].set(title = "nums of sigma = 20")
ax[0][2].imshow(img_restruct1.astype(np.uint8))
ax[0][2].set(title = "nums of sigma = 60")
ax[1][0].imshow(img_restruct2.astype(np.uint8))
ax[1][0].set(title = "nums of2 sigma = 120")
ax[1][1].imshow(img_restruct3.astype(np.uint8))
ax[1][1].set(title = "nums of sigma = 200")
ax[1][2].imshow(img_restruct4.astype(np.uint8))
ax[1][2].set(title = "nums of sigma = 400")
fig.savefig('./svd_return.jpg')
```


    
![![png](output_6_0.png)](https://raw.githubusercontent.com/wuyangzz/blog_image/main/!%5Bpng%5D(output_6_0.png).png)
    


### 奇异值数据


```python
# 奇异值分布图
x=plt.figure(figsize=(40, 20))
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.plot(Sigma.astype(np.int64))
# 3）显示图像
plt.title("Sigma",fontsize=100)
plt.show()
x.savefig('./Sigma.jpg')

```


    
![output_8_0](https://raw.githubusercontent.com/wuyangzz/blog_image/main/output_8_0.png)
    

