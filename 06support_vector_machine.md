# 支持向量机

**支持向量机（Support Vector Machine，简称SVM）**是一种针对二分类任务设计的分类器，它的理论相对神经网络模型来说更加完备和严密，并且效果显著，结果可预测，是非常值得学习的模型。

这一章的内容大致如下：

- **间隔与支持向量**：如何计算空间中任一点到超平面的距离？什么是支持向量？什么是间隔？支持向量机求解的目标是什么？

- **对偶问题**：求取最大间隔等价于怎样的对偶问题？KKT条件揭示出支持向量机的什么性质？如何用SMO算法进行高效求解？为什么SMO算法能高效求解？

- **核函数**：如何处理非线性可分问题？什么是核函数？为什么需要核函数？有哪些常用的核函数？核函数具有什么性质？

- **软间隔与正则化**：如何应对过拟合问题？软间隔和硬间隔分别指什么？如何求解软间隔支持向量机？0/1损失函数有哪些可选的替代损失函数？支持向量机和对率回归模型有什么联系？结构风险和经验风险分别指什么？

- **支持向量回归**：什么是支持向量回归？与传统回归模型有什么不同？支持向量回归的支持向量满足什么条件？

- **核方法**：什么是表示定理？什么是核方法？如何应用？

## 间隔与支持向量

给定一个二分类数据集，正类标记为+1，负类标记为-1（对率回归中负类标记是0，这点是不同的）。

分类学习试图从样本空间中找到一个超平面，使得该超平面可以将不同类的样本分隔开。但是满足这样条件的平面可能有很多，哪一个才是最好的呢？

#### 支持向量

在SVM中，我们试图找到**处于两类样本正中间的超平面**，因为这个超平面**对训练数据局部扰动的容忍性最好**，新样本最不容易被误分类。也就是说这个超平面**对未见示例的泛化能力最强**。

![SVM](http://research.microsoft.com/en-us/um/people/manik/projects/trade-off/figs/svm2.PNG)

上图的实线就是划分超平面，在线性模型中可以通过方程 $\mathbf{w}^T\mathbf{x}+b=0$ 来描述，在二维样本空间中就是一条直线。图中的 $\phi(\mathbf{x})$ 是使用了核函数进行映射，这里暂且不讨论。$\mathbf{w}$ 是线性模型的权重向量（又叫**投影向量**），也是**划分超平面的法向量，决定着超平面的方向**。偏置项 $b$ 又被称为 **位移项，决定了超平面和空间原点之间的距离**。

假设超平面能够将所有训练样本正确分类，也即对于所有标记为+1的点有 $\mathbf{w}^T\mathbf{x}+b>0$，所有标记为-1的点有 $\mathbf{w}^T\mathbf{x}+b<0$。只要这个超平面存在，那么我们必然可以对 $\mathbf{w}$ 和 $b$ 进行适当的**线性放缩**，使得：

$$\mathbf{w}^T\mathbf{x}+b\geq+1,\quad y_i = +1$$
$$\mathbf{w}^T\mathbf{x}+b\leq-1,\quad y_i = -1$$

而SVM中定义**使得上式等号成立的训练样本点**就是**支持向量（support vector）**（如果叫作**支持点**可能更好理解一些，因为事实上就是样本空间中的数据点，但因为我们在表示数据点的时候一般写成向量形式，所以就称为支持向量），它们是距离超平面最近的几个样本点，也即上面图中两条虚线上的点（但图中存在比支持向量距离超平面更近的点，这跟**软间隔**有关，这里暂不讨论）。

在SVM中，我们希望实现的是**最大化两类支持向量到超平面的距离之和**，那首先就得知道怎么计算距离。**怎样计算样本空间中任意数据点到划分超平面的距离**呢？

![PointToHyperPlane](https://github.com/familyld/Machine_Learning/blob/master/graph/PointToHyperPlane.png?raw=true)

画了一个图，方便讲解。图中蓝色线即超平面，对应直线方程 $\mathbf{w}^T\mathbf{x}+b=0$。投影向量 $\mathbf{w}$垂直于超平面，点 $x$ 对应向量 $\mathbf{x}$，过点 $x$ 作超平面的垂线，交点 $x_0$ 对应向量 $\mathbf{x_0}$。假设由点 $x_0$ 指向 点 $x$ 的向量为 $\mathbf{r}$，长度（也即点 $x$ 与超平面的距离）为 $r$。有两种方法计算可以计算出 $r$ 的大小：

##### 方法1：向量计算

> 由向量加法定义可得 $\mathbf{x} = \mathbf{x_0} + \mathbf{r}$。

> 那么向量 $\mathbf{r}$ 等于什么呢？它等于这个方向的单位向量乘上 $r$，也即有 $\mathbf{r} = \frac{\mathbf{w}}{\Vert \mathbf{w} \Vert} \cdot r$

> 因此又有 $\mathbf{x} = \mathbf{x_0} + \frac{\mathbf{w}}{\Vert \mathbf{w} \Vert} \cdot r$。

> 由于点 $x_0$ 在超平面上，所以有 $\mathbf{w}^T\mathbf{x_0}+b=0$

> 由 $\mathbf{x} = \mathbf{x_0} + \frac{\mathbf{w}}{\Vert \mathbf{w} \Vert} \cdot r$ 可得 $\mathbf{x_0} = \mathbf{x} - \frac{\mathbf{w}}{\Vert \mathbf{w} \Vert} \cdot r$，代入直线方程消去 $\mathbf{x_0}$：

> $$\mathbf{w}^T\mathbf{x_0}+b
= \mathbf{w}^T(\mathbf{x} - \frac{\mathbf{w}}{\Vert \mathbf{w} \Vert} \cdot r)+b
= 0$$

> 简单变换即可得到:

> $$r = \frac{\mathbf{w}^T\mathbf{x}+b}{\Vert \mathbf{w} \Vert}$$

> 又因为我们取距离为正值，所以要加上绝对值符号：

> $$r = \frac{|\mathbf{w}^T\mathbf{x}+b|}{\Vert \mathbf{w} \Vert}$$

##### 方法2：点到直线距离公式

> 假设直线方程为 $ax_1 + bx_2 + c= 0$，那么有点到直线距离公式：

> $$r = \frac{|ax + bx_2 + c|}{\sqrt{a^2+b^2}}$$

> 令 $\mathbf{w} = (a,b)$，$\mathbf{x} = (x_1,x_2)$，则可以把 $ax_1 + bx_2$ 写成向量形式 $\mathbf{w}^T\mathbf{x}$。把截距项设为$b$，则直线方程变为 $\mathbf{w}^T\mathbf{x}+b=0$，代入距离公式可得：

> $$r = \frac{|\mathbf{w}^T\mathbf{x}+b|}{\sqrt{\mathbf{w}^T\mathbf{w}}} = \frac{|\mathbf{w}^T\mathbf{x}+b|}{\Vert \mathbf{w} \Vert}$$

> 该式扩展到多维情况下也是通用的。

#### 间隔

前面已经提到，我们希望实现的是**最大化两类支持向量到超平面的距离之和**，而根据定义，所有支持向量都满足：

$$\mathbf{w}^T\mathbf{x}+b = +1,\quad y_i = +1$$
$$\mathbf{w}^T\mathbf{x}+b = -1,\quad y_i = -1$$

代入前面的距离公式可以得到支持向量到超平面的距离为 $\frac{1}{\Vert \mathbf{w} \Vert}$。

定义**间隔（margin）**为**两个异类支持向量到超平面的距离之和**：

$$\gamma = 2 \cdot \frac{1}{\Vert \mathbf{w} \Vert} = \frac{2}{\Vert \mathbf{w} \Vert}$$

SVM的目标便是找到**具有最大间隔（maximum margin）**的划分超平面，也即找到使 $\gamma$ 最大的参数 $\mathbf{w}$ 和 $b$：

$$\max_{\mathbf{w},b} \frac{2}{\Vert \mathbf{w} \Vert} \quad s.t. \quad y_i(\mathbf{w}^T\mathbf{x}+b) \geq 1, \quad  i=1,2,...,m$$

约束部分指的是全部样本都被正确分类，此时标记乘上预测值必定是一个大于等于1的数值。

看上去间隔只与 $\mathbf{w}$ 有关，但实际上位移项 $b$ 也通过约束影响着 $\mathbf{w}$ 的取值，进而对间隔产生影响。

由于最大化 $\Vert \mathbf{w} \Vert^{-1}$ 等价于最小化 $\Vert \mathbf{w} \Vert^{2}$，所以可以重写**目标函数**为：

$$\min_{\mathbf{w},b} \frac{1}{2} \Vert \mathbf{w} \Vert^2 \quad s.t. \quad y_i(\mathbf{w}^T\mathbf{x}+b) \geq 1, \quad  i=1,2,...,m\qquad(1)$$

这便是**支持向量机的基本型**。

特别地，还有以下定义：

**函数间隔**：$y_i(\mathbf{w}^T\mathbf{x}+b)$

**几何间隔**：$\frac{y_i(\mathbf{w}^T\mathbf{x}+b)}{\Vert \mathbf{w} \Vert^2}$

## 对偶问题

式（1）是一个**带约束的凸二次规划（convex quadratic programming）问题**（凸问题就意味着必定能求到全局最优解，而不会陷入局部最优）。对这样一个问题，可以直接用现成的优化计算包求解，但这一小节介绍的是一种更高效的方法。

首先为式（1）的每条约束添加拉格朗日乘子 $a_i \geq 0$（对应m个样本的m条约束），得到该问题的拉格朗日函数：

$$L(\mathbf{w},b,\mathbf{a}) = \frac{1}{2} \Vert \mathbf{w} \Vert^2 + \sum_{i=1}^m a_i(1-y_i(\mathbf{w}^T\mathbf{x}+b))\qquad(2)$$

其中 $\mathbf{a} = (a_1;a_2;...;a_m)$，对拉格朗日函数求 $\mathbf{w}$ 和 $b$ 的偏导，并令偏导为0可以得到：

$$\quad\mathbf{w} = \sum_{i=1}^m a_i y_i \mathbf{x}_i\qquad(3)$$
$$0 = \sum_{i=1}^m a_i y_i\qquad(4)$$

将式（3）代入式（2）可以消去 $\mathbf{w}$ 和 $b$，然后再考虑式（4）的约束就得到了式（1）的**对偶问题（dual problem）**：

$$\max_{\mathbf{a}} \sum_{i=1}^m a_i - \frac{1}{2} \sum_{i=1}^m\sum_{j=1}^m a_i a_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j \quad s.t. \quad \sum_{i=1}^m a_i y_i = 0, \quad a_i \geq 0, \quad i=1,2,...,m \qquad (5)$$

只要求出对偶问题的解 $\mathbf{a}$，就可以推出 $\mathbf{w}$ 和 $b$，从而得到模型：

$$f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b\\
\qquad = \sum_{i=1}^m a_i y_i \mathbf{x}_i^T \mathbf{x} + b \qquad (6)$$

注意，由于式（1）的约束条件是**不等式约束**，所以所得模型还要求满足**KKT（Karush-Kuhn-Tucker）条件**：

$$\left
\{\begin{array}
\\a_i \geq 0;
\\y_i f(\mathbf{x}_i)-1 \geq 0;
\\a_i (y_i f(\mathbf{x}_i)-1) = 0.
\end{array}
\right.$$

KKT条件说明了，对任何一个样本来说，要么对应的拉格朗日因子为0，要么函数间隔等于1（即式（1）的约束条件取等号）。如果拉格朗日因子为0，则这个样本对式（6）毫无贡献，不会影响到模型；如果函数间隔为1，则表明这个样本位于最大间隔边界上，是一个支持向量。它揭示了SVM的一个重要性质：**最终模型只与支持向量有关，因此训练完成后，大部分的训练样本都不需保留**。

#### SMO算法

可以发现对偶问题式（5）是一个二次规划问题，可以使用通用的二次规划算法求解。但**问题规模正比于样本数**，因此开销相当大。为了避免这个开销，可以使用高效的**SMO（Sequential Minimal Optimization）算法**。

初始化参数 $\mathbf{a}$ 后，SMO算法重复下面两个步骤直至收敛：

1. 选取一对需要更新的变量 $a_i$ 和 $a_j$
2. 固定 $a_i$ 和 $a_j$ 以外的参数，求解式（5）来更新 $a_i$ 和 $a_j$

**怎么选取 $a_i$ 和 $a_j$呢？**

注意到，只要选取的 $a_i$ 和 $a_j$ 中有一个不满足KKT条件，那么更新后目标函数的值就会增大。而且**违背KKT调成的程度越大，则更新后导致目标函数增幅就越大**。

因此，SMO算法**先选取一个违背KKT条件程度最大的变量 $a_i$**，然后再选一个使目标函数增长最快的变量 $a_j$，但由于找出 $a_j$ 的开销较大，所以SMO算法采用了一个**启发式**，使**选取的两变量对应的样本之间间隔最大**。这样两个变量差别很大，与选取两个相似变量相比，这种方法能为目标函数带来更大的变化，从而更快搜索到全局最大值。

由于SMO算法在每次迭代中，仅优化两个选定的参数，其他参数是固定的，所以会非常高效。此时，可将对偶问题式（5）的约束重写为：

$$a_iy_i + a_jy_j = c,\quad a_i \geq 0, a_j \geq 0 \qquad (7)$$

其中，$c = -\sum_{k \neq i,j} a_k y_k$ 看作是固定的常数。

利用式（7），我们可以把 $a_j$ 从式（5）中消去，这样就得到了一个**单变量二次规划问题**，只需考虑 $a_i \geq 0$ 这个约束。这样的问题具有**闭式解**，所以我们连数值优化方法都不需要了，可以直接算出 $a_i$ 和 $a_j$。

使用SMO算法计算出最优解之后，我们关注的是如何推出 $\mathbf{w}$ 和 $b$，从而得到最终模型。获得 $\mathbf{w}$ 很简单，直接用式（3）就可以了。而位移项 $b$ 则可以通过支持向量导出。

对于任一支持向量 $(\mathbf{x}_s, y_s)$，都有函数间隔等于1：

$$y_sf(\mathbf{x}) = y_s(\sum_{i \in S} a_i y_i \mathbf{x}_i^T \mathbf{x}_s + b)= 1 \qquad (8)$$

这里的 $S$ 是所有支持向量的下标集（事实上，用所有样本的下标也行，不过非支持向量的拉格朗日因子等于0，对求和没贡献，这一点前面已经提到了）。

理论上，我们只要选取任意一个支持向量代入式（8）就可以把 $b$ 算出来了。但实际任务中往往采用一种**更鲁棒**的做法：用所有支持向量求解的平均值。

$$b = \frac{1}{|S|} \sum_{s \in S} (\frac{1}{y_s} - \sum_{i \in S}a_i y_i \mathbf{x}_i^T \mathbf{x}_s)$$

## 核函数

#### 如何处理非线性划分

在现实任务中，我们更常遇到的是**在原始样本空间中非线性可分**的问题。对这样的问题，一种常用的思路是将样本从原始空间映射到一个更高维的特征空间，使得样本在该特征空间中线性可分。幸运的是，**只要原始空间是有限维的（也即属性数目有限），那就必然存在一个高维特征空间使样本线性可分**。

举个例子，二维平面上若干样本点呈如下分布：

![map](http://my.csdn.net/uploads/201206/03/1338655829_6929.png)

此时要划分两类样本，需要一个非线性的圆型曲线。假设原始空间中两个属性是 $x$ 和 $y$，如果我们做一个映射，把样本点都映射到一个三维特征空间，维度取值分别为 $x^2$，$y^2$ 和 $y$，则得到下面的分布：

![map](http://img.my.csdn.net/uploads/201304/03/1364952814_3505.gif)

可以看到这个时候，我们只需要一个线性超平面就可以将两类样本完全分开了，也就是说可以用前面的方法来求解了。



#### 核函数的性质

## 习题

#### 6.1

> 问：试证明样本空间中任一点$\mathbf{x}$到超平面的距离公式。


#### 6.2

> 问：试使用LIBSVM，在西瓜数据集3.0$a$上分别用线性核和高斯核训练一个SVM，并比较其支持向量的差别。


#### 6.3

> 问：选择两个UCI数据集，分别用线性核和高斯核训练一个SVM，并与BP神经网络和C4.5决策树进行实验比较。


#### 6.4

> 问：试讨论线性判别分析与线性核支持向量机在何种条件下等价。


#### 6.5

> 问：试述高斯核SVM与RBF神经网络之间的联系。


#### 6.6

> 问：试析SVM对噪声敏感的原因。


#### 6.7

> 问：试给出式(6.52)的完整KKT条件。


#### 6.8

> 问：以西瓜数据集3.0$a$的“密度”为输入，“含糖率”为输出，试使用LIBSVM训练一个SVR。


#### 6.9

> 问：是使用核技巧推广对率回归，产生“核对率回归”。


#### 6.10*

> 问：试设计一个能显著减少SVM中支持向量的数目而不显著降低泛化性能的方法。


分享一些蛮不错的问题和讲解：

- [支持向量机(SVM)是什么意思？](https://www.zhihu.com/question/21094489)
- [支持向量机中的函数距离和几何距离怎么理解？](https://www.zhihu.com/question/20466147)
- [几何间隔为什么是离超平面最近的点到超平面的距离？](https://www.zhihu.com/question/30217705)
- [支持向量机(support vector machine)--模型的由来](http://blog.csdn.net/zhangping1987/article/details/21931663)
- [SMO优化算法（Sequential minimal optimization）](http://www.cnblogs.com/jerrylead/archive/2011/03/18/1988419.html)
- [Linear SVM 和 LR 有什么异同？](https://www.zhihu.com/question/26768865)
- [SVM计算最优分类超平面时是否使用了全部的样本数据？](https://www.zhihu.com/question/46862433)
- [现在还有必要对SVM深入学习吗？](https://www.zhihu.com/question/41066458)
- [SVM的核函数如何选取？](https://www.zhihu.com/question/21883548)
- [支持向量机通俗导论（理解SVM的三层境界）](http://blog.csdn.net/v_july_v/article/details/7624837)
