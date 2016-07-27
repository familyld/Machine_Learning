# 支持向量机

**支持向量机（Support Vector Machine，简称SVM）**是一种针对二分类任务设计的分类器，它的理论相对神经网络模型来说更加完备和严密，并且效果显著，结果可预测，是非常值得学习的模型。

这一章的内容大致如下：

- **间隔与支持向量**：如何计算空间中任一点到超平面的距离？什么是支持向量？什么是间隔？支持向量机求解的目标是什么？

- **对偶问题**：求取最大间隔等价于怎样的对偶问题？如何使用SMO算法进行高效求解？

- **核函数**：如何处理非线性可分问题？什么是核函数？什么是核矩阵？有哪些常用的核函数？

- **软间隔与正则化**：如何应对过拟合问题？软间隔和硬间隔分别指什么？如何求解软间隔支持向量机？0/1损失函数有哪些可选的替代损失函数？支持向量机和对率回归模型有什么联系？结构风险和经验风险分别指什么？

- **支持向量回归**：什么是支持向量回归？与传统回归模型有什么不同？支持向量回归的支持向量满足什么条件？

- **核方法**：什么是表示定理？什么是核方法？如何应用？

## 间隔与支持向量

给定一个二分类数据集，正类标记为+1，负类标记为-1（对率回归中负类标记是0，这点是不同的）。

分类学习试图从样本空间中找到一个超平面，使得该超平面可以将不同类的样本分隔开。但是满足这样条件的平面可能有很多，哪一个才是最好的呢？

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

#### 方法1：向量计算

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

#### 方法2：点到直线距离公式

> 假设直线方程为 $ax + by + c= 0$，那么有点到直线距离公式：

> $$r = \frac{|ax + by + c|}{\sqrt{a^2+b^2}}$$

> 而这里的直线方程是 $\mathbf{w}^T\mathbf{x}+b=0$，也即 $a=\mathbf{w}^T$，$b=0$，$c=b$（b为位移项）。代入可得：

> $$r = \frac{|\mathbf{w}^T\mathbf{x}+b|}{\sqrt{\mathbf{w}^T\mathbf{w}+0^2}} = \frac{|\mathbf{w}^T\mathbf{x}+b|}{\Vert \mathbf{w} \Vert}$$

## 习题

#### 6.1

> 问：试证明样本空间中任一点$\mathbf{x}$到超平面的距离公式。


#### 6.2

> 问：试使用LIBSVM，在西瓜数据集3.0$\alpha$上分别用线性核和高斯核训练一个SVM，并比较其支持向量的差别。


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

> 问：以西瓜数据集3.0$\alpha$的“密度”为输入，“含糖率”为输出，试使用LIBSVM训练一个SVR。


#### 6.9

> 问：是使用核技巧推广对率回归，产生“核对率回归”。


#### 6.10*

> 问：试设计一个能显著减少SVM中支持向量的数目而不显著降低泛化性能的方法。


分享一些蛮不错的问题和讲解：

- [支持向量机(SVM)是什么意思？](https://www.zhihu.com/question/21094489)
- [支持向量机中的函数距离和几何距离怎么理解？](https://www.zhihu.com/question/20466147)
- [几何间隔为什么是离超平面最近的点到超平面的距离？](https://www.zhihu.com/question/30217705)
- [SMO优化算法（Sequential minimal optimization）](http://www.cnblogs.com/jerrylead/archive/2011/03/18/1988419.html)
- [Linear SVM 和 LR 有什么异同？](https://www.zhihu.com/question/26768865)
- [SVM计算最优分类超平面时是否使用了全部的样本数据？](https://www.zhihu.com/question/46862433)
- [现在还有必要对SVM深入学习吗？](https://www.zhihu.com/question/41066458)
- [SVM的核函数如何选取？](https://www.zhihu.com/question/21883548)
- [支持向量机通俗导论（理解SVM的三层境界）](http://blog.csdn.net/v_july_v/article/details/7624837)
