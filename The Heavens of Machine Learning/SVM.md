#! https://zhuanlan.zhihu.com/p/266533885
# 支持向量机 Support Vector Machine (SVM)

最近在回顾机器学习的一些经典方法，顺带推推公式以免生疏。检验自己是否充分理解一个方法最好的方式就是复述出来，看能否讲清楚。SVM 是机器学习中的一个代表性模型，涉及到不少基础知识点，所以干脆就针对 SVM 写一篇文章。

写 SVM 的文章太多了，但总有些概念或者细节会被忽视，有些术语可能被频繁地提到，但没有说为什么要用或者具体怎么用？写这篇文章也是希望可以把思路理清晰一些，这样当我们遇到类似的优化问题时就能想起这些工具。当然，因为个人的研究方向不是优化，所以这篇文章不会具体地谈到如何证明 Slater's 条件能得出强对偶性或者如果由 Slater's 条件得到 KKT 条件等等这类问题，但是会讲到在 SVM 的问题里为什么需要用到它们以及怎么用。

SVM 有三宝，间隔、对偶、核技巧。一般来说 SVM 可以分为三种类别，也即：

1. Hard-margin SVM
2. Kernel SVM
3. Soft-margin SVM

这篇推导也从最原汁原味的硬间隔 SVM 开始推导，然后引入核技巧，软间隔，最后讲解用于求解 SVM 对偶型的 SMO 算法。

## Hard-margin SVM

![SVM](https://pic4.zhimg.com/80/v2-e055ddb1d9fba54bcd85819db81587ca.png)

首先，SVM 最早是作为针对二分类任务设计的，给定数据集 

$$\mathcal{D} = \{(\mathbf{x}^{(i)},\ y^{(i)})\}_{i=1}^{N},$$

其中 $\mathbf{x}^{(i)} \in \mathbb{R}^d,\ y^{(i)} \in \{-1, +1\}$。SVM 可以理解为特征空间中的一个超平面，用 $\mathbf{w}^T\mathbf{x}+b=0$ 表示，由参数 $\mathbf{w} \in \mathbb{R}^d$ 和 $b \in \mathbb{R}$ 决定。使用 $\text{sign}(\mathbf{w}^T\mathbf{x}+b)$ 来输出样本 $\mathbf{x}$ 的类别，是一个标准的判别模型。

SVM 的目标是最大化间隔：

$$\begin{aligned}
\max_{\mathbf{w}, b} \text{margin}(\mathbf{w}, b) \quad \text{s.t.}\quad  
\left\{\begin{array}{l}
\mathbf{w}^T\mathbf{x}^{(i)}+b > 0,\ y^{(i)}=+1\\
\mathbf{w}^T\mathbf{x}^{(i)}+b < 0,\ y^{(i)}=-1
\end{array}\right. ,\forall i=1, \cdots, N
\end{aligned}$$

由于 $y^{(i)}$ 只有正负1两种取值，所以上式也可以简写为：

$$\begin{aligned}
\max_{\mathbf{w}, b} \text{margin}(\mathbf{w}, b) \quad \text{s.t.} \quad  
y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b) > 0\ ,\forall i=1, \cdots, N
\end{aligned}$$

那么怎么定义 $\text{margin}(\cdot)$ 函数呢？从几何的角度来看，它其实就是**样本点与超平面间的最短距离**，根据点到直线距离公式有：

$$\begin{aligned}
\text{margin}(\textbf{w}, b) 
&= \min_{\textbf{x}^{(i)}} \text{distance}(\textbf{w}, b, \textbf{x}^{(i)})\\
&= \min_{\textbf{x}^{(i)}} \frac{1}{\Vert \textbf{w}\Vert}|\mathbf{w}^T\mathbf{x}^{(i)}+b|\\
&= \min_{\textbf{x}^{(i)}} \frac{1}{\Vert \textbf{w}\Vert} \mathbf{y}^{(i)}\mathbf{w}^T\mathbf{x}^{(i)}+b
\end{aligned}$$

因此，SVM 的目标可以重写为：

$$\begin{aligned}
&\max_{\mathbf{w}, b} \text{margin}(\mathbf{w}, b) \quad \text{s.t.}\quad
y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b) > 0\ ,\forall i=1, \cdots, N\\\Leftrightarrow
&\max_{\mathbf{w}, b} \min_{\textbf{x}^{(i)}} \frac{1}{\Vert \textbf{w}\Vert} \mathbf{y}^{(i)}\mathbf{w}^T\mathbf{x}^{(i)}+b \quad \text{s.t.}\quad  
y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b) > 0\ ,\forall i=1, \cdots, N\\\Leftrightarrow
&\max_{\mathbf{w}, b} \frac{1}{\Vert \textbf{w}\Vert} \min_{\textbf{x}^{(i)}} \mathbf{y}^{(i)}\mathbf{w}^T\mathbf{x}^{(i)}+b \quad \text{s.t.}\quad  
y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b) > 0\ ,\forall i=1, \cdots, N
\end{aligned}$$

本质上，这是对模型（超平面）的归纳偏好，SVM 认为间隔最大的超平面是最好的，直觉上这其实就是要找**处于两类样本正中间的超平面**，因为这个超平面**对训练数据局部扰动的容忍性最好**。新样本最不容易被误分类，也就是说这个超平面**对未见示例的泛化能力最强**。

$\min_{\textbf{x}^{(i)}} \mathbf{y}^{(i)}\mathbf{w}^T\mathbf{x}^{(i)}+b$ 称为函数间隔，它与真实间隔相差了一个因子 $\frac{1}{\Vert \textbf{w}\Vert}$。我们不妨将函数间隔固定为1，这有助于后续推导，事实上，我们总能通过对参数 $\mathbf{w}$ 和 $b$ 进行放缩来满足这个条件。此时，SVM 的目标可以重写为：

$$\begin{align}
&\max_{\mathbf{w}, b} \frac{1}{\Vert \textbf{w}\Vert} \quad \text{s.t.}\quad  
y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b) \geq 1\ ,\forall i=1, \cdots, N \quad   \text{函数间隔为1，消掉了}\\\Leftrightarrow
&\min_{\mathbf{w}, b} {\Vert \textbf{w}\Vert} \quad \text{s.t.}\quad  
y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b) \geq 1\ ,\forall i=1, \cdots, N \quad  \text{一般写作最小化的形式}\\\Leftrightarrow 
&\min_{\mathbf{w}, b} \sqrt{\textbf{w}^T\textbf{w}} \quad \text{s.t.}\quad  
y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b) \geq 1\ ,\forall i=1, \cdots, N \\\Leftrightarrow
&\min_{\mathbf{w}, b} \frac{1}{2}\textbf{w}^T\textbf{w} \quad \text{s.t.}\quad  
y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b) \geq 1\ ,\forall i=1, \cdots, N \quad   \text{引入}\frac{1}{2}\text{便于求导}\tag{1}
\end{align}$$

公式（1）所示就是 SVM 的基本型，从几何的角度出发，将最大化间隔的二分类问题转换为一个约束优化问题，更准确地来说，这是一个**凸二次规划问题**，目标函数是（$\mathbf{w}$ 的）凸二次函数，约束都是线性约束。这类问题可以直接调用求解器求解，但对于样本数较多或特征维数较高（做非线性变换后甚至会出现无限维）的情况，直接求解的计算开销会非常大，因此，我们在使用 SVM 时更倾向于使用别的方法。

首先对约束项进行一个改写，将优化问题写作：

$$\begin{align}
\min_{\mathbf{w}, b} \frac{1}{2}\textbf{w}^T\textbf{w} \quad \text{s.t.}\quad  
1-y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b) \leq 0\ ,\forall i=1, \cdots, N\tag{2}
\end{align}$$

式（2）一般称作 SVM 的**原问题（primal problem）**。拉格朗日乘子法是求解这类约束优化问题的常用方法，通过引入拉格朗日乘子将所有约束写进目标函数中，然后通过令偏导为0的方式求出极值。定义原问题的拉格朗日函数为：

$$\begin{aligned}
\mathcal{L}(\mathbf{w},b,\mathbf{\lambda}) = \frac{1}{2}\textbf{w}^T\textbf{w} + \sum_{i=1}^N \lambda^{(i)} 
[1-y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b)]
\end{aligned}$$

得到**原问题的等价形式**：

$$\begin{align}
\min_{\mathbf{w}, b} \max_{\mathbf{\lambda}}\mathcal{L}(\mathbf{w},b,\mathbf{\lambda}) \quad \text{s.t.} \quad \lambda^{(i)} \geq 0\ ,\forall i=1, \cdots, N\tag{3}
\end{align}$$

式（3）和式（2）描述的式相同的问题，因为对式（3）而言，$\lambda$ 要最大化 $\mathcal{L}(\mathbf{w},b,\mathbf{\lambda})$ 其实就只有两种情况：

$$\begin{aligned} 
\left\{\begin{array}{l}
\lambda^{(i)}=0, \quad &1-y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b) \leq 0\\
\lambda^{(i)}=+\infty, \quad &1-y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b) \gt 0
\end{array}\right.
\end{aligned}$$

满足约束时，$\lambda$ 必须为0，否则式子就会加上一个负数变小；违背约束时，$\lambda$ 取为正无穷，可以让式子取值最大化。因此，式（3）和式（2）是完全等价的。由于式（3）外层是求 $\min$，它会迫使 $\mathbf{w}$ 和 $b$ 必须满足约束 $1-y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b) \leq 0$，从而让 $\lambda^{(i)}=0$。直观上，可以把 $\lambda$ 理解为违背约束时惩罚的力度。[从几何的角度理解](https://www.zhihu.com/question/38586401/answer/457058079)，这种构造方式的本质是取得极值时，目标函数和约束条件描述的曲线相切，有着相同的法线方向。

原问题本身可以解，但我们往往都会解它的对偶问题，也即：

$$\begin{align}
\max_{\mathbf{\lambda}} \min_{\mathbf{w}, b} \mathcal{L}(\mathbf{w},b,\mathbf{\lambda}) \quad \text{s.t.} \quad \lambda^{(i)} \geq 0\ ,\forall i=1, \cdots, N\tag{4}
\end{align}$$

为什么要解对偶问题这里暂且跳过，先分析**对偶问题与原问题的解（分别记作 $d^*$ 和 $p^*$）之间的关系**。首先，对偶问题是求最小值中的最大值，原问题是求最大值中的最小值，所以这里很直观地可以得到：

$$\begin{aligned}
 &d^* = \max_{\mathbf{\lambda}} \min_{\mathbf{w}, b}\mathcal{L}(\mathbf{w},b,\mathbf{\lambda}) \leq \min_{\mathbf{w}, b} \max_{\mathbf{\lambda}}\mathcal{L}(\mathbf{w},b,\mathbf{\lambda}) = p^*
\end{aligned}$$

也即对偶问题的解是原问题的解的下确界，这也称为**弱对偶性（weak duality）**，证明如下：

对于任意 $\mathbf{\lambda}$ 均有 $\min_{\mathbf{w}, b}\mathcal{L}(\mathbf{w},b,\mathbf{\lambda}) \leq \min_{\mathbf{w}, b} \max_{\mathbf{\lambda}} \mathcal{L}(\mathbf{w},b,\mathbf{\lambda})$。对于 $\mathbf{\lambda} = \arg\max_{\mathbf{\lambda}} \min_{\mathbf{w}, b}\mathcal{L}(\mathbf{w},b,\mathbf{\lambda})$ 亦然，得证。

由于我们要求的是原问题的最小值，得到一个下界没有什么用。我们真正需要的是**强对偶性**，也即 $d^* = p^*$，此时我们便可以通过求解对偶问题直接得到原问题的最优解。

为了得到强对偶性，我们需要用到 [Slater's Condition](https://en.wikipedia.org/wiki/Slater_condition)：

> In mathematics, Slater's condition (or Slater condition) is a **sufficient condition for strong duality** to hold for a **convex optimization problem**.

准确来说，它针对的是凸规划问题：

$$\begin{aligned}
\min _{x} f_{0}(x)
\quad \text{s.t.} \quad 
\left\{\begin{array}{l}
&f_{i}(x) \leq 0, \quad i=1, \cdots, m \\
&h_{i}(x)=0,  \quad i=1, \cdots, p
\end{array}\right.
\end{aligned}$$

其中，函数 $f_0, f_1, \cdots, f_m$ 都是凸函数，$h_0, h_1, \cdots, h_p$ 都是仿射变换（一次线性变换+平移）。**仅当存在点 $x^*$ 能满足所有约束项时（strictly feasible），该问题满足强对偶性**。

显然，式（3）所示的原问题属于凸规划问题，不过它没有等式约束，并且所有的不等式约束都是仿射变换。对于这种情形，我们可以使用 [a weak form of Slater’s condition](https://inst.eecs.berkeley.edu/~ee227a/fa10/login/l_dual_strong.html)，它放宽了要求，如果 $f_i(x)$ 是仿射变换，则不需要 strict feasibility。也就是说，式（3）满足 Slater's 条件，该问题满足强对偶性！

以上就是为什么可以通过求解对偶问题来解决 SVM 的原问题。在谈及如何求解对偶问题和对偶问题的优点前，这里还需要指出一点。原始的拉格朗日乘子法是用于求解仅包含等式约束的问题的，而 SVM 的原问题中都是不等式约束。**要将拉格朗日乘子法推广到包含不等式约束的问题，最优解就必须满足 [KKT 条件（Karush–Kuhn–Tucker conditions）](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions)**：

1. **原问题可行（Primal feasibility）**：$\quad\qquad f_{i}(x) \leq 0, \quad g_{i}(x) = 0$
2. **对偶问题可行（Dual feasibility）**：$\quad\qquad \lambda^{(i)} \geq 0$
3. **互补松弛（Complementary slackness）**：$\lambda^{(i)}f_{i}(x)=0$

有的文献还会写上 **Stationarity** 这一条，也即梯度为零，因为它包含在了后续求解过程中，所以有时候会忽略。那么 **SVM 原/对偶问题的解是否能满足 KKT 条件呢**？

要判断问题的解是否满足 KKT 条件，我们一般会看在该问题中 [Constraint Qualification (CQ)](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions#Regularity_conditions_(or_constraint_qualifications)) 是否成立，如果成立，那么问题的解就满足 KKT 条件。幸运的是，前面提到的 Slater's Condition 正是其中一种 CQ，所以 SVM 原/对偶问题的解均能满足 KKT 条件，并且 **KKT 条件是最优解的充要条件**。也即 SVM 的最优解满足：

$$\begin{align}
\left\{\begin{array}{l}
& 1-y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b) \leq 0\\
& \lambda^{(i)} \geq 0\tag{5}\\
& \lambda^{(i)}[1-y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b)]=0
\end{array}\right.
\end{align}$$


根据 Arthur Gretton 的[课件](http://www.gatsby.ucl.ac.uk/~gretton/coursefiles/Slides5A.pdf)，Slater's Condition、强对偶性、KKT 条件与最优解之间的关系可以归纳如下：

> If: <br>1. primal problem **convex** and<br>2. constraint functions satisfy **Slater's conditions**<br> then **strong duality** holds. If in addition: <br> - functions $f_i$, $h_i$ are **differentiable**<br> then KKT conditions necessary and sufficient for **optimality**.

**注**：可微是因为 KKT 还要求最优解有偏导数为零（Stationarity）的性质，这恰恰也是用拉格朗日乘子法求解时的一个步骤。

我们现在知道了对偶问题式（4）与原问题式（3）有相同的最优解，并且最优解满足 KKT 条件，那就可以开始求解目标函数了。我们首先解 $\min_{\mathbf{w}, b}\mathcal{L}(\mathbf{w},b,\mathbf{\lambda})$，写出目标函数对参数 $b$ 和 $\mathbf{w}$ 的偏导数，并令其为零：

$$\begin{align}
\frac{\partial \mathcal{L}(\mathbf{w},b,\mathbf{\lambda})}{\partial b}
&= \frac{\partial}{\partial b} \lgroup\frac{1}{2}\textbf{w}^T\textbf{w} + \sum_{i=1}^N \lambda^{(i)} 
[1-y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b)]\rgroup\\ 
&= \frac{\partial}{\partial b}[ \sum_{i=1}^N \lambda^{(i)} - \sum_{i=1}^N \lambda^{(i)} y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b)]\\
&= \frac{\partial}{\partial b} [- \sum_{i=1}^N \lambda^{(i)} y^{(i)}b]\\
&= - \sum_{i=1}^N \lambda^{(i)} y^{(i)} \triangleq 0\\
&\Leftrightarrow \sum_{i=1}^N \lambda^{(i)} y^{(i)} = 0
\tag{6}
\end{align}$$

$$\begin{align}
\frac{\partial \mathcal{L}(\mathbf{w},b,\mathbf{\lambda})}{\partial \mathbf{w}}&= \frac{\partial}{\partial \mathbf{w}} \lgroup\frac{1}{2}\textbf{w}^T\textbf{w} + \sum_{i=1}^N \lambda^{(i)} 
[1-y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b)]\rgroup\\
&= \frac{\partial}{\partial \mathbf{w}} [\frac{1}{2}\mathbf{w}^T\mathbf{w}]+ \frac{\partial}{\partial \mathbf{w}}[ \sum_{i=1}^N \lambda^{(i)} - \sum_{i=1}^N \lambda^{(i)} y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b)]\\
&= \frac{1}{2} \cdot 2 \cdot \mathbf{w} - \frac{\partial}{\partial \mathbf{w}} [\sum_{i=1}^N \lambda^{(i)} y^{(i)} \mathbf{w}^T\mathbf{x}^{(i)}]\\
&= \mathbf{w}- \sum_{i=1}^N \lambda^{(i)} y^{(i)}\mathbf{x}^{(i)} \triangleq 0\\
&\Leftrightarrow w^* = \sum_{i=1}^N \lambda^{(i)} y^{(i)}\mathbf{x}^{(i)} \tag{7}
\end{align}$$

然后将式（6）和式（7）回代到目标函数中：

$$\begin{aligned}
& \mathcal{L}(\mathbf{w},b,\mathbf{\lambda})\\
=& \frac{1}{2}\textbf{w}^{*^{T}}\textbf{w}^* + \sum_{i=1}^N \lambda^{(i)} 
[1-y^{(i)}(\textbf{w}^{*^{T}}\mathbf{x}^{(i)}+b^*)]\\
=& \frac{1}{2} \textbf{w}^{*^{T}}\textbf{w}^* + \sum_{i=1}^N \lambda^{(i)} - \sum_{i=1}^N \lambda^{(i)}y^{(i)}\textbf{w}^{*^{T}}\mathbf{x}^{(i)}-\sum_{i=1}^N \lambda^{(i)}y^{(i)} \cdot b^* \quad \text{根据式（6），最后一项为0}\\
=& \frac{1}{2} (\sum_{i=1}^N \lambda^{(i)} y^{(i)}\mathbf{x}^{(i)})^T(\sum_{j=1}^N \lambda^{(j)} y^{(j)}\mathbf{x}^{(j)}) + \sum_{i=1}^N \lambda^{(i)} - \sum_{i=1}^N \lambda^{(i)}y^{(i)} (\sum_{i=j}^N \lambda^{(j)} y^{(j)}\mathbf{x}^{(j)})^T \mathbf{x}^{(i)}\\
=& \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \lambda^{(i)} \lambda^{(j)} y^{(i)} y^{(j)} \mathbf{x}^{{(i)}^T} \mathbf{x}^{(j)} + \sum_{i=1}^N \lambda^{(i)} - \sum_{i=1}^N \sum_{j=1}^N \lambda^{(i)} \lambda^{(j)} y^{(i)} y^{(j)} \mathbf{x}^{{(i)}^T} \mathbf{x}^{(j)}\\
=& -\frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \lambda^{(i)} \lambda^{(j)} y^{(i)} y^{(j)} \mathbf{x}^{{(i)}^T} \mathbf{x}^{(j)} + \sum_{i=1}^N \lambda^{(i)}
\end{aligned}$$

这样，式（4）的优化问题就变为一个只含变量 $\mathbf{\lambda}$ 的优化问题了，这也称为 SVM 的对偶型：

$$\begin{align}
& \max_{\mathbf{\lambda}} \min_{\mathbf{w}, b} \mathcal{L}(\mathbf{w},b,\mathbf{\lambda}) \quad \text{s.t.} \quad \lambda^{(i)} \geq 0\ ,\forall i=1, \cdots, N\\\Leftrightarrow
& \max_{\mathbf{\lambda}} -\frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \lambda^{(i)} \lambda^{(j)} y^{(i)} y^{(j)} \mathbf{x}^{{(i)}^T} \mathbf{x}^{(j)} + \sum_{i=1}^N \lambda^{(i)} \quad \text{s.t.} \quad 
\left\{\begin{array}{l}
&\lambda^{(i)} \geq 0\\
&\sum_{i=1}^N \lambda^{(i)} y^{(i)} = 0
\end{array}\right.\ ,\forall i=1, \cdots, N
\\\Leftrightarrow
& \min_{\mathbf{\lambda}} \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \lambda^{(i)} \lambda^{(j)} y^{(i)} y^{(j)} \mathbf{x}^{{(i)}^T} \mathbf{x}^{(j)} - \sum_{i=1}^N \lambda^{(i)} \quad\quad \text{s.t.} \quad 
\left\{\begin{array}{l}
&\lambda^{(i)} \geq 0\\
&\sum_{i=1}^N \lambda^{(i)} y^{(i)} = 0
\end{array}\right.\ ,\forall i=1, \cdots, N
\tag{8}
\end{align}$$

式（8）同样是一个凸二次规划问题，一样可以用求解器来解，但实践中有更高效的求解方法 —— **SMO 方法**。这里先不论如何求解，假设最优参数 $\lambda^{(i)}$ 已知，我们来看看得到的分类器 $\text{sign}(\mathbf{w}^{*^T}\mathbf{x}+b^*)$ 是怎样的。首先，$w^*$ 在式（7）中已经给出：

$$
w^* = \sum_{i=1}^N \lambda^{(i)} y^{(i)}\mathbf{x}^{(i)}
$$

计算偏置 $b^*$ 需要利用到式（5），也即 KKT 条件中的互补松弛性质，由该性质我们可以知道 $\lambda^{(i)} \neq 0$ 时，必然有 $1-y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b)=0$。假设 $（\mathbf{x}^{(k)}, y^{(k)}）$ 是满足该等式的其中一个样本点，则有：

$$\begin{aligned}
& 1-y^{(k)}(\mathbf{w}^{*^T}\mathbf{x}^{(k)}+b^*)=0\\\Leftrightarrow
& y^{(k)}(\mathbf{w}^{*^T}\mathbf{x}^{(k)}+b^*)=1\\\Leftrightarrow
& \mathbf{w}^{*^T}\mathbf{x}^{(k)}+b^*=y^{(k)} \quad \text{等式左右都乘上了}y^{(k)}\\\Leftrightarrow
& b^* = y^{(k)}-\mathbf{w}^{*^T}\mathbf{x}^{(k)}\\\Leftrightarrow
& b^* = y^{(k)}-\sum_{i=1}^N \lambda^{(i)} y^{(i)}\mathbf{x}^{(i)^{T}}\mathbf{x}^{(k)}
\end{aligned}$$

实践中为了提高鲁棒性，参数 $b$ 往往会采用所有支持向量算出的均值。可以看出最优参数 $\mathbf{w}^{*^T}$ 和 $b^*$ 都是样本点的加权组合，权值为 $\lambda^{(i)}$。由互补松弛性质可得，$1-y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b) \neq 0 \Leftrightarrow \lambda^{(i)}=0$，所以（通常情况下）大多数样本都是无用的，模型只与满足约束 $1-y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b)=0$ 的样本点有关，我们把这些样本点称为**支持向量**。

**支持向量的数量一定很少吗？** 不一定，它只是求解 SVM 的一个副产物。支持向量很多也是完全有可能的，有时候这可能意味着过拟合。

那么到底**为什么要求解对偶问题而非原问题**呢？原因主要有以下几个方面：

1. **求解高效**：原问题需求解 $d+1$ 个变量，对偶问题只需求解 $N$ 个变量（早期的机器学习研究中数据集普遍较小，所以可能会有 $d \gg N$ 的情况），而且对偶问题有一些高效的解法（比如：SMO）；
2. **核技巧**：对偶问题中用到了 $\mathbf{x}^{(i)^{T}}\mathbf{x}^{(k)}$，也即特征向量的内积，**便于使用核技巧**，隐式地将特征向量投影到高维甚至无穷维来计算，从而可以解决非线性分类问题（**主要原因**）；
3. **计算高效**：对偶问题可以求出 $\lambda^{(i)}$，因为模型仅与支持向量有关，我们可以基于支持向量计算分类结果。利用核技巧就不需要显式地将特征向量投影到高维，直接与支持向量计算内积即可达到相同的效果。

## Kernel SVM

首先，[核方法](https://en.wikipedia.org/wiki/Kernel_method)是一类使用了[核函数](https://en.wikipedia.org/wiki/Positive-definite_kernel)的算法，包括 SVM，Gaussian processes，PCA，ridge regression，spectral clustering 等等。核函数有很多种，它们的特点就是可以在数据的原始表示空间 $\mathcal{X}$ 下计算它们在高维特征空间 $\mathcal{V}$ 下的相似性而无需显式地将其映射到 $\mathcal{V}$，也即用 $\kappa(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$ 替换 $\phi(\mathbf{x}^{(i)})^T\phi(\mathbf{x}^{(j)})$。这种使用核函数计算相似性的方法称为**核技巧（kernel trick）**。我们平常使用的神经网络等模型都是显式地对样本 $\mathbf{x}^{(i)}$ 进行映射，为此我们需要设定模型架构 $\phi(\mathbf{x}^{(i)})$，而核方法需要选择的则是核函数 $\kappa(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$。

![example](https://pic4.zhimg.com/80/v2-4efd5e6ca85897dc61e7492268cea642.png)

以上图为例，左边是数据的原始表示空间，两类数据是线性不可分的。但如果我们将其映射到右边的高维情况，它们就可以被一个超平面区分开。这里用到的映射函数 $\phi(\mathbf{x}^{(i)})=(\mathbf{x}^{(i)}_1, \mathbf{x}^{(i)}_2, \mathbf{x}^{(i)^2}_1+\mathbf{x}^{(i)^2}_2)$，对应的核函数是 $\kappa(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) = \mathbf{x}^{(i)^T}\mathbf{x}^{(j)} + \Vert \mathbf{x}^{(i)} \Vert^2\Vert \mathbf{x}^{(j)} \Vert^2$，不妨验证一下：

$$\begin{aligned}
\phi(\mathbf{x}^{(i)})^T\phi(\mathbf{x}^{(j)})
& = (\mathbf{x}^{(i)}_1, \mathbf{x}^{(i)}_2, \mathbf{x}^{(i)^2}_1+\mathbf{x}^{(i)^2}_2)^T(\mathbf{x}^{(j)}_1, \mathbf{x}^{(j)}_2, \mathbf{x}^{(j)^2}_1+\mathbf{x}^{(j)^2}_2)\\
& = \mathbf{x}^{(i)}_1\mathbf{x}^{(j)}_1 + \mathbf{x}^{(i)}_2\mathbf{x}^{(j)}_2 + (\mathbf{x}^{(i)^2}_1+\mathbf{x}^{(i)^2}_2)(\mathbf{x}^{(j)^2}_1+\mathbf{x}^{(j)^2}_2)\\
& = \mathbf{x}^{(i)^T}\mathbf{x}^{(j)} + \Vert \mathbf{x}^{(i)} \Vert^2\Vert \mathbf{x}^{(j)} \Vert^2
\end{aligned}$$

做显式映射 $\phi(\cdot)$ 需要把样本由 2 维映射到 3 维才能计算内积，而使用核函数 $\kappa(\cdot)$ 时所有的计算都能在 2 维的原始表示空间中进行。对于更为复杂的映射而言，使用核函数能节省很大的计算开销。

一些常见的核函数如下表所示：

![常见核函数](https://pic4.zhimg.com/80/v2-cab80be1d10a0f98131a46ffb51141cc.png)

将核函数用到 SVM 中就得到了 Kernel SVM，目标函数是：

$$\begin{align}
\min_{\mathbf{\lambda}} \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \lambda^{(i)} \lambda^{(j)} y^{(i)} y^{(j)} \kappa(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) - \sum_{i=1}^N \lambda^{(i)} \quad\quad \text{s.t.} \quad 
\left\{\begin{array}{l}
&\lambda^{(i)} \geq 0\\
&\sum_{i=1}^N \lambda^{(i)} y^{(i)} = 0
\end{array}\right.\ ,\forall i=1, \cdots, N \tag{9}
\end{align}$$

使用 Kernel SVM 进行预测：

$$\begin{aligned}
w^*\phi(\mathbf{x}^{(j)})+b^* = \sum_{i=1}^N \lambda^{(i)} y^{(i)}\kappa(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) + y^{(k)}-\sum_{i=1}^N \lambda^{(i)} y^{(i)} \kappa(\mathbf{x}^{(i)}, \mathbf{x}^{(k)})
\end{aligned}$$

核函数怎么选网上也有不少文章给出了分析，这里不再赘述。

## Soft-margin SVM

前面讨论 Hard-margin SVM 和 Kernel SVM 时，我们都是认为数据要“完美地”线性可分，但现实中由于噪声数据的存在，一味地追求这种完美可能会导致严重的过拟合，最大间隔可能会变得非常非常窄，从而无法做出 robust 的预测。更糟的是，异常点的存在可能会让本来线性可分的数据变得线性不可分。因此，研究者提出了 Soft-margin SVM。

Soft-margin SVM 的核心就是允许 SVM 犯错，但应该尽可能少地犯错，并且依据犯错的程度进行惩罚。为此，Soft-margin SVM 在参数 $\mathbf{w}$ 和 $b$ 之外，引入了一个用于度量样本违背式（10）约束的程度的变量 $\zeta^{(i)}$，称为**松弛变量（Slack variables）**。此时，原问题的约束就由

$$
y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b) \geq 1\tag{10}
$$

变为：

$$
y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b) \geq 1-\zeta^{(i)}\tag{11}
$$

$\zeta^{(i)}$ 越大，意味着样本 $\mathbf{x}^{(i)}$ 违背式（10）约束的程度越大。当 $\zeta^{(i)}>1$ 时，样本点甚至被允许误分类到超平面的另一边。在修改了约束之后，我们需要对原问题，也即式（2）做出相应改变，得到 Soft-margin SVM 的原问题：

$$\begin{align}
\min_{\mathbf{w}, b, \mathbf{\zeta}} \frac{1}{2}\textbf{w}^T\textbf{w} + C \cdot \sum_1^N \zeta^{(i)} \quad \text{s.t.}\quad  
\left\{\begin{array}{l}
&1-y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b) - \zeta^{(i)}\leq 0\\
&-\zeta^{(i)} \leq 0
\end{array}\right.\ ,\forall i=1, \cdots, N \tag{12}
\end{align}$$

这里有两点需要注意：

1. **超参数** $C$ 用于权衡对松弛向量的惩罚，如果将 $C$ 设置为 $+\infty$ 则会退化为 Hard-margin SVM，因为目标函数无法容忍任何违背式（10）约束的情况，而当我们设定 $C$ 为一个较小值时，会允许适当地违背约束以增大间隔，但要注意 $C$ 不应设为零或负数（否则就会无视约束甚至鼓励违背约束）；
2. **松弛变量** $\zeta^{(i)} \geq 0$，否则解这个最小化问题就会让 $\zeta^{(i)} \rightarrow -\infty$，这就与我们期望 $\zeta^{(i)}$ 起到的作用相悖，不但没有放松约束，而且还加紧了。

类似 Hard-margin SVM，对于 Soft-margin SVM，我们也可以写出它的拉格朗日函数，然后求解其对偶问题：

$$\begin{aligned}
\mathcal{L}(\mathbf{w},b,\mathbf{\zeta},\mathbf{\lambda},\mathbf{\eta}) = \frac{1}{2}\textbf{w}^T\textbf{w} + C \cdot \sum_1^N \zeta^{(i)} + \sum_{i=1}^N \lambda^{(i)} 
[1-y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b)-\zeta^{(i)}] + \sum_{i=1}^N \eta^{(i)} (-\zeta^{(i)})
\end{aligned}$$

对偶问题为：

$$\begin{align}
\max_{\mathbf{\lambda}, \mathbf{\eta}} \min_{\mathbf{w}, b, \mathbf{\zeta}} \mathcal{L}(\mathbf{w},b,\mathbf{\zeta},\mathbf{\lambda},\mathbf{\eta}) \quad \text{s.t.} \quad \lambda^{(i)} \geq 0\ ,\forall i=1, \cdots, N\tag{13}
\end{align}$$

令偏导数为零可得：

$$\begin{align}
\frac{\partial \mathcal{L}(\mathbf{w},b,\mathbf{\zeta},\mathbf{\lambda},\mathbf{\eta})}{\partial \mathbf{w}} \triangleq 0
\Rightarrow w^* = \sum_{i=1}^N \lambda^{(i)} y^{(i)}\mathbf{x}^{(i)} \tag{14}
\end{align}$$

$$\begin{align}
\frac{\mathcal{L}(\mathbf{w},b,\mathbf{\zeta},\mathbf{\lambda},\mathbf{\eta})}{\partial b} \triangleq 0
\Rightarrow \sum_{i=1}^N \lambda^{(i)} y^{(i)} = 0
\tag{15}
\end{align}$$

$$\begin{align}
\frac{\partial \mathcal{L}(\mathbf{w},b,\mathbf{\zeta},\mathbf{\lambda},\mathbf{\eta})}{\partial \zeta^{(i)}} \triangleq 0
\Rightarrow \lambda^{(i)}+\eta^{(i)}=C
\tag{16}
\end{align}$$

由式（16）可知 $\eta^{(i)} = C - \lambda^{(i)}$，又因为存在约束 $\eta^{(i)} \geq 0$，所以可以把约束写成 $0 \leq \lambda^{(i)} \leq C$ 消去变量 $\mathbf{\eta}$。代入式（13）消除掉除 $\mathbf{\lambda}$ 以外的所有变量，得到 Soft-margin SVM 的对偶型：

$$\begin{align}
\min_{\mathbf{\lambda}} \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \lambda^{(i)} \lambda^{(j)} y^{(i)} y^{(j)} \mathbf{x}^{{(i)}^T} \mathbf{x}^{(j)} - \sum_{i=1}^N \lambda^{(i)} \quad\quad \text{s.t.} \quad 
\left\{\begin{array}{l}
&0 \leq \lambda^{(i)} \leq C\\
&\sum_{i=1}^N \lambda^{(i)} y^{(i)} = 0
\end{array}\right.\ ,\forall i=1, \cdots, N
\tag{17}
\end{align}$$

**Note**：$0 \leq \lambda^{(i)} \leq C$ 这个约束也被称为 **box constraint**，因为 $\lambda^{(i)}$ 被限制在了边长为 $C$ 的“盒子”里。

最后，有的文献对式（12）的原问题会给出另外一种写法：

$$\begin{align}
\min_{\mathbf{w}, b} \frac{1}{N} \sum_{i=1}^{N} \max (0,1-y_{i}(\mathbf{w}^T \mathbf{x}^{(i)})+b) + \frac{C}{2}\textbf{w}^T\textbf{w}\tag{18}
\end{align}$$

它与式（12）是等价的，实质是令松弛变量 $\zeta^{(i)}=\max (0,1-y^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)})+b)$。式（18）的第一项称为经验风险，采用 **铰链损失（hinge loss）** 作为损失函数，度量了模型对训练数据的拟合程度；第二项称为结构风险，削减了假设空间，可以避免过拟合发生；超参数 $C$ 用于权衡两种风险。这是对 Soft-margin SVM 的另一种理解。从这个损失函数出发，我们也可以直接使用 SGD 等方法进行优化，使得 SVM 可适用于流数据场景。

## SMO

尽管前面提到的对偶型都是凸二次规划问题，可以使用一些 QP 工具包（比如：CVXOPT）直接求解，但实践中一般会使用更高效的 **SMO（sequential minimal optimization）算法**或者其它变体。主要的原因就是直接求解需要进行矩阵乘法，计算和存储开销都是 O(N^2) 的，如果数据集比较大那么直接求解是不太实际的（特别是对于 SVM 被提出时的年代），SMO 可以帮助我们避免这个问题。

首先提一下 **坐标上升/下降（Coordinate ascent/descent）** 算法，它是解决多变量优化问题的一个比较常用的方法。思路很简单，就是将一个优化问题简化为多个子优化问题。假设存在 $N$ 个变量，每次更新参数时固定住其中的 $N-1$ 个，此时优化问题就变为余下那一个变量的单变量优化问题。SMO 的思路是类似的，在 SVM 中我们要解决的优化问题是式（17）：

$$\begin{aligned}
\min_{\mathbf{\lambda}} \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \lambda^{(i)} \lambda^{(j)} y^{(i)} y^{(j)} \mathbf{x}^{{(i)}^T} \mathbf{x}^{(j)} - \sum_{i=1}^N \lambda^{(i)} \quad\quad \text{s.t.} \quad 
\left\{\begin{array}{l}
&0 \leq \lambda^{(i)} \leq C\\
&\sum_{i=1}^N \lambda^{(i)} y^{(i)} = 0
\end{array}\right.\ ,\forall i=1, \cdots, N
\end{aligned}$$

这个优化问题要求解 $N$ 个变量，但不同的是这里存在约束条件。由于存在 $\sum_{i=1}^N \lambda^{(i)} y^{(i)} = 0$ 这一约束，如果我们像坐标下降那样只修改一个变量 $\lambda^{(i)}$ 的值，就必然会违背约束。为了解决这个问题，我们可以每次修改两个变量值 $\lambda^{(i)}$ 和 $\lambda^{(j)}$，并且确保它们符合约束：

$$\begin{align}
\left\{\begin{array}{l}
&0 \leq \lambda^{(i)} \leq C\\
&0 \leq \lambda^{(j)} \leq C\\
&\lambda^{(i)}y^{(i)}+\lambda^{(j)}y^{(j)}=\lambda^{(i)}y^{(i)}+\lambda^{(j)}y^{(j)}-\sum_{k=1}^N \lambda^{(k)} y^{(k)} \triangleq K
\end{array}\right.\tag{19}
\end{align}$$

画成图其实就是这样：

<img src="https://pic4.zhimg.com/80/v2-cc2bd3b1348c492ae4a02dff362f27b6.png" style="zoom:50%" />

可行的 $\lambda^{(i)}$ 和 $\lambda^{(j)}$ 都必须在 $[0, C]$ 的“盒子”内，并且在直线 $\lambda^{(i)}y^{(i)}+\lambda^{(j)}y^{(j)}=K$ 上。直线与“盒子”的交点会给 $\lambda^{(j)}$ 确定上下界，假设分别是 $L$ 和 $H$（根据直线方程会出现不同情况，比如直线刚好是“盒子”的对角线，那就会有 $L=0, H=C$）。由直线方程可知：

$$\begin{align}
\lambda^{(i)} 
&= \frac{(K-\lambda^{(j)}y^{(j)})}{y^{(i)}}\\
&= \frac{(K-\lambda^{(j)}y^{(j)})}{y^{(i)}} \cdot y^{(i)^2} \quad y^{(i)}\in\{-1, +1\}\text{, 所以乘上 }y^{(i)^2}=1\text{ 不会改变}\\
&= y^{(i)}(K-\lambda^{(j)}y^{(j)})\tag{20}
\end{align}$$

将式（20）代入到式（17）中，我们就得到了一个单变量优化问题，令目标函数对 $\lambda^{(j)}$ 的偏导数为零，即可求出最优的 $\lambda^{(j)}$。当然，为了满足约束，$\lambda^{(j)}$ 必须在 $[L, H]$ 区间内，所以还需要进行**裁剪（clipping）**，将得到的 $\lambda^{(j)}$ 代入到式（20）即可得到 $\lambda^{(i)}$。重复多次选取 $\lambda^{(i)}$ 和 $\lambda^{(j)}$，并求解子问题就构成了 SMO 算法。

最后，还有三个值得注意的点：

第一个是**如何初始化 $\mathbf{\lambda}$**？SMO 是直接将全部的拉格朗日因子初始化为0。

第二个是**如何选取** $\lambda^{(i)}$ 和 $\lambda^{(j)}$？最简单的当然是随机选取，但是实践中为了加快收敛速度，一般会采用某种启发式使得每次对目标函数的改变能最大化，比方说选取违背 KKT 条件最大的 $\lambda^{(i)}$，然后再选取与 $\lambda^{(i)}$ 间隔最大的 $\lambda^{(j)}$。

第三个是**如何判断收敛**？从前面的推导我们已经知道了 SVM 的最优解必然满足 KKT 条件，实践中我们会设定一个阈值 $\epsilon$，如果对 KKT 条件中互补松弛性质违背的程度低于 $\epsilon$，就可以认为算法已经收敛到最优解了。

SMO 算法高效的秘诀一个在于它将问题简化为容易解决的子问题，另一个就是优化子问题时选取更新变量的启发式对收敛起到了加速作用。又因为 SMO 每次只选择两个拉格朗日乘子进行更新，所以计算开销和存储开销的问题都能大大缓解。

在实践中，SMO 算法还有很多小的细节，这里只关注思路而不一一赘述。

## 总结

最后总结一下：

- SVM 本质上是从几何的角度切入，期望找出最鲁棒的一个分类超平面，把这个目标转化为了求最大间隔的问题。

- 在最大化间隔的同时对分类结果进行约束就得到了 SVM 的原问题，又因为对偶问题有着更优良的性质，所以我们更倾向于求解对偶问题。

- Slater's 条件给出的强对偶性表明求解对偶问题就可以得到原问题的解，而 KKT 条件则允许我们使用拉格朗日乘子法来求解这个带不等式约束的问题，给出了最优解需要满足的性质。

- 使用拉格朗日乘子法消除掉其它变量后得到的对偶型中存在形如 $\mathbf{x}^{(i)^T}\mathbf{x}^{(j)}$ 的内积项，因此可以方便地引入核函数，利用核技巧将 SVM 拓展为非线性分类模型。

- 为了避免过拟合，可以允许一部分样本违背原问题的约束，甚至误分类，这就是软间隔。我们引入了松弛变量 $\mathbf{\zeta}$ 来刻画样本违背约束的程度，并在新的目标函数中对其施加惩罚，然后同样可以推导出对偶型并求解对偶问题。

- 由于使用求解器的计算&存储开销较大，所以实践中常用 SMO 算法求解对偶问题。SMO 采用的是一种类似坐标上升的思路，但为了保证约束条件成立，会每次选取两个拉格朗日乘子进行优化。

## 参考

1. [《Support Vector Machines Succinctly》](https://www.syncfusion.com/ebooks/support_vector_machines_succinctly) - Alexandre Kowalczyk, 2017（**非常推荐**）
2. [CS229 Lecture notes - Part Ⅴ Support Vector Machines](http://cs229.stanford.edu/notes/cs229-notes3.pdf) by Andrew Ng, Stanford（**经典课程**）
3. [Advanced Topics in Machine Learning: COMPGI13 - Lecture 9: Support Vector Machines](http://www.gatsby.ucl.ac.uk/~gretton/coursefiles/Slides5A.pdf) by Arthur Gretton, UCL
4. [机器学习-白板推导系列(六)-支持向量机SVM（Support Vector Machine）](https://www.bilibili.com/video/BV1Hs411w7ci) - shuhuai008 - B站（**良心 up 主**）
5. [从零推导支持向量机(SVM)](https://zhuanlan.zhihu.com/p/31652569) - 张皓的文章 - 知乎
6. [超详细SVM（支持向量机）知识点，面试官会问的都在这了](https://zhuanlan.zhihu.com/p/76946313) - 韦伟的文章 - 知乎
7. [SVM---这可能是最直白的推导了](https://zhuanlan.zhihu.com/p/86290583) - 文建华的文章 - 知乎

以上材料都是很好的参考材料，但因为作者的背景和水平各有不同，所以侧重点也会有所不同，说法可能会有出入，或者有一些小的错漏，这篇文章亦然，欢迎指正。

