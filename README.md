# Machine_Learning

本项目主体是对周志华教授的《机器学习》一书所做的笔记，以及书中习题的试答（**周教授未提供习题的标准答案，笔者仅作试答，如有谬误，欢迎指出。习题中带\*星号的题目难度较大。**）。除此之外，本项目还会逐渐引入一些对其他精彩的机器学习相关文章的摘录与点评。想了解《机器学习》一书不妨查看周教授的[主页](http://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/MLbook2016.htm)，上面除了简介之外也包含了该书各印刷版的勘误情况。

全书共16章，因此，我的笔记也分为相应的16个章节，可以从以下目录进行快速访问：

- [01. 绪论](https://github.com/familyld/Machine_Learning/blob/master/01introduction.md)
- [02. 模型评估与选择](https://github.com/familyld/Machine_Learning/blob/master/02model_evaluation_and_model_selection.md)
- [03. 线性模型](https://github.com/familyld/Machine_Learning/blob/master/03linear_model.md)
- [04. 决策树](https://github.com/familyld/Machine_Learning/blob/master/04decision_tree.md)
- [05. 神经网络](https://github.com/familyld/Machine_Learning/blob/master/05neural_network.md)
- [06. 支持向量机](https://github.com/familyld/Machine_Learning/blob/master/06support_vector_machine.md)
- [07. 贝叶斯分类器](https://github.com/familyld/Machine_Learning/blob/master/07Bayes_classifier.md)
- [08. 集成学习](https://github.com/familyld/Machine_Learning/blob/master/08ensemble_learning.md)
- [09. 聚类](https://github.com/familyld/Machine_Learning/blob/master/09clustering.md)
- [10. 降维与度量学习](https://github.com/familyld/Machine_Learning/blob/master/10dimension_reduction_and_metric_learning.md)
- [11. 特征选择与稀疏学习](https://github.com/familyld/Machine_Learning/blob/master/11feature_selection_and_sparse_learning.md)
- [12. 计算学习理论](https://github.com/familyld/Machine_Learning/blob/master/12computational_learning_theory.md)
- [13. 半监督学习](https://github.com/familyld/Machine_Learning/blob/master/13semi-supervised_learning.md)
- [14. 概率图模型](https://github.com/familyld/Machine_Learning/blob/master/14probabilistic_graphical_model.md)
- [15. 规则学习](https://github.com/familyld/Machine_Learning/blob/master/15rule_learning.md)
- [16. 强化学习](https://github.com/familyld/Machine_Learning/blob/master/16reinforcement_learning.md)

这16个章节可以大致分为3个部分：第1部分包括第1~3章，是本书的引入部分，介绍了机器学习的一些基础知识；第2部分包括第4~10章，介绍一些经典而且常用的机器学习方法；第3部分包括第11~16章，介绍了一些进阶知识。除前3章以外，各章内容相对独立，可以根据兴趣和时间选择学习。

## 内容简介

### 绪论

本章首先讲述了什么是机器学习以及机器是如何学习的，然后引入了一些机器学习的基本概念。接下来从假设空间和归纳偏好两个方面来讲述模型的产生。最后介绍了机器学习的发展历程以及应用现状。

### 模型评估与选择

本章首先引入了经验误差和泛化误差的概念，从而很自然地引伸出了机器学习中模型选择的问题。然后通过评估方法、性能度量、比较检验三个章节来讲述模型选择的整个流程。最后还介绍了偏差-方差分解，这可以帮助我们更好地解释模型的泛化性能。

### 线性模型

本章首先通过最简单的线性回归讲述如何利用线性模型进行预测，并且使用最小二乘法来进行参数估计。接下来从单属性扩展到多属性的情形，也即多元线性回归，并进一步推广到可以求取输入空间到输出空间的非线性函数映射的广义线性模型。然后针对分类任务，介绍了两种线性分类方法——对数几率回归（逻辑回归）和线性判别分析（LDA）。接下来讨论了使用二分类模型解决多分类问题时的三种拆分策略。最后还介绍了解决类别不平衡问题的几种思路。

### 决策树

本章首先介绍了决策树模型的结构以及决策树学习的目标，然后自然地引入了在建立树结构时如何选择最优划分属性的问题，并介绍了三种最为常用的指标（信息增益、信息增益率和基尼指数）。针对过拟合问题，作者讲解了预剪枝和后剪枝这两种解决方案以及它们各自的优缺点。接下来还给出了数据集的连续值离散化以及缺失值处理的一些思路。最后简单地介绍了结合线性模型从而实现减少预测时间开销这一目的的多变量决策树模型。

### 神经网络

本章首先介绍了神经网络最基本的组成单位——神经元。然后引入了最简单的只有两层神经元的感知机，并在此基础上又引入了多层网络和多层前馈神经网络的概念。接下来介绍了神经网络的典型学习方法——BP算法，分为标准BP算法和累积BP算法两种。针对过拟合问题和陷入局部最小问题，作者给出了一些比较常见的思路。接下来作者还简单地介绍了一些其他的神经网络模型。在本章的最后，作者简要概述了今年最火的深度学习的思想，以及如何节省训练时间开销。

### 支持向量机

本章首先引入了支持向量机中最基础的两个概念——间隔和支持向量。然后介绍了如何把获取最大间隔超平面转换为对偶问题并使用SMO算法求解。接下来介绍了如何使用核函数来解决线性不可分问题以及有哪些常用的核函数。针对过拟合问题，作者介绍了软间隔这个概念以及软间隔支持向量机的求解方式，并讨论了常用的替代损失函数。接下来，作者介绍了支持向量回归以及对应的求解方法。在本章的最后，作者还介绍了核方法，也即通过引入核函数将线性学习器转换为非线性学习器的方法。

### 贝叶斯分类器

### 集成学习

### 聚类

### 降维与度量学习

### 特征选择与稀疏学习

### 计算学习理论

### 半监督学习

### 概率图模型

### 规则学习

### 强化学习
