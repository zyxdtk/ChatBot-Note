# 1. GCN-Note
图神经网络(Graph Neural Networks，GNN)

- [1. GCN-Note](#1-gcn-note)
  - [1.1. 图神经网络介绍](#11-图神经网络介绍)
  - [1.2. 关键知识点](#12-关键知识点)
  - [1.3. 基础知识](#13-基础知识)
  - [1.4. 学习资料](#14-学习资料)

## 1.1. 图神经网络介绍

21世纪AI浪潮的兴起归因于三个条件：
- 算力(GPU)
- 数据(互联网沉淀的大量训练数据)
- 深度学习(从欧氏空间数据中提取潜在特征的有效性)

图嵌入(Graph Embedding)的算法大致分为三个类别：
- 矩阵分解
- 随机游走
- 深度学习(属于GNN)
  - autoencoder: DNGR、SDNE
  - Graph Convolution Networks: GraphSage 

图神经网络可以分为五大类别：
- 图卷积网络 （Graph Convolution Networks，GCN），基础结构
  - 谱 spectral-based 傅里叶变换+滤波器，只能用于无向图
    - Spectral CNN 
    - Chebyshev Spectral CNN 
    - Adaptive Graph Convolution Networks AGCN
  - 空间 spatial-based 从邻域聚合特征信息，更灵活高效
    - Recurrent-based
    - Composition-based
- 图注意力网络（Graph Attention Networks）
  - Graph Attention Network,GAT,是基于空间的GCN，用attention确定邻居节点的权重
  - Gated Attention Netw,GAAN,用到了multi-head机制
  - Graph Attention Model, GAM, 用到了lstm，解决图形分类问题
- 图自编码器（ Graph Autoencoders），邻接矩阵稀疏
  - Graph Autoencoder (GAE) 对邻接矩阵的项重加权
  - Adversarially Regularized Graph Autoencoder (ARGA)
  - Network Representations with Adversarially Regularized Autoencoders (NetRA) 将图线性化为序列
  - Deep Recursive Network Embedding (DRNE)
  - Deep Neural Networks for Graph Representations (DNGR) 重构一个更密集的矩阵PPMI
  - Structural Deep Network Embedding (SDNE) 对0项进行惩罚
- 图生成网络（ Graph Generative Networks）基于领域的，生成新图
  - Molecular Generative Adversarial Networks (MolGAN)  用到了GCN、GAN、RL
  - Deep Generative Models of Graphs (DGMG) 基于spatial-based gcn得到图的隐式表达，递归的在图中生成节点和变
  - GraphRNN
  - NetGAN
- 图时空网络（Graph Spatial-temporal Networks） 例如交通网络
  - Diffusion Convolutional Recurrent Neural Network (DCRNN)
  - CNN-GCN 
  - Spatial Temporal GCN (ST-GCN)
  - Structural-RNN

图神经网络的应用
- 计算机视觉
- 推荐系统
- 交通
- 化学 chemistry

## 1.2. 关键知识点

- 怎么表征图
  - 邻接矩阵
- 怎么学习图结构信息 
  - Message Passing Framework，用节点周围的邻居信息来更新节点本身的向量。这里GNN学习到两种信息：结构信息、feature-based信息。操作上分为Aggregate和Update，$h_u^{(k)}=\sigma(W_{self}^{(k)}h_u^{(k-1)}+W_{neigh}^{(k)}\sum_{v\epsilon N(u)}{h_v^{(k-1)} + b^{(k)}})$ ，有一个简化是做一个self loop，这样之后不用单独搞一个$W_{self}$
- 怎么聚合邻居信息呢  Aggregate
  - 标准化。求和是不稳定的，至少要做平均，更好的方案是对称标准化。$m_{N(u)} = \sum_{v\epsilon N(u)}{\frac {h_v}{\sqrt{|N(u)||N(v)|}}}$ ，GCN用到了标准化和self loop优化
  - polling,邻居节点是无序的，只要满足置换不变性的聚合器都可以用于gnn，所以其实可以用max、min这样的池化操作，但是这样容易过拟合。
  - Janossy polling, 先构造一些邻居节点的序，然后用置换敏感的函数处理这些序，最后对结果做聚合。这里的置换敏感函数可以用LSTM等。序的构造有两种方案：随机采样，按照典型特征排序(如，按照度降序)加上一点随机。
  - Neighborhood Attention，GAT里面用到了。也可以用multi-head。attention 是一个提升GNN模型表达能力的有效手段。
- 怎么更新节点向量 Update
  - Over-smoothing 过平滑问题，随着迭代次数的增加，节点向量可能趋同。
  - concatenations or skip connections, 直接把上一次的node verctor与更新之后的concat到一起，把本次更新的邻居信息和node本身的信息分开。或者类似resnet，把node和邻居的向量做一个加权和。这个不仅可以缓解过平滑问题，还能让dnn更深。
  - Gated updates 基于门的更新。例如用GRU
  - Jumping Knowledge Connections 跳跃知识连接。就是把过往的每次迭代生成的向量，拼接到一起，然后再生成最终的节点向量。
- 边特征和多关系GNN，怎么处理异构网络、多关系网络
  - Relational Graph Neural Networks RGCN，为每个关系都指定一个转换矩阵。$m_{N(u)}=\sum_{\tau \epsilon R}\sum_{v\epsilon N_\tau(u)} {\frac{W_\tau h_v}{f_n(N(u),N(v))}}$, 这个原生RGCN的问题是参数量大容易过拟合，一种解决思路是参数共享，$W_\tau = \sum_{i=1}^b {\alpha _{i,\tau}B_i}$
  - Attention和feature concatenation，把邻居节点和与邻居的多个边concat到一起，然后用attention把他们聚合起来。
- Graph Polling 图池化，对应于想要学习整个图的emb
  - 直接把node emb加和，或者做标准化
  - 用LSTM和attention的组合来聚合node emb，用一个query向量来对所有节点做一个加权求和得到输出，输出再和query一起过一个lstm得到下一个query。
  - Graph coarsening approaches 图粗化方法。池化会丢失图结构信息，为了引入图结构信息，一种方式是通过图聚类或者粗化。不断的聚类减少图的size，学习新图的向量。


## 1.3. 基础知识

线性代数
- [拉普拉斯矩阵与正则化](https://blog.csdn.net/weixin_42973678/article/details/107190663) L=D-A, L是拉普拉斯矩阵，D是度矩阵，A是邻接矩阵。正则化 $L^{sym} = D^{-1/2}AD^{-1/2}$ 本质的意义就是，把邻接矩阵的对角线用1代替，其他表示边的1，用该边所连接的两个顶点的度数乘积的-1/2 次方的相反数代替。
- [实对称半正定](https://zhuanlan.zhihu.com/p/44860862) 


## 1.4. 学习资料

- [2019]. [A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/abs/1901.00596) [【汉】](https://zhuanlan.zhihu.com/p/75307407) 图神经网络综述
- [2020]. [《Deep Learning on Graphs》](https://web.njit.edu/~ym329/dlg_book/) 看的英文pdf，挺好的教材，浅显易懂
    - [《Graph Representation Learning Book》](https://www.cs.mcgill.ca/~wlh/grl_book/) [【本地PDF】](static/pdf/GRL_Book.pdf) 
    - [《图深度学习》](https://item.jd.com/13221338.html?cu=true&utm_source=kong&utm_medium=tuiguang&utm_campaign=t_2009678457_&utm_term=fcda510799ae4c6497dfaca3bc8a0c08) 
    
 

- TODO
  - [2020]. [A Practical Guide to Graph Neural Networks](https://deepai.org/publication/
  - a-practical-guide-to-graph-neural-networks) [【PDF】](static/pdf/A%20Practical%20Guide%20to%20Graph%20Neural%20Networks.pdf)
  - [2020]. [Deep Learning for Learning Graph Representations](https://arxiv.org/abs/2001.00293v1)[【PDF】](static/pdf/Deep%20Learning%20for%20Learning%20Graph%20Representations.pdf)
  - [2020]. [宾夕法尼亚大学《图神经网络》课程(2020) by Alejandro RIbeiro](https://www.bilibili.com/video/av457264185/?vd_source=72bd417d3c61f48a1851179442d7083c) 


开源笔记：
- [280] https://github.com/leerumor/ai-study