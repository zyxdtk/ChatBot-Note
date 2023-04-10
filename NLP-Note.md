# NLP-Note 自然语言处理笔记

## 评价

- [困惑度 Perplexity](https://blog.csdn.net/index20001/article/details/78884646)
- [BLEU](https://blog.csdn.net/qq_21190081/article/details/53115580)
- [BLEU](https://blog.csdn.net/qq_36652619/article/details/87544918)
- [acc](https://zhuanlan.zhihu.com/p/45465603)  acc=right/all 

## NLP的用途和挑战 
为什么难
- 语言是一个交流工具，依赖真实世界、常识、上下文
- 语言是模棱两可的，ambiguous

## word2vec

- distribute similarity，分布相似性。
- 算法
  - Skip-grams(SG) 
    - 每个单词有两个向量。一个输入，一个输出。
  - Continuous Bag of Words（CBOW）
    - 加权的bow(向量除以词频)+移除special direction(预测时加上词频)
    - 负采样k的个数在10个左右
    - window lenght 一般5-10，对称
    - 内积距离，欧式距离都可以
  - Glove(目前表现最好)
    - 捕捉单词词频的总体统计数据
    - 最好的向量维度是300，window size 在8左右是最好的

看看下面的论文和中文解读可能理解起来更容易点
- Glove [NLP模型笔记】GloVe模型简介](https://blog.csdn.net/edogawachia/article/details/105804378) 相比起绝对地描述一个词语，通过与第三者的比较，得到第三者与两个词语中的哪个更接近这样一个相对的关系，来表达词语的含义，实际上更符合我们对于语言的认知。这样学习出来的vector space具有一个meaningful substructure。
- [dav-word2vec](https://github.com/dav/word2vec) [【汉】](https://www.jianshu.com/p/471d9bfbd72f) google的word2vec

## Word Window
这里有一些比较基础的softmax梯度计算的问题
 - [softmax、cross entropy和softmax loss学习笔记](https://www.cnblogs.com/smartwhite/p/8601477.html) softmax loss 和 cross entropy 其实是一样的
 - [【技术综述】一文道尽softmax loss及其变种](https://zhuanlan.zhihu.com/p/34044634) 

知识点
- 矩阵运算比for循环快了12倍
- DL为什么需要非线性。如果没有非线性单元，他其实只是一个线性转化。


## 学习资料

- [【李宏毅】自然语言处理(2020)B站视频](https://www.bilibili.com/video/BV1wE411W7TV?p=2)
- [专题-句向量（Sentence Embedding）](https://github.com/DarLiner/Algorithm_Interview_Notes-Chinese/blob/master/B-%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86/B-%E4%B8%93%E9%A2%98-%E5%8F%A5%E5%B5%8C%E5%85%A5.md)
- [NLP-词向量](https://github.com/DarLiner/Algorithm_Interview_Notes-Chinese/blob/master/B-%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86/B-%E4%B8%93%E9%A2%98-%E8%AF%8D%E5%90%91%E9%87%8F.md)
- [【官方】【中英】CS224n 斯坦福深度自然语言处理课 B站视频](https://www.bilibili.com/video/BV1pt411h7aT?from=search&seid=335450819919994778) 英文还是优点吃不消
- [NLP-NLP基础](https://github.com/DarLiner/Algorithm_Interview_Notes-Chinese/blob/master/B-%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86/A-%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E5%9F%BA%E7%A1%80.md)
- https://github.com/leerumor/ai-study


- https://kexue.fm/archives/9529
- https://github.com/TingFree/NLPer-Arsenal  收录NLP竞赛策略实现、各任务baseline、相关竞赛经验贴（当前赛事、往期赛事、训练赛）、NLP会议时间、常用自媒体、GPU推荐等
- www.sbert.net