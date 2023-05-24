# 文本向量

## 用途

## 数据集和Benchmark

## 数据集

英文
- [2014] [SemEval-2014 Task 10: Multilingual Semantic Textual Similarity](https://aclanthology.org/S14-2010/)
- [msmarco](https://microsoft.github.io/msmarco/) 微软开源的深度学习数据集
- [openai/human-eval](https://github.com/openai/human-eval) 评估代码语言模型

- [UER:About Open Source Pre-training Model Framework in PyTorch & Pre-trained Model Zoo](https://github.com/dbiir/UER-py)
- [Making a MIRACL: Multilingual Information Retrieval Across a Continuum of Languages](https://arxiv.org/abs/2210.09984) 支持18种语言，77k的query
- [2021] [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663)

中文
- [百度千言](https://www.luge.ai/#/luge/dataDetail?id=55)
  - [DuReaderretrieval: A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine](https://arxiv.org/pdf/2203.10232.pdf)
- [CHINESE OPEN INSTRUCTION GENERALIST: A PRELIMINARY RELEASE](https://arxiv.org/pdf/2304.07987.pdf)
- [T2Ranking: A large-scale Chinese Benchmark for Passage Ranking](https://arxiv.org/pdf/2304.03679.pdf)   [THUIR/T2Ranking](https://github.com/THUIR/T2Ranking/)
- [Toyhom/Chinese-medical-dialogue-data](https://github.com/Toyhom/Chinese-medical-dialogue-data) 中文医疗对话数据集
- [Alibaba-NLP/Multi-CPR](https://github.com/Alibaba-NLP/Multi-CPR) 阿里开源的多领域检索数据集，电商、医疗、娱乐等。query10w
  - [Multi-CPR: A Multi Domain Chinese Dataset for Passage Retrieval](https://arxiv.org/abs/2203.03367)  
- [shibing624/nli_zh ](https://huggingface.co/datasets/shibing624/nli_zh) text2vec用到的数据集
- [CLUE-QQ浏览器搜索相关性数据集](https://modelscope.cn/datasets/damo/QBQTC/summary)

### Benchmark

英文
- [sbert](https://www.sbert.net/docs/pretrained_models.html) 有基于bert训练的一些句向量模型的Benchmark,核心是英文也有通过蒸馏得到的多语言版本
  - [openai-embedding的性能讨论](https://github.com/UKPLab/sentence-transformers/issues/1897)
    - [openai-text-embedding-v1与Google、Sentence-Transformer的对比评测](https://medium.com/@nils_reimers/openai-gpt-3-text-embeddings-really-a-new-state-of-the-art-in-dense-text-embeddings-6571fe3ec9d9) openai的v1版本Embedding性能不是最优
- [MTEB: Massive Text Embedding Benchmark](https://github.com/embeddings-benchmark/mteb#leaderboard)
  - [MTED评测结果](https://huggingface.co/spaces/mteb/leaderboard) openai的ada02变现很好，但是不是最优
- [SimCSE](https://github.com/princeton-nlp/SimCSE#model-list)

中文
- [shibing624/text2vec](https://github.com/shibing624/text2vec) 中文的文本向量模型Benchmark

  

## 算法和开源工具

### Transformer算法

- [One Embedder, Any Task: Instruction-Finetuned Text Embeddings](https://arxiv.org/abs/2212.09741) sota，用到了指令，模型5G
  - [HKUNLP/instructor-embedding](https://github.com/HKUNLP/instructor-embedding)
  - [huggingface/instructor-xl](https://huggingface.co/hkunlp/instructor-xl/tree/main)
- [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813) 用蒸馏的思路做多语音Embedding
  - [sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) 1.1G, 768 dim

- [2019 ][Multilingual Universal Sentence Encoder for Semantic Retrieval](https://arxiv.org/pdf/1907.04307.pdf) google出的多语音句向量模型。
- [2019] [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/pdf/1908.10084.pdf) 用mean polling, 用到TripletLoss
- [2020] [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2004.04906.pdf)
  - [facebookresearch/DPR](https://github.com/facebookresearch/DPR)
- [2021] [EMNLP'2021: SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://github.com/princeton-nlp/SimCSE) [论文](https://arxiv.org/pdf/2104.08821.pdf)
  - [SimCSE](https://github.com/princeton-nlp/SimCSE#model-list)
- [SGPT: GPT Sentence Embeddings for Semantic Search](https://arxiv.org/pdf/2202.08904v5.pdf)
- [2022] [Text and Code Embeddings by Contrastive Pre-Training](https://arxiv.org/pdf/2201.10005.pdf)   contrastive pre-training on unsupervised data at scale leads to high quality vector representations of text and code

- [Title:Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)  [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- [Contrastive-Learning-NLP-Papers](https://github.com/ryanzhumich/Contrastive-Learning-NLP-Papers) Paper List for Contrastive Learning for Natural Language Processing

- [Contrastive Code Representation Learning: functionality-based JavaScript embeddings through self-supervised learning](https://github.com/parasj/contracode)
- [Multilingual Sentence & Image Embeddings with BERT](https://github.com/UKPLab/sentence-transformers)
- [DPTDR: Deep Prompt Tuning for Dense Passage Retrieval](https://arxiv.org/pdf/2208.11503.pdf)
- [RocketQA](https://github.com/PaddlePaddle/RocketQA)
- [RocketQAv2: A Joint Training Method for Dense Passage Retrieval
and Passage Re-ranking](https://arxiv.org/pdf/2110.07367.pdf)

### 训练技巧

- 如何处理False Negative
  - [推荐系统召回模型batch内负采样训练时出现false negative问题的一些解决方案](https://zhuanlan.zhihu.com/p/613206891) GHM 梯度调和法
  - [端到端问答新突破：百度提出RocketQA 精](https://ai.baidu.com/forum/topic/show/972410) 跨批次负采样
### 开源工具和产品

- [openai-embedding](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) 只在英文上表现不错，在其他语言上表现比MB25差
- [huggingface/transformers](https://github.com/huggingface/transformers) 跟Transformer有关的各种模型。
- [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) 基于BERT的多语言的句子、图像向量
  - [用1B对比对训练一个最优Embedding模型](https://discuss.huggingface.co/t/train-the-best-sentence-embedding-model-ever-with-1b-training-pairs/7354) 更多的数据和更大的BatchSize
  - [microsoft/mpnet-base](https://huggingface.co/microsoft/mpnet-base) sbert认为最好的预训练模型
- [shibing624/text2vec](https://github.com/shibing624/text2vec) 文本向量工具，实现了Word2Vec、RankBM25、Sentence-BERT、CoSENT等文本表征、文本相似度计算模型，开箱即用。也可以用huggineface、sentence-transformers 调用。
- [cohere](https://docs.cohere.com/docs/cross-lingual-content-moderation) 提供支持100+种语言的Embedding