# DataSet

## 数据
### 数据集

- [msmarco](https://microsoft.github.io/msmarco/) 微软开源的深度学习数据集
- [sbert](https://www.sbert.net/examples/training/paraphrases/README.html)
- [百度千言](https://www.luge.ai/#/luge/dataDetail?id=55)
  - [DuReaderretrieval: A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine](https://arxiv.org/pdf/2203.10232.pdf)
- [openai/human-eval](https://github.com/openai/human-eval) 评估代码语言模型
- [CHINESE OPEN INSTRUCTION GENERALIST: A PRELIMINARY RELEASE](https://arxiv.org/pdf/2304.07987.pdf)
- [T2Ranking: A large-scale Chinese Benchmark for Passage Ranking](https://arxiv.org/pdf/2304.03679.pdf)   [THUIR/T2Ranking](https://github.com/THUIR/T2Ranking/)

- [UER:About Open Source Pre-training Model Framework in PyTorch & Pre-trained Model Zoo](https://github.com/dbiir/UER-py)
- [huggingface-transformers](https://huggingface.co/docs/transformers/model_summary)


语义相似度
- [CLUEbenchmark/SimCLUE](https://github.com/CLUEbenchmark/SimCLUE)
- [shibing624/nli_zh](https://huggingface.co/datasets/shibing624/nli_zh)
- [liucongg/NLPDataSet](https://github.com/liucongg/NLPDataSet)
- [千言数据集：文本相似度](https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition)
- [DMetaSoul/chinese-semantic-textual-similarity](https://huggingface.co/datasets/DMetaSoul/chinese-semantic-textual-similarity)

大模型数据集
- [togethercomputer/RedPajama-Data](https://github.com/togethercomputer/RedPajama-Data)
  - [RedPajama-Data-CommonCrawl](https://github.com/togethercomputer/RedPajama-Data/tree/main/data_prep/cc)
- [c4](https://paperswithcode.com/dataset/c4)
  - [2022] [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683v3.pdf)
  - [huggingface/c4](https://huggingface.co/datasets/allenai/c4#license)
- [2020] [The Pushshift Reddit Dataset](https://arxiv.org/abs/2001.08435)

### Benchmark

- [BEIR](https://github.com/beir-cellar/beir) 信息检索Benchmark
- [CLUEbenchmark](https://github.com/CLUEbenchmark) 中文语言理解测评基准
- [Xiezhi: An Ever-Updating Benchmark for Holistic Domain Knowledge Evaluation](https://arxiv.org/pdf/2306.05783.pdf)
  -  [XiezhiBenchmark](https://github.com/MikeGu721/XiezhiBenchmark)
- [MMLU](https://github.com/hendrycks/test) 大量多任务语言理解，57个任务
  - [MEASURING MASSIVE MULTITASK LANGUAGE UNDERSTANDING](https://arxiv.org/pdf/2009.03300.pdf) 
- [Leaderboard - C-Eval](https://cevalbenchmark.com/static/leaderboard.html) 中文评估集
- [GSM8K](https://github.com/openai/grade-school-math) 研究生数学题
- [BBH-HARD](https://github.com/suzgunmirac/BIG-Bench-Hard)
  - [Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them](https://arxiv.org/abs/2210.09261)
## 计算框架

- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 
- [ColossalAI](https://github.com/hpcaitech/ColossalAI)

优化算法
- [平行策略](hhttps://huggingface.co/transformers/v4.9.0/parallelism.html)
  - Data Parallelism
  - [Pipeline Parallelism](https://pytorch.org/docs/stable/pipeline.html) pipeline parallelism splits the input minibatch into multiple microbatches and pipelines the execution of these microbatches across multiple GPUs.
  - [Tensor Parallelism](https://pytorch.org/docs/stable/distributed.tensor.parallel.html) Rowwise, Colwise and Pairwise Parallelism.
    - [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
  - Sequence Parallelism
  - [Zero Redundancy Optimizer (ZeRO)](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)
  - Auto-Parallelism
- [flash-attention](https://github.com/HazyResearch/flash-attention) Fast and memory-efficient exact attention
- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/pdf/2203.03466.pdf)
- [如何用数据并行训练万亿参数模型？](https://zhuanlan.zhihu.com/p/402232568)

- [Pytorch 分布式训练](https://zhuanlan.zhihu.com/p/76638962)


- [Billion-scale similarity search with GPUs](https://arxiv.org/pdf/1702.08734.pdf)
## Perf

- [MLCommons](https://github.com/orgs/mlcommons/repositories?type=all)  The mission of MLCommons™ is to make machine learning better for everyone.
  - [2022:We’re Training AI Twice as Fast This Year as Last](https://spectrum.ieee.org/mlperf-rankings-2022)


## 数据清洗
大量、优质、非重复的数据。

- [2019] [CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data](https://arxiv.org/abs/1911.00359)
  - [facebookresearch/cc_net](https://github.com/facebookresearch/cc_net)
  - [kpu/kenlm](https://github.com/kpu/kenlm)
  - [google/sentencepiece](https://github.com/google/sentencepiece)
- [ngram语言模型—基于KneserNey及Modified Kneser Ney平滑](https://blog.csdn.net/weixin_42498517/article/details/103608763)
- [The RefinedWeb Dataset for Falcon LLM:Outperforming Curated Corpora with Web Data, and Web Data Only](https://arxiv.org/pdf/2306.01116.pdf)
  - [the pile](https://pile.eleuther.ai/)
  - [Quality at a Glance:An Audit of Web-Crawled Multilingual Datasets](https://arxiv.org/pdf/2103.12028.pdf) 小语种质量差，用数据之前最好抽样100条看看。
  - [CopyCat: Near-Duplicates Within and Between the ClueWeb and the Common Crawl](https://dl.acm.org/doi/10.1145/3404835.3463246) Our analysis shows that 14--52, of the documents within a crawl and around~0.7--2.5, between the crawls are near-duplicates.
  - [2021] [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805) 通过prompt从LLM提取训练数据，大模型比小模型更容易被攻击。
  - [2023] [Quantifying Memorization Across Neural Language Models](https://arxiv.org/abs/2202.07646) 6B的GPT-J模型至少记忆了Pile中1%的数据。模型越大、数据重复越多、上下文越长的case越容易记忆。对训练数据做去重可以减少记忆带来的危害。
- [2022] [DeepMind: Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446) 在152种任务上测试发现：规模对阅读理解、事实检查、毒性鉴别有用，对逻辑推理、数学推理作用较小。用了一堆启发式规则来过滤低质。用goggle的安全搜索网页过滤内容。

### 去重

- [ChenghaoMou/text-dedup](https://github.com/ChenghaoMou/text-dedup) 对比了多种方法，minhash最好
- [2022] [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) 去重可以避免记忆，用到精确子串去重+minhash
  - [deduplicate-text-datasetsr](https://github.com/google-research/deduplicate-text-datasets)
- minhash+lsh
  - [ekzhu/datasketch](https://github.com/ekzhu/datasketch) 实现了minhash等多种去重算法
  - [bigcode-dataset/near_deduplication](https://github.com/bigcode-project/bigcode-dataset/tree/main/near_deduplication) minhash+lsh+基于连通图去重更多
    - [Large-scale Near-deduplication Behind BigCode](https://chenghaomou.github.io/posts/20230220150602)
  - [Document deduplication using Locality Sensitive Hashing (LSH) with minhash](https://github.com/mattilyra/LSH/blob/master/examples/Introduction.ipynb)
  - [MinHashLSH](https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.ml.feature.MinHashLSH.html)
  - [Uber实践基于局部敏感哈希LSH](https://blog.csdn.net/sinat_15443203/article/details/83756187)
  - [non_dialogue_minhash_homework](https://gitlab.xaminim.com/nlp/daodao/-/blob/feat/zhanglingyu/data_clean/solution/pretrain/data_clean/non_dialogue_data_clean/non_dialogue_minhash_homework.py) 灵玉的minhash+lsh


  
## 数据可视化

- [hazyresearch/meerkat](https://github.com/hazyresearch/meerkat) 基础大模型的数据可视化

## 重采样

- [DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining](https://arxiv.org/pdf/2305.10429.pdf)
- [2023] [Data Selection for Language Models via Importance Resampling](https://arxiv.org/pdf/2302.03169.pdf)


