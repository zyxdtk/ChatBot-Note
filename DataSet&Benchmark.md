# DataSet


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


### Benchmark

- [BEIR](https://github.com/beir-cellar/beir) 信息检索Benchmark
- [CLUEbenchmark](https://github.com/CLUEbenchmark) 中文语言理解测评基准


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


## 分词

- [rsennrich/subword-nmt](https://github.com/rsennrich/subword-nmt)