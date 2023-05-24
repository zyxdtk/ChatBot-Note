# Tokenizer

## 算法

- [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162.pdf)
- [BPE-Dropout: Simple and Effective Subword Regularization](https://arxiv.org/pdf/1910.13267.pdf)
- [BPE]()
  - [tokenizers小结](https://zhuanlan.zhihu.com/p/360290118)
  - [BPE 算法原理及使用指南【深入浅出】](https://juejin.cn/post/7088322473640329230)
  - [A study on the evaluation of tokenizer performance in natural language processing](https://www.tandfonline.com/doi/full/10.1080/08839514.2023.2175112) 用一些下游用户评估tokenizer，发现BPE比Mecab-Ko好
  - [tokenizer_summary](https://huggingface.co/transformers/v3.0.2/tokenizer_summary.html)
## 开源工程

- [google/sentencepiece](https://github.com/google/sentencepiece)
  - [Issue: Manually modifying SentencePiece model?](https://github.com/google/sentencepiece/issues/121) 
  - [Chinese-LLaMA-Alpaca--merge_tokenizers](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/merge_tokenizers.py)
  - [Working on folders (Training) ](https://github.com/google/sentencepiece/issues/489)
  - [BPE、WordPiece和SentencePiece](https://www.jianshu.com/p/d4de091d1367)
  - [llama/tokenizer.py](https://github.com/facebookresearch/llama/blob/main/llama/tokenizer.py)
  - [options](https://github.com/google/sentencepiece/blob/master/doc/options.md)
  - [can we train by Parallel Computing or Multithreading or multi-Progress ](https://github.com/google/sentencepiece/issues/366) 2023-05-01左右的时候作者回复说会在下一个版本加入并行计算优化
- [VKCOM/YouTokenToMe](https://github.com/VKCOM/YouTokenToMe) bpe的高性能版本
  - [benchmark](https://github.com/VKCOM/YouTokenToMe/blob/master/benchmark.md)
  - [convert from sentencepiece](https://github.com/VKCOM/YouTokenToMe/issues/9) 暂时没有对sp格式的支持
- [huggingface/tokenizers](https://github.com/huggingface/tokenizers)
- [rsennrich/subword-nmt](https://github.com/rsennrich/subword-nmt/tree/master)
  - [bpe_toy.py](https://github.com/rsennrich/subword-nmt/blob/master/subword_nmt/bpe_toy.py) 用于理解bpe


