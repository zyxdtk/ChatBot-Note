

- [1. 大纲 Deep Learning for Human Language Processing](#1-大纲-deep-learning-for-human-language-processing)
- [2. 语音识别 Speech Recognition](#2-语音识别-speech-recognition)
  - [2.1. LAS listen,attend,spell](#21-las-listenattendspell)
  - [2.2. CTC Connectionist Temporal Classification](#22-ctc-connectionist-temporal-classification)
  - [2.3. HMM](#23-hmm)
  - [2.4. Alignment](#24-alignment)
  - [2.5. RNN-T](#25-rnn-t)
  - [2.6. Language Modeling](#26-language-modeling)
- [3. Voice Conversion](#3-voice-conversion)
- [4. Speech Separation](#4-speech-separation)



# 1. 大纲 Deep Learning for Human Language Processing

[courses_DLHLP20](http://speech.ee.ntu.edu.tw/~tlkagk/courses_DLHLP20.html)
[B站-[李宏毅]自然语言处理(2020)](https://www.bilibili.com/video/BV1wE411W7TV?p=1&vd_source=72bd417d3c61f48a1851179442d7083c)

- 课程内容：Text vs Speech = 5:5
  - 只有56%的语言有文字
  - 语音很复杂。同一个人不可能说同样的话。
  - 文字很复杂。最长的英文句子是13955。
- 6类任务：
  - 语音转文字 ASR(Auto Speech Recognition)
    - 传统语音识别。
      - 用声音语料训练声音模型(acoustic model)
      - 用户文本语料学习语言模型(language model)
      - 然后有一个lexicon(词典)把声音和文字对应起来。
      - 首先将声音信号处理得到特征向量，然后找到匹配的声音，然后通过查字典得到对应的文本。 
    - 模型可以达到2GB大小，预测可以压缩到80M放在手机里面
  - 语音到语音 
    - Speech Separation 可以识别不同的人的语音
    - Voice Conversion 语音转换
      - Unsupervised Voice Conversion(One Shot learning)
  - 语音到类别
    - Speaker Recognition 
    - Keyword Spotting 关键词发现(用于唤醒词 Wake up words )
  - 文字转语音 TTS(Text To Speech)
    - 可用DNN来搞定，但是在某些情况下会有一些badcase需要解决。
  - 文本到文本
    - Translation 
    - Summarization
    - Chat-bot 
    - Question Answering
    - 语法解析 syntactic parsing
  - 文本到类别
- 另外会讲的内容
  - 会讲到 meta learning 
  - 从无监督数据中学习
  - adversarial attack 对手攻击
  - explainable ai 解释AI

语言模型越来越大
- ELMO 94M
- BERT 340M
- GPT-2 1542M
- Megatron 8G
- T5 11G

文本生成
- autoregression 自回归
- 非自回归

# 2. 语音识别 Speech Recognition

- 描述SR问题
  - speech  一个vector的序列 (长度T，维度d)
    - Phoneme 发声的基本单位。需要语言学的知识，来创建lexicon：word to phonemes(32%)
  - text  一个token的序列 (长度N，V个不同的tokens，一般T>N)
    - Grapheme 书写系统的最小单元。英文需要空白符，中文不需要。(41%)
    - Word 英文的V可能超过10w，所以不适合中来当Grapheme(10%)
    - Morpheme 可以表示意思的最小单位。(17%)
    - Bytes 可以做到语言无关。UTF-8 包含所有语言
- 输出部分
  - 语言识别，输出word embedding
  - +翻译
  - +意图识别
- 输入部分 acoustic feature
  - 窗口25ms，移动步长10ms 
    -  39-dim MFCC (18%)
    -  80-dim filter bank output (75%)
  - DFT把waveform转换成spectrogram，取log，然后DCT处理得到MFCC 
- 语料
  - TIMIT 4hr
  - WSJ 80hr
  - Swiboard 300hr
  - Librispeech 960hr
  - Fisher 2000hr
  - Google Voice Search论文用了1w小时，实际比论文的高一个数据量级
- 两种视角 Seq-to-Seq  HMM 
- 模型
  - LAS(40%)
  - CTC(24%)
  - LAS+CTC(11%)
  - RNN-T(10%)
  - HMM-hybrid(15%)


## 2.1. LAS listen,attend,spell

- Listen 
  - RNN、1-D CNN
  - Down Sampling
    - Pyramid RNN 
    - Pooling over time 
  - TDNN Time-delay DNN 
  - Truncated Self-attention
    - match的方法
      - dot-product attention 点乘
      - additive attention 加和+tanh
- Beam Search  每次保留topN的最好路径，然后生成子节点，重新进行排序，选择topN
- Teacher Forcing 训练过程中，预测下一个字母的时候，前面的文本用正确答案
- Location-aware attention

LAS不是一开始就超越传统模型的，是语料逐渐增加，逐渐优化才超越传统模型。LAS比传统模型小，只用0.4G，但是传统模型可能要7G+。

LAS的限制，不能一边听一边输出

## 2.2. CTC Connectionist Temporal Classification

## 2.3. HMM

## 2.4. Alignment

## 2.5. RNN-T

## 2.6. Language Modeling

# 3. Voice Conversion

- 应用
  - 不同的人的声音作用不同
  - Deep Fake
  - 个性化的TTS
  - 唱歌
  - 隐私保护
  - 风格转换
    - 情绪 emotion
    - Normal-to-Lombard 把声音变清除一点
    - Whisper-to-Normal 悄悄话转成正常
    - Singers 技巧
  - 提升发音
    - 把声音变清晰
    - 口音转换
  - 数据加强
    - 增加训练数据
    - 把数据增加噪声
- 问题
  - 可以不等长，但是目前主要还是等长序列转换
  - Vocoder
    - Rule-based： Griffin-Lim algorithm
    - Deep Learning: WaveNet
  - 训练语料
    - 对称语料
    - 不对称语料 
      - 特征区分 Feature Disentangle
        - 内容信息 Content Encoder
          - 直接用ASR 
          - 用NN转成state 
          - 用一个GAN，保证输出无法被识别出Speaker的信息 
          - instance normalization
        - 语者信息 Speaker Encoder
          - one-hot 
          - Speaker embedding
        - Decoder
          - adaptive instance normalization
      - 直接转换
        - cycle GAN
        - starGAN


# 4. Speech Separation

- Evaluation
  - Signal-to-noise ratio(SNR) 不太好
  - Scale invariant signal-to-distortion ratio(SI-SDR) 
  - PESQ 
  - STOI
- 模型
  - Mask
    - Ideal Binary Mask
    - Deep Clustering
  - PIT (permutation invariant training)