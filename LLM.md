# LLM

## 模型训练
### PreTrain

- [2018]. [Personalizing Dialogue Agents: I have a dog, do you have pets too?](https://arxiv.org/abs/1801.07243) 
- [GPT]
- [GPT2]()
- [15.7k] [GPT2](https://github.com/openai/gpt-2)
- [izeyao/GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese) 中文版GPT2
- [2020]. [GPT3:Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf) [【汉】](https://zhuanlan.zhihu.com/p/200978538)
- [11.4k] [openai/gpt-3](https://github.com/openai/gpt-3) 
- [3.6k] [elyase/awesome-gpt3](https://github.com/elyase/awesome-gpt3)
- [GPT1、GPT2、GPT3原理](https://blog.csdn.net/qq_41357569/article/details/121731981)
- [GPT-3的50种玩法告诉你，它很酷，但是没有通过图灵测试](https://zhuanlan.zhihu.com/p/252851574)
- [lamda: language models for dialog applications](https://arxiv.org/pdf/2201.08239.pdf)[【汉】](https://zhuanlan.zhihu.com/p/462022601)
- [Towards a Human-like Open-Domain Chatbot](https://arxiv.org/abs/2001.09977)
- [EVA2.0：大规模中文开放域对话预训练模型](https://blog.csdn.net/weixin_42001089/article/details/123595667)
- [2021]. [Blender Bot 2.0](https://ai.facebook.com/blog/blender-bot-2-an-open-source-chatbot-that-builds-long-term-memory-and-searches-the-internet/)
- CPM 清华做的中文预训练模型
- [yangjianxin1/CPM](https://github.com/yangjianxin1/CPM) 基于CPM的中文文本生成,开源代码库
- [CPM-2: Large-scale Cost-effective Pre-trained Language Models](https://arxiv.org/pdf/2106.10715.pdf) [【汉】](https://blog.csdn.net/BAAIBeijing/article/details/118125026)
- 任务
  - [2022]. [Language Models that Seek for Knowledge: Modular Search & Generation for Dialogue and Prompt Completion](https://arxiv.org/abs/2203.13224) 引入搜索到的知识，包含三个模块依次得到：搜搜query，知识序列，最终的回应。在对话任务上超过了blender bot2，评价指标是知识性、事实正确、吸引力。在语言建模上，相比GPT2和GPT3，幻想更少，更有话题性。
      - [parlAI seeker](https://github.com/MiniMax-AI/ParlAI/tree/main/projects/seeker)


- [GPT-4 Technical Report](https://arxiv.org/pdf/2303.08774.pdf)
- [BloombergGPT: A Large Language Model for Finance](https://arxiv.org/pdf/2303.17564.pdf)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [TRAIN SHORT, TEST LONG: ATTENTION WITH LINEAR BIASES ENABLES INPUT LENGTH EXTRAPOLATION](https://arxiv.org/pdf/2108.12409.pdf)
- [OPT]()
  - [Open Pretrained Transformers - Susan Zhang | Stanford MLSys #77](https://www.youtube.com/watch?v=p9IxoSkvZ-M) 


怎么理解LLM
- [为什么说 GPT 是无损压缩](https://bigeagle.me/2023/03/llm-is-compression/)
- [Compression for AGI - Jack Rae | Stanford MLSys #76](https://www.youtube.com/watch?v=dO4TPJkeaaU)
- [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446)
- [Data Selection for Language Models via Importance Resampling](https://arxiv.org/abs/2302.03169)
- [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/pdf/2112.06905.pdf)
模型结构和调参
- [让研究人员绞尽脑汁的Transformer位置编码](https://kexue.fm/archives/8130)
多模态
- [MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models](https://github.com/Vision-CAIR/MiniGPT-4)   [demo](https://minigpt-4.github.io/)
代码
- [2021] [Evaluating Large Language Models Trained on Code](https://arxiv.org/pdf/2107.03374.pdf)
  - [openai/human-eval](https://github.com/openai/human-eval)
- [CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X](https://arxiv.org/abs/2303.17568)


### SFT

- [CHINESE OPEN INSTRUCTION GENERALIST: A PRELIMINARY RELEASE](https://arxiv.org/pdf/2304.07987.pdf)

### RLHF

- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)
- [Reinforcement Learning for Language Models](https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81)
- [Deep Reinforcement Learning from Human Preferences](https://proceedings.neurips.cc/paper/2017/file/d5e2c0adad503c91f91df240d0cd4e49-Paper.pdf)
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/pdf/2212.08073.pdf)
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/pdf/2210.10760.pdf)
- [2021]. [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/abs/2112.00861) PMP，偏好模型预训练，用互联网数据如Reditt等的语料做预训练。

### Prompt

- [2021]. [Controllable Generation from Pre-trained Language Models via Inverse Prompting](https://arxiv.org/pdf/2103.10685.pdf) 用反向prompt来校验prompt的输出
- [2021]. [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/abs/2112.00861) 
- [2022]. [Prompt-Driven Neural Machine Translation](https://aclanthology.org/2022.findings-acl.203.pdf) 尝试了单独给prompt一个encode，把prompt加载输入上等方式。
- [2022]. [Chain of Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- [Extracting Training Data from Large Language Models](https://www.usenix.org/system/files/sec21-carlini-extracting.pdf)
- [Imitation Attacks and Defenses for Black-box Machine Translation Systems](https://arxiv.org/pdf/2004.15015.pdf)
- [OpenAssistant Conversations -- Democratizing Large Language Model Alignment](https://arxiv.org/abs/2304.07327)


## 标注

- [AnnoLLM: Making Large Language Models to Be Better Crowdsourced Annotators](https://arxiv.org/abs/2303.16854)

### 基础
采样策略
- Beam Search
- [Nucleus Sampling: The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
  - [知乎：Nucleus Sampling与不同解码策略简介](https://zhuanlan.zhihu.com/p/442557114)

## 安全

- [Is Power-Seeking AI an Existential Risk?](https://arxiv.org/pdf/2206.13353.pdf)

## 模型应用
- [CRSLab：可能是最适合你的对话推荐系统开源库
](https://picture.iczhiku.com/weixin/message1610089596644.html)

### 新闻

- 2023-04-10 商汤发布商量大模型
- 2023-04-11 阿里发布通义大模型，阿里CEO张勇称阿里所有产品将用大模型改造。

