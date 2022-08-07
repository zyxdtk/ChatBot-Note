# 1. ChatBot-Note
对话机器人学习笔记

- [1. ChatBot-Note](#1-chatbot-note)
- [2. 文本对话](#2-文本对话)
  - [2.1. 数据集&指标](#21-数据集指标)
  - [2.2. 对话模型](#22-对话模型)
  - [2.3. 开源项目](#23-开源项目)
  - [2.4. NLP基础](#24-nlp基础)
- [3. 语音对话](#3-语音对话)
  - [ASR 语言识别](#asr-语言识别)
  - [TTS 语言合成](#tts-语言合成)
- [视频对话](#视频对话)
  - [人物识别](#人物识别)
  - [3D人物建模](#3d人物建模)
- [4. 产品](#4-产品)
  - [4.1. 业界产品](#41-业界产品)
  - [4.2. 对话界面](#42-对话界面)
  - [4.3. QQ机器人](#43-qq机器人)
- [5. 参考资料](#5-参考资料)


# 2. 文本对话

## 2.1. 数据集&指标

- [BLEU](https://blog.csdn.net/qq_36652619/article/details/87544918)

语料库
- [腾讯AI实验室的语料库](https://ai.tencent.com/ailab/nlp/zh/download.html)
- [悟道2.0语料](https://resource.wudaoai.cn/home?ind&name=WuDaoCorpora%202.0&id=1394901288847716352) WuDaoCorpora2.0由全球最大的纯文本数据集(总量3TB、开源200GB)、全球最大的多模态数据集(93TB，6.5亿图文对，开源500w对)和全球最大的中文对话数据集(181G,对话轮数1.4B，不开源)三部分构成。
- [CLUECorpus2020](https://github.com/CLUEbenchmark/CLUECorpus2020)  通过对Common Crawl的中文部分进行语料清洗，最终得到100GB的高质量中文预训练语料
- [百度DuReader-Retrieval](https://github.com/baidu/DuReader/tree/master/DuReader-Retrieval) 百度利用真实场景下的用户搜索日志，建立了首个大规模高质量中文段落检索数据集

## 2.2. 对话模型

- 预训练大模型
  - [2018]. [Personalizing Dialogue Agents: I have a dog, do you have pets too?](https://arxiv.org/abs/1801.07243) 
  - [GPT]
  - [GPT2]()
    - [5.1k] [GPT2]
    - [izeyao/GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese) 中文版GPT2
  - [2020]. [GPT3](https://arxiv.org/pdf/2005.14165.pdf) [【汉】](https://zhuanlan.zhihu.com/p/200978538)
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
- prompt干预
  - [2021]. [Controllable Generation from Pre-trained Language Models via Inverse Prompting](https://arxiv.org/pdf/2103.10685.pdf) 用反向prompt来校验prompt的输出
  - [2021]. [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/abs/2112.00861) 
  - [2022]. [Prompt-Driven Neural Machine Translation](https://aclanthology.org/2022.findings-acl.203.pdf) 尝试了单独给prompt一个encode，把prompt加载输入上等方式。
  - [2022]. [Chain of Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)


## 2.3. 开源项目

- 开源框架(2022-07-05更新star数)
  - [9k] [facebookresearch/ParlAI](https://github.com/facebookresearch/ParlAI) 提供开源对话数据集上训练和评估AI，有一些AI模型的实现
    - https://parl.ai/
  - [huggingface/transformers](https://github.com/huggingface/transformers) 包含了很多ml的模型  
- 对话模型  
  - [12.4k] [ChatterBot](https://github.com/gunthercox/ChatterBot) 最近没怎么更新了
    - [chamkank/flask-chatterbot](https://github.com/chamkank/flask-chatterbot) 基于ChatterBot做的一个简单的web对话机器人
  - [4.8k] [PaddlePaddle/PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP) 百度开源的NLP项目
    - [unified_transformer](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/dialogue/unified_transformer)  
    - [plato-2](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/dialogue/plato-2)   plato-2模型，开放域聊天机器人
  - [4.4k] [NVIDIA/NeMo](https://github.com/NVIDIA/NeMo) 英伟达的对话机器人工具箱
  - [3.4k] [wzpan/wukong-robot](https://github.com/wzpan/wukong-robot) 中文语音对话机器人/智能音箱项目
  - [2.7k] [zhaoyingjun/chatbot](https://github.com/zhaoyingjun/chatbot) 这是一个可以使用自己语料进行训练的中文聊天机器人项目，包含tensorflow.2x版本和pytorch版本
  - [806] [tensorlayer/seq2seq-chatbot](https://github.com/tensorlayer/seq2seq-chatbot) 200行的对话机器人


## 2.4. NLP基础
- [Evolved Transformer](https://arxiv.org/abs/1901.11117)[【汉】](https://nopsled.blog.csdn.net/article/details/108713234)[【code】](https://github.com/Shikhar-S/EvolvedTransformer)
- 卷积模型
  - WaveNet
  - Gated
  - NASNet [【汉】](https://zhuanlan.zhihu.com/p/52616166)

# 3. 语音对话
## ASR 语言识别

## TTS 语言合成

# 视频对话

## 人物识别

## 3D人物建模

# 4. 产品 
  
## 4.1. 业界产品

- 大公司产品
  - 小冰
    - [The Design and Implementation of XiaoIce,an Empathetic Social Chatbot](https://arxiv.org/pdf/1812.08989.pdf) 
  - [replika](https://replika.ai/)
    - [Building a compassionate AI friend](https://blog.replika.com/posts/building-a-compassionate-ai-friend) 
  - [aidungeon](https://aidungeon.cc/) 文字冒险
    -  [GPT-2的大规模部署：AI Dungeon 2 如何支撑百万级用户](https://blog.csdn.net/weixin_42137700/article/details/104359367)
    -  [【游戏推荐】 AIDungeon 2](https://zhuanlan.zhihu.com/p/104476177)
- 公开的api
  - [小寰API](http://81.70.100.130/) 
  - [青云客](http://api.qingyunke.com/)
  - [思知](https://www.ownthink.com/robot.html)


## 4.2. 对话界面


## 4.3. QQ机器人

- [16.2k] [hubotio/hubot](https://github.com/hubotio/hubot) 也是一个机器人开发工具，但是好久不更新了
- [10.8k] [howdyai/botkit](https://github.com/howdyai/botkit) 聊天机器人开发工具
- [2.7k] [errbotio/errbot](https://github.com/errbotio/errbot/)
- [nonebot/awesome-nonebot](https://github.com/nonebot/awesome-nonebot)
- [botuniverse/onebot-11](https://github.com/botuniverse/onebot-11) OneBot 标准是从原 CKYU 平台的 CQHTTP 插件接口修改而来的通用聊天机器人应用接口标准。
- [2.1k] [nonebot/nonebot2](https://github.com/nonebot/nonebot2) python版本的qq聊天机器人，支持插件
- [650] [FloatTech/ZeroBot-Plugin](https://github.com/FloatTech/ZeroBot-Plugin) go版本的qq机器人插件，要跟下面的go-cqhttp配合
  - [5.3k] [Mrs4s/go-cqhttp](https://github.com/Mrs4s/go-cqhttp) cqhttp的golang实现，轻量、原生跨平台

# 5. 参考资料
- [737] [qhduan/ConversationalRobotDesign](https://github.com/qhduan/ConversationalRobotDesign) 对话机器人（聊天机器人）设计思考
- [285] [aceimnorstuvwxz/awesome-chatbot-list](https://github.com/aceimnorstuvwxz/awesome-chatbot-list) 
- [5.4k] [lcdevelop/ChatBotCourse](https://github.com/lcdevelop/ChatBotCourse) 自己动手做聊天机器人教程