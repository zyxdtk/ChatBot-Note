# 1. ChatBot-Note
对话机器人学习笔记

- [1. ChatBot-Note](#1-chatbot-note)
- [2. 数据集&指标](#2-数据集指标)
- [3. 大模型](#3-大模型)
- [4. 相关公司&开源项目&API](#4-相关公司开源项目api)
- [5. 参考资料](#5-参考资料)

# 2. 数据集&指标

- [BLEU](https://blog.csdn.net/qq_36652619/article/details/87544918)

语料库
- [腾讯AI实验室的语料库](https://ai.tencent.com/ailab/nlp/zh/download.html)

# 3. 大模型 

- [GPT2]()
  - [5.1k] [Morizeyao/GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese) 中文版GPT2
- [GPT3]()
  - [11.4k] [openai/gpt-3](https://github.com/openai/gpt-3) 
  - [3.6k] [elyase/awesome-gpt3](https://github.com/elyase/awesome-gpt3)
- [lamda: language models for dialog applications](https://arxiv.org/pdf/2201.08239.pdf)[【汉】](https://zhuanlan.zhihu.com/p/462022601)
- [Towards a Human-like Open-Domain Chatbot](https://arxiv.org/abs/2001.09977)
- [EVA2.0：大规模中文开放域对话预训练模型](https://blog.csdn.net/weixin_42001089/article/details/123595667)
- [2021] [Blender Bot 2.0](https://ai.facebook.com/blog/blender-bot-2-an-open-source-chatbot-that-builds-long-term-memory-and-searches-the-internet/)

一些模型结构：
- [Evolved Transformer](https://arxiv.org/abs/1901.11117)[【汉】](https://nopsled.blog.csdn.net/article/details/108713234)[【code】](https://github.com/Shikhar-S/EvolvedTransformer)
- 卷积模型
  - WaveNet
  - Gated
  - NASNet [【汉】](https://zhuanlan.zhihu.com/p/52616166)


# 4. 相关公司&开源项目&API 

- 公开的api
  - [小寰API](http://81.70.100.130/) 
  - [青云客](http://api.qingyunke.com/)
  - [思知](https://www.ownthink.com/robot.html)
- 开源项目(2022-07-05更新star数)
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
  - qq机器人
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