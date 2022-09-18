# Avatar-Note
声音到表情

## 基础概念

- Phoneme 音素，人类语言中能够区别意义的最小声音单位
- Viseme 视素，视觉方式呈现出的音素，描绘出发音时的嘴部姿态

表情分类
- https://en.wikipedia.org/wiki/Emotion_classification
- https://gas.graviti.com/dataset/graviti/RadboudFaces
- 30种表情 https://www.youtube.com/watch?v=y2YUMPJATmg
- [人脸姿态&表情识别的10个数据集](https://zhuanlan.zhihu.com/p/380050143)


## 形象建模

VAM信息：
- [Virt-A-Mate是一款面向成年人的沙盒VR性爱游戏](http://www.vamgame.com/882.html)
- [Virt a Mate资源下载站](https://vam3.com/)
- [如何评价《virt a mate》游戏（vam）？](https://www.zhihu.com/question/474761394)
  - Virt A Mate 是一款由Meshed VR社区大佬开发的VR游戏和模拟器，原生支持VR设备，也可以普通模式运行。根据账号等级不同，在游戏内分为Play和Edit模式。顾名思义，Play就是欣赏或者游玩其他人制作好的场景，而Edit模式可以自由编辑和重新设计。游戏提供了男女基础人物素体，让玩家自己捏人改变身体特征、皮肤纹理、服饰、头发，并可以通过关节（Joint）设置人物的姿势和动作。甚至可以修改牙齿、舌头、眼部（眉毛、睫毛、眼白、眼仁）的纹理和颜色。有意思的是，Build默认都是黄金比例的模特身材，玩家想要捏成普通人的样子反而很有难度。关于身体各部位的调整，系统自带有几百个参数（morph）可以调整身体器官的形状和尺寸，自己增加了plugin的话可以扩展上万个。比如双手的五个手指可以分别设置弯曲角度，以及脸部各种表情。
  - vr黄油目前当然的no.1，没有之一。吊打i社的什么playhome，honey select，honey select2，ai girls。唯一能匹敌的大概是老滚试验室mod吧。
- [Mesh是微软为开发者提供用来创建多人XR应用程序的开发工具](https://vr.ofweek.com/news/2021-03/ART-815003-8110-30488583.html)
- [CINEMA 4D篇—简介/安装/界面/基本功能](https://huke88.com/article/3227.html)


## audio2face

- [omniverse-audio2face](https://www.nvidia.com/en-us/omniverse/apps/audio2face/)
  - [2017]. [Audio-Driven Facial Animation by Joint End-to-End Learning of Pose and Emotion](https://research.nvidia.com/sites/default/files/publications/karras2017siggraph-paper_0.pdf) [【本地PDF】](static/pdf/karras2017siggraph-paper_0.pdf) [【汉】](https://zhuanlan.zhihu.com/p/463827738) 用3-5分钟数据
  - [B站-根据语音自动生成表情动画-Omniverse Audio2Face](https://www.bilibili.com/video/BV1sq4y127ta?spm_id_from=333.337.search-card.all.click&vd_source=72bd417d3c61f48a1851179442d7083c)
- [2017]. [A Deep Learning Approach for Generalized Speech Animation](https://home.ttic.edu/~taehwan/taylor_etal_siggraph2017.pdf)  
- [2019]. [Capture, Learning, and Synthesis of 3D Speaking Styles](https://arxiv.org/pdf/1905.03079.pdf) [【演示视频】](https://www.youtube.com/watch?v=XceCxf_GyW4) 
  - https://voca.is.tue.mpg.de/ 
- [2019]. [First Order Motion Model for Image Animation](https://aliaksandrsiarohin.github.io/first-order-model-website/)
- [2021]. [LipSync3D: Data-Efficient Learning of Personalized 3D Talking Faces from Video using Pose and Lighting Normalization](https://arxiv.org/pdf/2106.04185v1.pdf)
- [2021]. [Joint Audio-Text Model for Expressive Speech-Driven 3D Facial Animation](https://arxiv.org/pdf/2112.02214.pdf)
- [2021]. [Audio2Head: Audio-driven One-shot Talking-head Generation with Natural Head Motion](https://arxiv.org/pdf/2107.09293.pdf)
- [2021]. [Write-a-speaker: Text-based Emotional and Rhythmic Talking-head Generation](https://ojs.aaai.org/index.php/AAAI/article/view/16286)
- [2021]. [RoI Tanh-polar Transformer Network for Face Parsing in the Wild](https://paperswithcode.com/paper/roi-tanh-polar-transformer-network-for-face)
- [2022]. [MeshTalk: 3D Face Animation from Speech using Cross-Modality Disentanglement](https://arxiv.org/pdf/2104.08223.pdf) [【演示视频】](https://www.bilibili.com/video/BV1f5411c7jU/) 
- [2022]. [FaceFormer: Speech-Driven 3D Facial Animation with Transformers](https://arxiv.org/pdf/2112.05329.pdf)
  - https://evelynfan.github.io/audio2face/
- [2022]. [One-Shot Talking Face Generation from Single-Speaker Audio-Visual Correlation Learning](https://ojs.aaai.org/index.php/AAAI/article/view/20154)
- [2022]. [MegaPortraits: One-shot Megapixel Neural Head Avatars](https://samsunglabs.github.io/MegaPortraits/)
- [2022]. [General Facial Representation Learning in a Visual-Linguistic Manner](https://paperswithcode.com/paper/general-facial-representation-learning-in-a)
- [2022]. [Decoupled Multi-task Learning with Cyclical Self-Regulation for Face Parsing
](https://paperswithcode.com/paper/decoupled-multi-task-learning-with-cyclical)

综述
- [现有技术可以让游戏⾓⾊的口型和表情根据对话产⽣变化吗？](https://www.zhihu.com/question/537482503) 
- https://github.com/JosephPai/Awesome-Talking-Face
- https://github.com/harlanhong/awesome-talking-head-generation
- [2020]. [What comprises a good talking-head video generation?: A Survey and Benchmark](https://arxiv.org/pdf/2005.03201v1.pdf)
- [2022]. [Deep Learning for Visual Speech Analysis: A Survey](https://arxiv.org/pdf/2205.10839.pdf)
