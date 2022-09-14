# ASR-Note


## 基础概念

[Phoneme的相关概念以及Triphone](https://zhuanlan.zhihu.com/p/322194937)
1. 词 words
2. 单音 syllables  实际上它本身并不是一个音，而是好几个拼在一起的，比如中文里的声母韵母。syallable其实没法直接评价，因为他是依赖于语言的，不同语言的syllable总数会相差很多
3. 音素 phonemes  语言学中一个语言最小基本单位音
4. Triphone  直接解决了前后文影响（context-dependency）的问题，并且也满足普适性，但他数据量太大


[ASR中常用的语音特征之FBank和MFCC](https://blog.csdn.net/Magical_Bubble/article/details/90295814)
- FBank特征（Filter Banks） FBank特征的提取更多的是希望符合声音信号的本质，拟合人耳接收的特性。因为神经网络对高度相关的信息不敏感,FBank特征越来越流行。
- MFCC特征（Mel-frequency Cepstral Coefficients）MFCC特征多的那一步则是受限于一些机器学习算法。很早之前MFCC特征和GMMs-HMMs方法结合是ASR的主流。
