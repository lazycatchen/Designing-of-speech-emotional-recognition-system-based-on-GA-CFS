# Designing-of-speech-emotional-recognition-system-based-on-GA-CFS
基于LabVIEW的声音情感识别系统（BP内核）
CFS改进的GA算法选择最优特征集文件夹：CFS公式作为GA适应度函数，改进GA。ga_speech_opt.m为主程序，若需要降维其他特征则用其他数据代替ga_speech_opt.m中的data即可。Fitness.m文件为适应度函数子程序。
LabVIEW调用的matlab程序文件夹：BP_speech_prediction.m为BP算法主程序，剩余三个文件分别为四五六分类的情感训练子文件。
test_data文件夹：四组不同情绪音乐示例。
emotion_recognize.vi：为情感识别系统的主要程序框图，其余为调用的子vi，直接运行即可。存在的问题为四个不同的模块每次单独运行都需要先结束上一个模块运行，主要原因是不同功能的切换用的是“选项卡”表示，布尔控件有冲突，布尔控件表示不了太复杂的情况，解决办法：将录音、特征提取、训练、识别分别用四个不同的循环结构或者时间结构表示，即并行运算（并未编程，理论上可行）。
