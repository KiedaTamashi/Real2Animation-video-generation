Real2Animation-video-generation
===========================

[Video Demo (Poseted on Bilibili)](https://www.bilibili.com/video/BV1aT4y1u7Ep)

[Detailed Report(In paper version)](./Report_of_what_I_do.pdf)

## Introduction

For amateur anime lovers who want to produce their own anime shorts, it takes great effort to master extra skills and techniques even if they have designed their own anime character in a painting. This project proposes an approach to generate an anime video sequence automatically given a real human video sequence and an image of anime character.

## Example of the work. 

![alt](https://github.com/XiaoSanGit/Real2Animation-video-generation/blob/master/figs/exp/total.png)

Left `Reference` is the given anime character image. The bottom of right, which is marked as `Real Human`, is the given real human video sequence. The upper of right, which is marked as `Anime`, is the generated anime video sequence. The real human video performs the motion that we wish to transfer to target anime character image. We namely making target anime character do what the real human do in the video. We separately train different phases of transferring and combined them together as a video-to-video translation in testing, enabling practitioners to use it as reference and amateurs to make anime on their own easily.

More Examples:

![alt](https://github.com/XiaoSanGit/Real2Animation-video-generation/blob/master/figs/exp/stage1a.png)
![alt](https://github.com/XiaoSanGit/Real2Animation-video-generation/blob/master/figs/exp/stage2b.png)

## Overall Pipeline. 

![alt](https://github.com/XiaoSanGit/Real2Animation-video-generation/blob/master/figs/overall_stage.png)

**Stage one** Address the problem: We don’t have anime video performing the same motion `x` as the real videos we collected and it is impossible to manually pair videos from the network.  
**Stage two** We use the intermediate feature to bridge the input and output. We apply our feature extraction and normalization method in Stage Two. 
**Stage three** With the help of the end-to-end generative network, we combine the features prepared to synthesis the final result.
