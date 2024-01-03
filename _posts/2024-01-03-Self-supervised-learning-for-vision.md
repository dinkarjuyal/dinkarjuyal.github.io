---
title: "SSL for vision"
date: 2023-01-03
---

*This post is a collection of observations around self-supervised learning for vision-based datasets.
This is by no means a complete survey of SSL techniques, and assumes familiarity with Contrastive Learning, Masked Image Modeling and Masked AutoEncoder frameworks. 
For readers intending to get up to speed on these topics, there are excellent resources available online ([1](https://arxiv.org/abs/2304.12210), [2](https://lilianweng.github.io/posts/2021-05-31-contrastive/)).* 

### Setting 
The datasets in the real world are rarely as well-understood and worked upon as academic datasets like ImageNet-1k. They may have long-tailed distributions and/or high degree of label imbalance for a subset of classes. The labels may be noisy and the image metadata might be missing/unreliable. Further, label collection/verification might be time-consuming and expensive. There might be systematic confounders present as a result of the data collection. We might be expected to deploy the model in situations where the data distribution might be unknown to us. At the same time, if our model performance on the metrics that we care about degrades, we should be able to correct for that with minimum effort and disruption to our workflows.
In this context, we want to harness SSL pretraining (or [pre-pretraining](https://openaccess.thecvf.com/content/ICCV2023/papers/Singh_The_Effectiveness_of_MAE_Pre-Pretraining_for_Billion-Scale_Pretraining_ICCV_2023_paper.pdf)) to alleviate the need of labels, generate synthetic data for rare classes and learn rich representations that can be used for a variety of downstream tasks.

![label noise](/assets/self-supervised-learning-for-vision/vision-dataset-label-errors.png)
<p align="center">
Vision datasets can have pervasive label issues (<a href="https://arxiv.org/abs/2103.14749">Source</a>)
</p>

### Contrastive Learning + Masked Image Modeling : Best of both worlds?
Contrastive learning (CL) involves bringing representations of similar samples closer and pushing apart representations of dissimilar samples. 
Initial methods for CL used a CNN encoder.
The need for positive and negative pairs within the same batch meant that these methods relied on bigger batches and correct assumptions about what constitutes positive and negative examples.  Modifications that did not require negative pairs while avoiding the trivial collapsed features as a solution ([SimSiam](https://arxiv.org/abs/2011.10566), [BYOL](https://arxiv.org/abs/2006.07733), [VICReg](https://arxiv.org/abs/2210.01571)), and deal with noisy negative examples ([RINCE loss](https://arxiv.org/pdf/2201.04309.pdf)) were suggested. With the advent of ViT, new approaches that integrate ViT into the SSL framework were designed. Works such as [Park et al](https://arxiv.org/abs/2202.06709) cover the difference between the properties of multi-head self-attention (MHSA) used in ViT and CNNs - MHSA are low-pass filters and shape-biased, while convolutions are high-pass filters and texture-biased. There is also literature that suggests that the [patchify operation](https://arxiv.org/abs/2201.09792) is the key behind ViT’s performance. Interestingly, there is work in both directions - [replacing patchify with convolution](https://arxiv.org/abs/2106.14881) to improve training stability and performance and using [patchify stem in CNN](https://arxiv.org/abs/2201.03545). While a natural next step is to [swap out CNN with ViT](https://arxiv.org/pdf/2104.02057.pdf) in the CL training pipeline, other works like [DINO](https://arxiv.org/abs/2104.14294) employ self-distillation to achieve impressive zero-shot results on segmentation & detection. [Multi-view hypothesis](https://arxiv.org/abs/2012.09816) explains the adaptability of these features by stating that self-distillation performs implicit ensembling and promotes learning of multiple concepts. Other popular methods continue using the distillation framework -  [iBOT](https://arxiv.org/abs/2111.07832) employs masked image modeling (MIM) to learn patch level features, along with applying distillation on the CLS tokens; and [DINOv2](https://arxiv.org/abs/2304.07193) combines this loss with the DINO loss. Finally, [Park et al](https://arxiv.org/abs/2305.00729) compares the pros and cons of CL and MIM. The authors also suggest a linear combination of these 2 losses as outperforming the individual components.
   
![CL vs MIM](/assets/self-supervised-learning-for-vision/CL-MIM-comparison.png)
<p align="center">
Comparison between Contrastive Learning and Masked Image Modeling (<a href="https://github.com/naver-ai/cl-vs-mim">Source</a>)
</p>

### Link between temperature and number of classes- 
In CL, the temperature term serves to sharpen the output values, with a temperature -> 0 corresponding to one-hot representations. The gradient term of the loss shows that lower values of temperature lead to higher penalty values for hard-negative examples. 

![CL gradients](/assets/self-supervised-learning-for-vision/cl_loss_weights.png)
<p align="center">
Differnt contrastive losses and the impact of temperature on loss gradients (<a href="https://arxiv.org/abs/2002.05709">Source</a>)
</p>

Temperature is a crucial parameter, with CL based methods demonstrating a dramatic sensitivity to its values <optional, DINO temperature ablation and comment about student and teacher differences>.  
