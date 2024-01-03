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
