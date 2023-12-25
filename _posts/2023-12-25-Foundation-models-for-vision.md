---
title: "Foundation-Models-for-vision"
date: 2023-12-25
---

This image was used in Andrej Karpathy’s blog in 2012 to highlight the challenges for a vision-based foundation model (VFM). To accurately understand this picture, the model has to go beyond making disconnected predictions of certain objects (such as human faces), and make sense of the spatial relationships,perform visual reasoning along with being able to ingest multimodal data. Fast forward to 2023, and we have made tremendous progress in terms of solving these problems, in large part driven by the emergence of foundation models. Indeed, it is reported that GPT-4V is able to reason about and describe the humor in this image. Assuming that this image was not in the train set, does it mean that vision is a solved problem? What if we ask this model to segment out only those people who are not smiling, or count the squares in the floor that are covered by shadows?

In this post, we will discuss ideas and themes in terms of building foundation models for vision, along with selected methods that implement them. We will start by discussion some ideal characteristics that we might expect from a VFM:
