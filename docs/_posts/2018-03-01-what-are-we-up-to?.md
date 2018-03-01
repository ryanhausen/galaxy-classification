---
layout: default
title: "What are we up to?"
date: 2018-03-01
categories:
bands:
  - "h"
  - "j"
  - "v"
  - "z"
---

{% include mathjax.html  %}

### Semantic Segmentation For Morphological Classification

Rather than get a single classification for an entire image we would like a
classification per pixel.

![semantic]({{site.baseurl}}/assets/imgs/2017-03-01/semantic.png)

We have 6 classes:

Spheroid, Disk, Irregular, Point Source, Unknown, Background

The model:

![unet]({{site.baseurl}}/assets/imgs/2017-03-01/unet1.png)

![tf-unet]({{site.baseurl}}/assets/imgs/2017-03-01/tf-unet.png)

![tf-segment]({{site.baseurl}}/assets/imgs/2017-03-01/tf-segment.png)