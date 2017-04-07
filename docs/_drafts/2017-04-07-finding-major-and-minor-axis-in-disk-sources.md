---
layout: default
title: "Finding Major and Minor Axis in Disk Sources"
date: 2017-04-07
categories:
---

{% include mathjax.html  %}

### Calculation

To find the major and minor axis for disk sources we can use the raw and central image moments within the effective radius of the source.

More on image moments can be found [here](https://en.wikipedia.org/wiki/Image_moment) and a great walkthrough for application can be found [here](https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/). 

The raw image moments  are defined as:

$$M_{ij}=\sum_x\sum_yx^iy^jI(x,y)$$

Where $I(x,y)$ is the value at pixel $x,y$.

The central image moments we use are:


To get the major and minor axis of the source we need to construct a covariance matrix for the pixels using the second order central moments.

$$\text{cov}[I(x,y)]=\left[\begin{array}{ c c }\mu'_{20} & \mu'_{11} \\ \mu'_{11} & \mu'_{02} \end{array} \right]$$

Where :
$\bar{x} = M_{10}/M_{00}$
$\bar{y} = M_{01}/M_{00}$
$\mu'_{20} = \mu_{20}/\mu_{00} = M_{20}/M_{00} - \bar{x}^2$
$\mu'_{02} = \mu_{02}/\mu_{00} = M_{02}/M_{00} - \bar{y}^2$
$\mu'_{11} = \mu_{11}/\mu_{00} = M_{11}/M_{00} - \bar{x}\bar{y}$

The eigenvectors of this matrix correspond to the major and minor axis of the source.

### Example

The source we'll use is Deep 2 10064:

![src-img]({{site.url}}/assets/imgs/2014-04-07/src-image.png)
With the segmap
![segmap]({{site.url}}/assets/imgs/2014-04-07/segmap.png)

First find the effective radius by summing values from the center of the source upto half of the total 