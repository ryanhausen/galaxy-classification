---
layout: default
title: "Validating Generated Disk Images"
date: 2017-04-24
categories:
---

{% include mathjax.html  %}


### Making Disk Sources

Start with the developed gaussian process method for making a spheroid.

![gp-sample]({{ site.baseurl  }}/assets/imgs/2017-04-24/gp-sample.png)

![sph-img]({{ site.baseurl  }}/assets/imgs/2017-04-24/sph-img.png)

To change the spheroid source into a disk source we use a transformation matrix. Our new disk source will have an axis ratio of 0.5. To find the the major and minor axis, solve:

$$a = \sqrt{r^2 * 0.5}$$

$$b = \sqrt{r^2 / 0.5} $$

where $r$ is the effective radius in pixel units. $a$ and $b$ are used to make the transformation matrix which is an affine transformation:

$$M_{origin} \cdot M_{scale} \cdot M_{center}$$

Where $M_{scale}$ is a [scaling matrix](https://en.wikipedia.org/wiki/File:2D_affine_transformation_matrix.svg) with a in the [0,0] location and b in the [1,1]  location. $M_{origin}$ and $M_{center}$ are [translation matrices](https://en.wikipedia.org/wiki/File:2D_affine_transformation_matrix.svg) that move to origin and then restore it.

Then change the axis ratio of the spheroid using the following transformation matrix:

$$
\begin{bmatrix}
1.41421356 & 0 &  -17.39696962 \\
0 & 0.70710678 &  12.30151519 \\
0 & 0 & 1
\end{bmatrix}
$$

After applying the affine transformation the disk object looks like this:


![disk-img]({{ site.baseurl  }}/assets/imgs/2017-04-24/disk-img.png)

To confirm that the new image has the properties intended, the axis ratio can be measured directly from the image. We intended an axis ratio of 0.5 anf the measured axis ratio is 0.4956688764680927.

### Update the Axis Ratio and Effective Radius Binning 

I changed the numer of bins to 100 and also set the bounds for the binning to be shared between disks and spheroids for comparison. The new histograms look like this:

![disk-bins]({{ site.baseurl  }}/assets/imgs/2017-04-24/disk-bins.png)
![sph-bins]({{ site.baseurl  }}/assets/imgs/2017-04-24/sph-bins.png)