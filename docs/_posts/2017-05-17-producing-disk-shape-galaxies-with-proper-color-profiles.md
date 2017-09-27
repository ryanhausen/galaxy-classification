---
layout: default
title: "Producing Disk Shape Galaxies With Proper Color Profiles"
date: 2017-05-17
categories:
---

{% include mathjax.html  %}

### Creating an Image With a Full Color Profile

To make an image with the full color profile we use the following steps:

The images here used an effective radius of 0.30 arcseconds (0.06 arcseconds / pixel)

- Train a Gaussian Process on the surface brightness profiles of an images normalized to $I_e$ and $R_e$.
- Draw a sample from the trained process. Using the sample, generate an image.



![gp-sbp]({{ site.baseurl }}/assets/imgs/2017-05-17/sbp-gp.png)

![sph-h]({{ site.baseurl }}/assets/imgs/2017-05-17/h-sph.png)

- Train a Gaussian Process on the ratio of the surface brightness profiles of the J, V, and Z bands to H band. Normalized to H band's $R_e$

![gp-color]({{ site.baseurl }}/assets/imgs/2017-05-17/color-gp.png)

- Generate a realiztion of those ratios in an image.
- Apply those ratios to the H band sample to produce that sample in each of the other bands.

![sph-j]({{ site.baseurl }}/assets/imgs/2017-05-17/j-sph.png)
![sph-v]({{ site.baseurl }}/assets/imgs/2017-05-17/v-sph.png)
![sph-z]({{ site.baseurl }}/assets/imgs/2017-05-17/z-sph.png)

### To Make a Disk Object
- Start with a disk surface brightness profile and perform the same steps.
- Apply a transformation matrix to the image to transform from a spheroid in shape to a disk in shape. 

These image were generated with axis ratio of 0.5 it maintains a $R_e$ of ~0.30

![disk-j]({{ site.baseurl }}/assets/imgs/2017-05-17/j-disk.png)
![disk-j]({{ site.baseurl }}/assets/imgs/2017-05-17/j-disk.png)
![disk-v]({{ site.baseurl }}/assets/imgs/2017-05-17/v-disk.png)
![disk-z]({{ site.baseurl }}/assets/imgs/2017-05-17/z-disk.png)

