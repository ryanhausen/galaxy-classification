---
layout: default
title: "Checking V Band Metrics"
date: 2017-07-04
categories:
---

{% include mathjax.html  %}

### Central Values for V-Band

![v-center]({{ site.url }}/assets/imgs/2017-07-04/v-center.png)
![v-center-ie]({{ site.url }}/assets/imgs/2017-07-04/v-center-ie.png)

### Double Check $R_e$ and $I_e$ Distributions

It turns out I did this in the previous post when I explored using a two dimensional gaussian.

Check [here.](https://ryanhausen.github.io/galaxy-classification/2017/05/23/using-a-gaussian-to-describe-effective-radius-and-axis-ratio.html)

### Updated Signal to Noise Ratios

I changed the calculation in the signal to noise ratio to:

Signal = The sum of the pixel values within the the effective radius
Noise = The RMS of the noise pixels outside 3 * the effective radius

![disk-h]({{ site.url }}/assets/imgs/2017-07-04/disk-h-hist.png)
![disk-j]({{ site.url }}/assets/imgs/2017-07-04/disk-j-hist.png)
![disk-v]({{ site.url }}/assets/imgs/2017-07-04/disk-v-hist.png)
![disk-z]({{ site.url }}/assets/imgs/2017-07-04/disk-z-hist.png)

![spheroid-h]({{ site.url }}/assets/imgs/2017-07-04/spheroid-h-hist.png)
![spheroid-j]({{ site.url }}/assets/imgs/2017-07-04/spheroid-j-hist.png)
![spheroid-v]({{ site.url }}/assets/imgs/2017-07-04/spheroid-v-hist.png)
![spheroid-z]({{ site.url }}/assets/imgs/2017-07-04/spheroid-z-hist.png)