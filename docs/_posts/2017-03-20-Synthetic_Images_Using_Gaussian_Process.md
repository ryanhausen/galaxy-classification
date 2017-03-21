---
layout: default
title: "Synthetic Images Using Gaussian Process"
date: 2017-03-20
categories: synth-images, gaussian-process
---

{% include mathjax.html %}

### Adding Noise The Analytic Graph



### H Band Spheroid Median plotted with Disk Median

![median-overplot]({{ site.url }}/assets/imgs/2017-03-20/median-overplot.png)

### Ratio of Bands

![disk-band-ratio]({{ site.url }}/assets/imgs/2017-03-20/disk-band-ratio.png)

![spheroid-band-ratio]({{ site.url }}/assets/imgs/2017-03-20/spheroid-band-ratio.png)

## Fitting with a Gaussian Process

We'll fit using the data from the Disk class in the H-Band which looks like this after we moved the data to log scale:

![disk-h-band]({{ site.url }}/assets/imgs/2017-03-20/disk-h-band.png)


The data is heteroskedastic, we'll remedy that by taking the mean of the difference between the 84th percentile and the median.

![disk-h-band-homoskedastic]({{ site.url }}/assets/imgs/2017-03-20/disk-h-band-homoskedastic.png)


