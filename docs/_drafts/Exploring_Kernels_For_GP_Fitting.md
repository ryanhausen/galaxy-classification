---
layout: default
title: "Exploring Kernels For GP Fitting"
date: 2017-03-28
categories: synth-images, gaussian-process
---

{% include mathjax.html %}

## Exploring Different Kernel Functions for A Proper Fit

All the functions below include a additive noise term ${\Sigma_n}$ which represents the noise per datapoint along the diagonal of a matrix where are all other entries are 0. 

$$cov(x,x') = k(x, x') + \Sigma_n$$

The data used for fitting each process is the measured values($16^{th}$, median, $84^{th}$) from the confident disks in the H band.

![measured-data]({{ site.url }}/assets/imgs/2017-03-28/measured_data.png)

### Radial Basis Function

Definition:
$$k(x, x')=\exp(\frac{|x-x'|^2}{2l^2})$$

Prior:

![rbf-prior]({{ site.url }}/assets/imgs/2017-03-28/rb_prior.png)



