---
layout: default
title: "Exploring Kernels For GP Fitting"
date: 2017-03-31
categories: synth-images, gaussian-process
---

{% include mathjax.html %}

# Exploring Different Kernel Functions for A Proper Fit

All the functions below include a additive noise term ${\Sigma_n}$ which represents the noise per datapoint along the diagonal of a matrix where are all other entries are 0. 

All of the functions(excluding Squared Dot Product) also have a $\sigma_f$ parameter for scaling the kernel function 

$$cov(x,x') = \sigma_fk(x, x') + \Sigma_n$$

## Disks

The data used for fitting each process is the measured values($16^{th}$, median, $84^{th}$) from the confident disks in the H band.

![measured-data]({{ site.baseurl }}/assets/imgs/2017-03-28/measured_data.png)

### Radial Basis Function

Definition:
$$k(x, x')=\exp(\frac{|x-x'|^2}{2l^2})$$

Where $l$ is the length scale paramter.

![fitted-model]({{ site.baseurl }}/assets/imgs/2017-03-28/rbf-fitted.png)

Samples from distribution:

![fitted-model]({{ site.baseurl }}/assets/imgs/2017-03-28/rbf-sample-mean.png) ![fitted-model]({{ site.baseurl }}/assets/imgs/2017-03-28/rbf-samples.png)


Image generated from  the learned distriubtion and the samples path against the model:

![fitted-model]({{ site.baseurl }}/assets/imgs/2017-03-28/rbf-gen-image.png) ![fitted-model]({{ site.baseurl }}/assets/imgs/2017-03-28/rbf-fit-sample.png)


### Matern

Definition:

$$k(x,x') = \sigma^2\frac{2^{1-\nu}}{\Gamma(\nu)}\Big(\sqrt{2\nu}
\frac{|x-x'|}{l}\Big)^\nu K_\nu\Big(\sqrt2\nu\frac{|x-x'|}{l}\Big)$$

Where $\Gamma$ is the gamma function and $K_\nu$ is the Bessel function,  $l$ is the length scale, and $\nu$ is a non-negative parameter.

![fitted-model]({{ site.baseurl }}/assets/imgs/2017-03-28/matern-fitted.png)

Samples from distribution:

![fitted-model]({{ site.baseurl }}/assets/imgs/2017-03-28/matern-sample-mean.png) ![fitted-model]({{ site.baseurl }}/assets/imgs/2017-03-28/matern-samples.png)


Image generated from  the learned distriubtion and the samples path against the model:

![fitted-model]({{ site.baseurl }}/assets/imgs/2017-03-28/matern-gen-image.png) ![fitted-model]({{ site.baseurl }}/assets/imgs/2017-03-28/matern-fit-sample.png)

### Rational Quadratic

Definition:

$$k(x,x')=\Big(1+\frac{|x-x'|^2}{2\alpha l^2}\Big)^{-\alpha}$$



![fitted-model]({{ site.baseurl }}/assets/imgs/2017-03-28/ratq-fitted.png)

Samples from distribution:

![fitted-model]({{ site.baseurl }}/assets/imgs/2017-03-28/ratq-sample-mean.png) ![fitted-model]({{ site.baseurl }}/assets/imgs/2017-03-28/ratq-samples.png)


Image generated from  the learned distriubtion and the samples path against the model:

![fitted-model]({{ site.baseurl }}/assets/imgs/2017-03-28/ratq-gen-image.png) ![fitted-model]({{ site.baseurl }}/assets/imgs/2017-03-28/ratq-fit-sample.png)

### Squared Dot Product

Definition:

$$k(x, x')=(\sigma_0^2+ x \cdot x')^2$$


![fitted-model]({{ site.baseurl }}/assets/imgs/2017-03-28/dotprod-fitted.png)

Samples from distribution:

![fitted-model]({{ site.baseurl }}/assets/imgs/2017-03-28/dotprod-sample-mean.png) ![fitted-model]({{ site.baseurl }}/assets/imgs/2017-03-28/dotprod-samples.png)


Image generated from  the learned distriubtion and the samples path against the model:

![fitted-model]({{ site.baseurl }}/assets/imgs/2017-03-28/dotprod-gen-image.png) ![fitted-model]({{ site.baseurl }}/assets/imgs/2017-03-28/dotprod-fit-sample.png)

### Work in Progress

My main concern is that all of the kernel functions seem to be biased to values lower than the median as it moves closer to the center. This may not matter too much for disk objects, but for spheroid objects I think this will be more problematic. I ran the kernels for the spheroid surface brightness profiles and observed the same behavior. Right now, I am exploring, sums, products, and exponenations of kernels along with defining a new kernel function. I think we should be able to find something that works. 
