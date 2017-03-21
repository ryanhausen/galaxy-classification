---
layout: default
title: "Synthetic Images Using Gaussian Process"
date: 2017-03-20
categories: synth-images, gaussian-process
---

{% include mathjax.html %}

### Adding Noise The Analytic Graph

![analytic-with-noise]({{ site.url }}/assets/imgs/2017-03-20/analytic-with-noise.png)


### H Band Spheroid Median plotted with Disk Median

![median-overplot]({{ site.url }}/assets/imgs/2017-03-20/median-overplot.png)


### Ratio of Bands

![disk-band-ratio]({{ site.url }}/assets/imgs/2017-03-20/disk-band-ratio.png)

![spheroid-band-ratio]({{ site.url }}/assets/imgs/2017-03-20/spheroid-band-ratio.png)


## Fitting with a Gaussian Process

We'll fit using the data from the Disk class in the H-Band which looks like this after we moved the data to log scale:

![disk-h-band]({{ site.url }}/assets/imgs/2017-03-20/disk-h-band.png)

The data is heteroskedastic, we'll remedy that by taking the mean of the difference between the $84^{th}$ percentile and the median.

![disk-h-band-homoskedastic]({{ site.url }}/assets/imgs/2017-03-20/disk-h-band-homoskedastic.png)

To fit the Gaussian process we'll use the following kernel function:

$$k(x,x')=\sigma_f^2\exp\{-\frac{(x-x')^2}{2l^2}\}$$

Where $\sigma_f$ is a parameter that controls the variance and $l$ is the length scale parameter.

<!--
To find these paramters we will minimize the negative log likelihood:

$$\log(\textbf{y}|\textbf{x},\Theta) = -\frac{1}{2}\textbf{y}^\text{T}K^{-1}\textbf{y}-\frac{1}{2}\log|K|-\frac{n}{2}\log2\pi$$

Where $K=k(x,x)+\sigma_n^2\delta(x,x')$, $\delta(x,x')$ is the Kronecker delta function, $\sigma_n$ is from our measured data, and $\Theta$ is a vector of our paramters $\sigma_f$ and $l$.

With the fit paramters, we can predict new values using the following:
-->

With the paramters, we can predict new values using the following:

$$y_*|\textbf{y} \text{~} \mathcal{N}(K_*K^{-1}\textbf{y}, K_{**}-K_*K^{-1}K_*^\text{T}$$

Where $K=k(x,x)+\sigma_n^2\delta(x,x')$, $K_*=k(x,x')$, and $K_{**}=k(x',x')$

The best predicted value for point $y_*$ is the mean of the distribution:

$$\bar{y_*}= K_*K^{-1}\textbf{y}$$

And the uncertainty is in the variance of the distribution:

$$\text{var}(y_*)=  K_{**}-K_*K^{-1}K_*^\text{T}$$

With the fitted paramters and the calculations to make predictions, we can now sample from this process to create images.

To sample unique value we introduce some noise to $\textbf{y}$ by adding from a normal distribution with $\sigma=\sigma_n$. Although this may cause the observed values to be more jumpy, they should still be within an acceptable range, and the sample we draw from will tend to be smooth as a result of the Gaussian Process.

Using the new data we will use the Gaussian Process to predict $I_e$ normalized intensities at different $R_e$ normalized radii.

To generate an sample image we choose the paramters, $I_0$, $R_e$, $R$, then we draw a sample from the Gaussian Process getting $y_*$ for each pixel's radus value normalized to $R_e$

For example if we set: $I_0=1$, $R_e=0.24$ $R=4.0$, the scale arcsecond/pixel = $0.06$.

We get this source and surface birghtness profile in log scale.:

![sample-source]({{ site.url }}/assets/imgs/2017-03-20/sample-source.png)









