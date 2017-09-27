---
layout: default
title: "Creating Synthetic Images With Noise"
date: 2017-05-31
categories:
---

{% include mathjax.html  %}


Noise is added to the image in the following way:
- Noise pixels are drawn from a collection of pixels selected at random from a collection of noise pixels specific to the band we need.
- Noise is scaled accoding to a signal to noise ratio uniformly drawn from a collection of signal to noise ratios for all images.

### Noise Pixels 
The collection of noise pixels for each band was taken from an image from the collection which had the most noise in its image. 

### Scaling the Noise
For this process the signal to noise ratio is defined as:

$$ \max(signal)/\text{mean}(noise) $$

In my experiments this seemed to work the best. The signal to noise values for a single image are drawn at random and these are used to scale the noise so that the ratio holds for our synthetic image.

### Example

![h]({{ site.baseurl }}/assets/imgs/2017-05-31/h-no-noise.png)
![h]({{ site.baseurl }}/assets/imgs/2017-05-31/h-noise.png)

![j]({{ site.baseurl }}/assets/imgs/2017-05-31/j-no-noise.png)
![j]({{ site.baseurl }}/assets/imgs/2017-05-31/j-noise.png)

![v]({{ site.baseurl }}/assets/imgs/2017-05-31/v-no-noise.png)
![v]({{ site.baseurl }}/assets/imgs/2017-05-31/v-noise.png)

![z]({{ site.baseurl }}/assets/imgs/2017-05-31/z-no-noise.png)
![z]({{ site.baseurl }}/assets/imgs/2017-05-31/z-noise.png)