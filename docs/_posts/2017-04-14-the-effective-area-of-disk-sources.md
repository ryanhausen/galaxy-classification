---
layout: default
title: "The Effective Area of Disk Sources"
date: 2017-04-14
categories:
---

{% include mathjax.html  %}


### Source Image

![segmap]({{ site.url }}/assets/imgs/2017-04-14/segmap.png)

![src-image]({{ site.url }}/assets/imgs/2017-04-14/src-image.png)

### Major And Minor Axis

Using the same technique from a [prev post]({{ site.baseurl }}{% post_url 2017-04-07-finding-major-and-minor-axis-in-disk-sources %}) we can find the major and minor axis and their ratio.

![maj-min-axis]({{ site.url }}/assets/imgs/2017-04-14/maj-min-axis.png)

### From Circular to Elliptical Effective Radius

We can find $R_e$ manually by measuring out from the the center of the source until we get reach $I_e$

![circ-re]({{ site.url }}/assets/imgs/2017-04-14/circ-re.png)

To find the ellipse that has the same area as the circle above with the same axis ratio as the major and minor axis above we solve for a and b in the following equations:

$$\pi ab = \pi R_e^2$$

$$ \frac{a}{b} = Q $$


Where $Q$ is the axis ratio found in our eigen decompostion of the image covariance matrix. 

With that solved we get the following ellipse

![ell-re]({{ site.url }}/assets/imgs/2017-04-14/ell-re.png)

If we perform this for the image in all four bands we get the following. I lowered the alpha int the images so the ellipses would be easier to see. 

![ell-all]({{ site.url }}/assets/imgs/2017-04-14/ell-all.png)

