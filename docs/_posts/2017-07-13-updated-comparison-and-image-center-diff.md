---
layout: default
title: "Updated Comparison and Image Center Diff"
date: 2017-07-13
categories:
bands:
  - "h"
  - "j"
  - "v"
  - "z"
---

{% include mathjax.html  %}

### Visual Comparison of Max Pixel and Image Moment

**Centroid Text Above Pixel -- Max Text Below Pixel**

{% for b in page.bands %}
	{% assign img_name = "spheroid-" | append: b | append: "-center-diff.png" %}

![sph-center]({{ site.url }}/assets/imgs/2017-07-13/{{ img_name }})

{% endfor %}

{% for b in page.bands %}
	{% assign img_name = "disk-" | append: b | append: "-center-diff.png" %}

![disk-center]({{ site.url }}/assets/imgs/2017-07-13/{{ img_name }})

{% endfor %}

### Top V→H Ratio Outliers 

Here are the top 5 V→H ratio outliers. The ratio is taken at r=0, using $I_e$ normalized values.

Using the former calculation(now current), $I_e=\frac{1}{n}(\sum_{r=0}^{R_e}I_r)$

ers2_14393, V→H: 27.21627

![img]({{ site.url }}/assets/imgs/2017-07-13/ers2_14393.png)
[fits]({{ site.url }}/assets/files/2017-07-13/ers2_14393.tar.gz)

wide2_14393, V→H: 27.21627

![img]({{ site.url }}/assets/imgs/2017-07-13/wide2_14393.png)
[fits]({{ site.url }}/assets/files/2017-07-13/wide2_14393.tar.gz)

deep2_9832, V→H: 22.588188

![img]({{ site.url }}/assets/imgs/2017-07-13/deep2_9832.png)
[fits]({{ site.url }}/assets/files/2017-07-13/deep2_9832.tar.gz)

ers2_12623, V→H: 15.808301

![img]({{ site.url }}/assets/imgs/2017-07-13/ers2_12623.png)
[fits]({{ site.url }}/assets/files/2017-07-13/ers2_12623.tar.gz)

wide2_3550, V→H: 15.08471

![img]({{ site.url }}/assets/imgs/2017-07-13/wide2_3550.png)
[fits]({{ site.url }}/assets/files/2017-07-13/wide2_3550.tar.gz)


Using the new calculation(now former), $I_e = (\sum_{r=0}^{R_e}I_r)/2\pi R_e^2$

deep2_6739, V→H: 8.3352230869571535

![img]({{ site.url }}/assets/imgs/2017-07-13/deep2_6739.png)
[fits]({{ site.url }}/assets/files/2017-07-13/deep2_6739.tar.gz)

wide2_3550, V→H: 6.2215523992065398

![img]({{ site.url }}/assets/imgs/2017-07-13/wide2_3550.png)
[fits]({{ site.url }}/assets/files/2017-07-13/wide2_3550.tar.gz)

ers2_12623, V→H: 4.3822057636621583

![img]({{ site.url }}/assets/imgs/2017-07-13/ers2_12623.png)
[fits]({{ site.url }}/assets/files/2017-07-13/ers2_12623.tar.gz)

wide2_10906, V→H: 4.2032536117614816) 

![img]({{ site.url }}/assets/imgs/2017-07-13/wide2_10906.png)
[fits]({{ site.url }}/assets/files/2017-07-13/wide2_10906.tar.gz)

wide2_208, V→H:  4.0198099171477359)

![img]({{ site.url }}/assets/imgs/2017-07-13/wide2_208.png)
[fits]({{ site.url }}/assets/files/2017-07-13/wide2_208.tar.gz)