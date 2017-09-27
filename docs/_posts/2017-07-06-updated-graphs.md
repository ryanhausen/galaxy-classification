---
layout: default
title: "Updated Graphs"
date: 2017-07-06
categories:
---

{% include mathjax.html  %}

### Images of V-Band Outlier

Ordered by V-band signal = $\sum_{r=0}^{I_e}I_r$, descending

wide2_386

![img]({{ site.url }}/assets/imgs/2017-07-06/wide2_386.png)

[FITS FILES]({{ site.url }}/assets/files/2017-07-06/wide2_386.tar.gz)

ers2_14166

![img]({{ site.url }}/assets/imgs/2017-07-06/ers2_14166.png)

[FITS FILES]({{ site.url }}/assets/files/2017-07-06/ers2_14166.tar.gz)

ers2_14166

![img]({{ site.url }}/assets/imgs/2017-07-06/ers2_14166.png)

[FITS FILES]({{ site.url }}/assets/files/2017-07-06/ers2_14166.tar.gz)

ers2_15757

![img]({{ site.url }}/assets/imgs/2017-07-06/ers2_15757.png)

[FITS FILES]({{ site.url }}/assets/files/2017-07-06/ers2_15757.tar.gz)

ers2_16767

![img]({{ site.url }}/assets/imgs/2017-07-06/ers2_16767.png)

[FITS FILES]({{ site.url }}/assets/files/2017-07-06/ers2_16767.tar.gz)



Ordered by V-band signal = max(vband-img)

wide2_1468

![img]({{ site.url }}/assets/imgs/2017-07-06/wide2_1468.png)

[FITS FILES]({{ site.url }}/assets/files/2017-07-06/wide2_1468.tar.gz)

ers2_12623

![img]({{ site.url }}/assets/imgs/2017-07-06/ers2_12623.png)

[FITS FILES]({{ site.url }}/assets/files/2017-07-06/ers2_12623.tar.gz)

ers2_14546

![img]({{ site.url }}/assets/imgs/2017-07-06/ers2_14546.png)

[FITS FILES]({{ site.url }}/assets/files/2017-07-06/ers2_14546.tar.gz)

deep2_10673

![img]({{ site.url }}/assets/imgs/2017-07-06/deep2_10673.png)

[FITS FILES]({{ site.url }}/assets/files/2017-07-06/deep2_10673.tar.gz)

ers3_12930

![img]({{ site.url }}/assets/imgs/2017-07-06/ers2_12930.png)

[FITS FILES]({{ site.url }}/assets/files/2017-07-06/ers2_12930.tar.gz)


### $I_e$ $R_e$ Scatterplot

The Pearson Correlation Coeffiecent is at the top of each graph in the title

{% assign bands = "h,j,v,z" | split: "," %}
{% for b in bands %}
	{% assign img_name = "spheroid-" | append: b | append: "-iere-scatter.png" %}
	
![sph-iere]({{ site.url }}/assets/imgs/2017-07-06/{{ img_name }})
{% endfor %}


{% assign bands = "h,j,v,z" | split: "," %}
{% for b in bands %}
	{% assign img_name = "disk-" | append: b | append: "-iere-scatter.png" %}
	
![disk-iere]({{ site.url }}/assets/imgs/2017-07-06/{{ img_name }})
{% endfor %}

### SNR in Logspace

There doesn't seem to be a strong correlation between $I_e$ and $R_e$.

Hisograms from original data

{% assign bands = "h,j,v,z" | split: "," %}
{% for b in bands %}
	{% assign img_name = "spheroid-" | append: b | append: "-snr-hist.png" %}

![sph-hist]({{ site.url }}/assets/imgs/2017-07-06/{{ img_name }})
{% endfor %}

{% assign bands = "h,j,v,z" | split: "," %}
{% for b in bands %}
	{% assign img_name = "disk-" | append: b | append: "-snr-hist.png" %}
	
![dk-hist]({{ site.url }}/assets/imgs/2017-07-06/{{ img_name }})
{% endfor %}


### Histogram $I_e$

Hisograms from original data

{% assign bands = "h,j,v,z" | split: "," %}
{% for b in bands %}
	{% assign img_name = "spheroid-" | append: b | append: "-ie-hist.png" %}

![sph-hist]({{ site.url }}/assets/imgs/2017-07-06/{{ img_name }})
{% endfor %}

{% assign bands = "h,j,v,z" | split: "," %}
{% for b in bands %}
	{% assign img_name = "disk-" | append: b | append: "-ie-hist.png" %}
	
![dk-hist]({{ site.url }}/assets/imgs/2017-07-06/{{ img_name }})
{% endfor %}


### 3D Gaussian of $R_e$, $I_e$, and Axis Ratio

{% assign bands = "h,j,v,z" | split: "," %}
{% for b in bands %}
	{% assign img_name = "spheroid-" | append: b | append: "-ierear-scatter.png" %}

![sph-hist]({{ site.url }}/assets/imgs/2017-07-06/{{ img_name }})
{% endfor %}

{% assign bands = "h,j,v,z" | split: "," %}
{% for b in bands %}
	{% assign img_name = "disk-" | append: b | append: "-ierear-scatter.png" %}
	
![dk-hist]({{ site.url }}/assets/imgs/2017-07-06/{{ img_name }})
{% endfor %}




