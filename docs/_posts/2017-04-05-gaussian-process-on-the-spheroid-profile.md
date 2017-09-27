---
layout: default
title: "Gaussian Process on the Spheroid Profile"
date: 2017-04-05
categories:
---

{% include mathjax.html  %}


This post is an extension of the previous post but with the spheroid surface brightness profile in the H band this time.

To make the mock data we fit two lines. The first was a second degree polynomial to the median values and the second line was a simple line to the first three values in the $84^{th}$ percentile. 

![data-with-mock]({{ site.baseurl }}/assets/imgs/2017-04-05/data-with-mock.png)

The new fitted function looks like this:

![data-with-mock-fit]({{ site.baseurl }}/assets/imgs/2017-04-05/data-with-mock-fit.png)

And at the same zoom:

![data-with-mock-fit]({{ site.baseurl }}/assets/imgs/2017-04-05/data-with-mock-fit-same-dim.png)

This looks much better here is a generated image:

![sample-image]({{ site.baseurl }}/assets/imgs/2017-04-05/sample-image.png)
![fit-with-sample]({{ site.baseurl }}/assets/imgs/2017-04-05/fit-with-sample.png)

The recovered mean:

![recovered-mean]({{ site.baseurl }}/assets/imgs/2017-04-05/recovered-mean.png)

Samples from the fitted process:

![samples]({{ site.baseurl }}/assets/imgs/2017-04-05/samples.png)