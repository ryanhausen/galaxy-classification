---
layout: default
title: "Adding Data for a Better GP Fit"
date: 2017-04-03
categories: synthdata
---


{% include mathjax.html  %}


### Problem

The GP fit that we were getting wasn't capturing our data accurately as we appoached the zero. To try and fix this we'll add mock negative data to pull the fit line up.

The data with the current fit using the disks in the H band:



![data-points]({{ site.baseurl }}/assets/imgs/2017-04-03/data-points.png)

![data-with-fit]({{ site.baseurl }}/assets/imgs/2017-04-03/data-with-fit.png)



To make the mock data we fit two lines. The first was a second degree polynomial to the median values and the second line was a simple line to the first three values in the $84^{th}$ percentile. 

The coefficients for the median function are (0.12389692, -1.54251171, 1.22524334) and for the $84^{th}$ are (-1.89426097, 1.67573137). With the mock data we have the following graph:

![data-with-mock]({{ site.baseurl }}/assets/imgs/2017-04-03/data-with-mock.png)

The new fitted function looks like this:

![data-with-mock-fit]({{ site.baseurl }}/assets/imgs/2017-04-03/data-with-mock-fit.png)

And at the same zoom:

![data-with-mock-fit]({{ site.baseurl }}/assets/imgs/2017-04-03/data-with-mock-fit-same-dim.png)

This looks much better for disks here is a generated image:

![sample-image]({{ site.baseurl }}/assets/imgs/2017-04-03/sample-image.png)
![fit-with-sample]({{ site.baseurl }}/assets/imgs/2017-04-03/fit-with-sample.png)

The recovered mean:

![recovered-mean]({{ site.baseurl }}/assets/imgs/2017-04-03/recovered-mean.png)

Samples from the fitted process:

![samples]({{ site.baseurl }}/assets/imgs/2017-04-03/samples.png)
