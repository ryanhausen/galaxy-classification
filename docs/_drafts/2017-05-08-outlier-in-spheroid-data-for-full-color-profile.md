---
layout: default
title: "Outlier in Spheroid Data for Full Color Profile"
date: 2017-05-08
categories:
---

{% include mathjax.html  %}

### Updated Graph
After getting noisy results from the first run. I rexamined the algorithm I wrote and found some bugs and also some interesting artifacts in the data. After fixing the bugs and adding a new rule to exclude images where the centroids between different bands are more than 8 pixels away. The new color profiles look like this:

profiles

There is a considerable amount of noise just after 5 effective radaii. If we look at the histogram for that radial bin of values we see that there is a single outliers:

histograms

To remove that outlier I added a new rule to exclude sources where the color ratio value is greater than 10. With that new condition we get the following new spheroid profile:

new-spheroid-profile

Then with a GP trained on the new input the output sample looks like

new-gp-draw

new-image