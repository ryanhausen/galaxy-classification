---
layout: default
title: "Fitting Sersic Profiles To Raw Data"
date: 2017-02-07
categories: synth-images
---

```
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
```



To create more accurate synthetic images for training the model we fit all the source images to a Sersic profile. Equation we use to define the Sersic profile is:

$$ I_e \exp(-b_m((\frac{R}{R_e})^{\frac{1}{m}}-1) )$$

Where $m=1$ defines an exponential disk profile, $m=4$ defines a De Vaucouleurs spheroid profile, and $b_m=2m-0.324$

The parameters we want to fit are $I_e$ and $R_e$. Which will be optimized using `fmin`  from `scipy.optimize`

```python
{% highlight python %}
import numpy as np

def sersic(Ie, Re, R, m):
    bm = 2.0*m - 0.324
    return Ie * np.exp(-bm * ((R/Re)**(1.0/m) - 1.0))

def sersic_optimize(x, m, y):
    """
    	x: an indexed object with two items Ie=x[0], Re=[1]
    	m: the m param for the sersic profile
    	y: the measured surface brightness profile of the object
    """
    
    pix = .06  # arc/pixel
    bins = 100 # number of bins 
    center_dist = np.sqrt((0.5*84)**2 + (0.5*84)**2)
    r = .06 * center_dist * np.arange(bins) / float(bins)
	
	# get the sersic profile for Ie = x[0] and Re = x[1]
    s = sersic(x[0], x[1], r, m)
    
    # return the squared error
    return ((s[y>0]-y[y>0])**2).mean()

{% endhighlight %}
```

...Still working on 