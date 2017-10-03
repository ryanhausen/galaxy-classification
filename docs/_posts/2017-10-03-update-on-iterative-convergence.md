---
layout: default
title: "Update on Iterative Convergence"
date: 2017-10-03
categories:
bands:
  - "h"
  - "j"
  - "v"
  - "z"
---

{% include mathjax.html  %}

## Each Sources Effective Can Be Measured One of 4 Ways

### The Source's Radius is <= 15 Pixels

The effective radius is measured within the segmap.

$$\Sum_r^{Re}Img_r / Sum_r^{R}Img_r = \frac{1}{2} $$

### The Effective Radius Can Be Measured Iteratively

**Steps**
1. Guess and $Re$
**While not converged**
2. Measure at csbp at $R_e$ and $5R_e$
3. If $\frac{R_e}{5R_e} != 0.5$ adjust $R_e$ to move $\frac{R_e}{5R_e}$ closer to 0.5

### The Effective Radius Won't Converge When Measured Iteratively

**Maximum Iterations Reached**
If the image doesn't converge take the pixel closest to what the effective radius would be.

**Effective Radius Out Grows Image Bounds**
If the image won't converge, then measure within the segmap.


## Results

![hist]({{site.baseurl}}/assets/imgs/2017-10-03/hist.png)
![cbsp]({{site.baseurl}}/assets/imgs/2017-10-03/csbp.png)
