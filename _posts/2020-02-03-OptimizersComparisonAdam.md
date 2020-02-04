---
title: "Machine Learning Theory Comprehension: Optimizers Comparison on a Quadratic Regression, Adam"                
date: 2020-01-28
tags: [Machine Learning Theory Comprehension]
header:
  image: "/OptimizerComparison/losslandscapeAdam.jpg"
excerpt: "Machine Learning, Optimizer Comparison"
mathjax: "true"
toc: true
toc_label: "Contents"
toc_icon: "cog"
---


## Adam optimizer Rationale

The ADAptive Moment Optimizer blends together the methods of RMSProp and Momentum by applying the momentum to the rescaled gradients. This is done by storing an **exponentially decay average** of:

* Past gradients like the Momentum method
* Past element-wise square gradient like RMSPRop

The two accumulating terms are estimates respectively of the first moment (Mean) and second moment (uncenterced variance) of the gradients.
As the terms are initialized with zero, their moments are biased towards zero, especially during the first iteration and for small decay rates. The methods introduces therefore a bias-correction for the mean and variance estimates.

## Adam implementation

Initialize global learning rate $$\epsilon$$, the array of the weights $$\theta$$,the gradient estimates accumulation variables for the mean $$s$$ and variance $$r$$ and their decay rates $$\rho1$$ and $$\rho2$$.
Define a small value $$\delta$$ to avoid division by zero in the algorithm.
* Calculate the gradient
$$ \textbf{g} \leftarrow \frac{1}{m}\nabla_\theta\sum_{i=1}^m L(\textit{f} (\mathbf{x}^{(i)};\mathbf{\theta}),\mathbf{y}^{(i)})$$
* t \leftarrow t+1
* Update biased first moment estimate:
$$ \textbf{s} \leftarrow \rho1\textbf{s}+(1-\rho1)\textbf{g}$$
* Update biased second moment estimate:
$$ \textbf{r} \leftarrow \rho2\textbf{r}+(1-\rho2)\textbf{g}\odot\textbf{g}$$
* Correct bias in first moment:
$$\widehat{\textbf{s}} \leftarrow \frac{\textbf{s}}{1-{\rho1}^t}$$
* Correct bias in second moment:
$$\widehat{\textbf{r}} \leftarrow \frac{\textbf{r}}{1-{\rho2}^t}$$
* Compute update:
$$ \mathbf{\Delta}\mathbf{\theta} \leftarrow -\epsilon\frac{\widehat{\textbf{s}}}{\delta+\sqrt{\widehat{\textbf{r}}}}$$
(Note that the multiplication is done element wise between the vector in which appear $$r$$ and the vector of the partisl derivatives for each element $$g$$)
* Apply update:
$$ \mathbf{\theta} \leftarrow \mathbf{\theta}+\mathbf{\Delta}\mathbf{\theta} $$


## AdaGrad Optimizer for a Quadratic Regression

You can find the **notebook** [here](https://github.com/DavideDaz/TokyoDataScience/blob/master/Assignments/Gradient%20Descent%20Assignment/Basis%20Neural%20Network%20-%20Quadratic%20-%20Adam.ipynb). Right click on the Colab button and open in a new tab.

You can see below that the result does not differ too mush from the RMSProp for the quadratic regression case.

<figure class="half full">
<img src="{{ site.url }}{{ site.baseurl }}/OptimizerComparison/AdamReg.png" alt="Adam Reg">
<img src="{{ site.url }}{{ site.baseurl }}/OptimizerComparison/AdamLoss.png" alt="Adam Loss">
<figcaption>Figure 2: Regression and Loss function for epsilon global = 0.01</figcaption>
</figure>

## References

+ [1] https://ruder.io/optimizing-gradient-descent/index.html#gradientdescentoptimizationalgorithms
+ [2] http://www.deeplearningbook.org/
