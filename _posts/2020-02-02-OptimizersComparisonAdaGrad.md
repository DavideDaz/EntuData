---
title: "Machine Learning Theory Comprehension: Optimizers Comparison on a Quadratic Regression, AdaGrad and RMSProp"                
date: 2020-01-28
tags: [machine learning, data science, optimizers, machine learning theory]
header:
  image: "/OptimizerComparison/loss-landscape-AdaGRes.jpg"
excerpt: "Machine Learning Projects, Optimizer Comparison, Data Science"
mathjax: "true"
toc: true
toc_label: "Contents"
toc_icon: "cog"
---

The optimization methods described in the previous two posts (SGD with momentum and with Nesterov momentum) employ a fixed learning rate, independently from the type of descent of the weights along the loos function.

AdaGrad propose instead an adaptive learning rate depending on the historical squared values of the gradient. Let's see this in details.

## AdaGrad optimizer Rationale


The learning rate is adapted for each parameter by scaling them with the square root of a term called $$r$$. This is computed by adding the value stored from the previous iteration to the element-wise squared values of the gradient. This means that the parameters with a systematic large partial derivative experience a rapid decrease in the learning rate, while the parameters with a small partial derivative slightly decrease their learning rate over the iterations.


## AdaGrad implementation

Initialize global learning rate $$\epsilon$$, the array of the weights $$\theta$$, and the gradient accumulation variable $$r$$.
Define a small value $$\delta$$ to avoid division by zero in the algorithm.
* Calculate the gradient
$$ \textbf{g} \leftarrow \frac{1}{m}\nabla_\theta\sum_{i=1}^m L(\textit{f} (\mathbf{x}^{(i)};\mathbf{\theta}),\mathbf{y}^{(i)})$$
* Accumulate squared gradient:
$$ \textbf{r} \leftarrow \textbf{r}+\textbf{g}\odot\textbf{g}$$
* Compute update:
$$ \mathbf{\Delta}\mathbf{\theta} \leftarrow -\frac{\epsilon}{\sqrt{\delta+\textbf{r}}}\odot\textbf{g}$$
(Note that the multiplication is done element wise between the vector in which appear $$r$$ and the vector of the partisl derivatives for each element $$g$$)
* Apply update:
$$ \mathbf{\theta} \leftarrow \mathbf{\theta}+\mathbf{\Delta}\mathbf{\theta} $$


## AdaGrad Optimizer for a Quadratic Regression

You can find the **notebook** [here](https://github.com/DavideDaz/TokyoDataScience/blob/master/Assignments/Gradient%20Descent%20Assignment/Basis%20Neural%20Network%20-%20Quadratic%20-%20AdaGrad.ipynb). Right click on the Colab button and open in a new tab.

For the quadratic regression the element-wise product of the gradient is systematically very large. Therefore, by using a small value of global learning rate the update of the weights ($$\Delta\theta$$) at every iteration is very small. In Figure 1 we can see that with a $$\epsilon = 0.01$$ the regression line can't fit the generated points  and the loss is still very high after 2000 epochs. This is because the updates are so small that the learning process of the gradient descent is strongly slowed down by the AdaGrad optimizer.

<figure class="half full">
<img src="{{ site.url }}{{ site.baseurl }}/OptimizerComparison/AdaReg01.png" alt="SGD with momentum loss">
<img src="{{ site.url }}{{ site.baseurl }}/OptimizerComparison/AdaLoss01.png" alt="SGD with momentum loss mag">
<figcaption>Figure 1: Regression and Loss function for epsilon global = 0.01</figcaption>
</figure>

By increasing the global learning rate, we can compensate the effect of AdaGrad. In Figure 2 is shown the attempt with a very large learning rate (epsilon global = 1)

<figure class="half full">
<img src="{{ site.url }}{{ site.baseurl }}/OptimizerComparison/AdaReg1.png" alt="SGD with momentum loss">
<img src="{{ site.url }}{{ site.baseurl }}/OptimizerComparison/AdaLoss1.png" alt="SGD with momentum loss mag">
<figcaption>Figure 1: Regression and Loss function for epsilon global = 0.01</figcaption>
</figure>

## RMSProp optimizer Rationale



## RMSProp implementation


## RMSProp Optimizer for a Quadratic Regression
