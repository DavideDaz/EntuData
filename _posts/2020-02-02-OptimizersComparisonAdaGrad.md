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


## SGD with Nesterov Momentum implementation

Initialize global learning rate $$\epsilon$$, the array of the weights $$\theta$, and the gradient accumulation variable $$r$$.
Define a small value $$\delta$$ to avoid division by zero in the algorithm.
* Calculate the gradient
$$ \textbf{g} \leftarrow \frac{1}{m}\nabla_\theta\sum_{i=1}^m L(\textit{f} (\mathbf{x}^{(i)};\mathbf{\theta}),\mathbf{y}^{(i)})$$
* Accumulate squared gradient:
$$ \textbf{r} \leftarrow \textbf{r}+\textbf{g}\odot\textbf{g}$$
* Compute update:
$$ \mathbf{\Delta}\mathbf{\theta} \leftarrow -\frac{\epsilon}{sqrt{\delta+\textbf{r}}}\odot\textbf{g}$$
(Note that the multiplication is done element wise between the vector in which appear $$r$$ and the vector of the partisl derivatives for each element $$g$$)
* Apply update:
$$ \mathbf{\theta} \leftarrow \mathbf{\theta}+\mathbf{\Delta}\mathbf{\theta} $$


## Comparison with the simple SGD for the Quadratic Regression

You can find the notebook [here](https://github.com/DavideDaz/TokyoDataScience/blob/master/Assignments/Gradient%20Descent%20Assignment/Basis%20Neural%20Network%20-%20Quadratic%20-%20SGD%20with%20Nesterov%20momentum.ipynb). Right click on the Colab button and open in a new tab.

The effect of the Nesterov momentum is to quickly push down the loss function after 200 iteration avoiding the initial oscillations.

<img src="{{ site.url }}{{ site.baseurl }}/OptimizerComparison/lossNesterov.png" alt="loss Nesterov" class="align-center">
<figcaption>Figure 2: Loss function over the epochs</figcaption>
