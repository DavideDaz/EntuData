---
title: "Machine Learning Theory Comprehension: Optimizers Comparison on a Quadratic Regression, SGD with Nesterov Momentum"                
date: 2020-01-28
tags: [Machine Learning Theory Comprehension]
header:
  image: "/OptimizerComparison/loss-landscapeNesterovRes.jpg"
excerpt: "Machine Learning, Optimizer Comparison"
mathjax: "true"
toc: true
toc_label: "Contents"
toc_icon: "cog"
---

In the previous post I anticipated that the gradient descent with Momentum alone show a limitation in case of changes of the loss function. Because of its update method, the step keeps 'accelerating' even though there is an imminent slope up of the loss function, leading therefore the parameters away from the target.


## SGD with Nesterov Momentum Rationale
The idea behind the Nesterov momentum is to sense the change in the loss function and apply a correction to the velocity term in order to predict its trend.

To do this, the Nesterov method applies an initial estimate of the parameters with the velocity calculated in the previous iteration. Starting from this position the gradient is calculated and a further update is applied to the weights in order to introduce a correction. Now, if the initial update was in the direction of the local minima, no or small correction is applied. In case the loss calculated with the initial update increases, the correction tends to adjust the direction again towards the minima.

We can see from Figure 1 the difference of the two approaches:

* The simple momentum optimization first calculates the gradient (small blue vector) and after add the contribute of the previous velocity (long blue vector)

* The Nesterov momentum initially updates the weights based on the previous velocity (brown vector) and based on the gradient estimate it applies a correction (red vector). The overall update is given by the green vector.

<img src="{{ site.url }}{{ site.baseurl }}/OptimizerComparison/nesterovarrow.png" alt="NesterovArrows">
<figcaption>Figure 1: SGD with Momentum and SGD with Nesterov Momentum weights update (Source: G. Hinton's lecture 6c)</figcaption>


## SGD with Nesterov Momentum implementation

Its implementation differs from the other just for the instant at which the gradient is calculated:

* Update the weights with previously calculated velocity:
$$ \mathbf{\widetilde{\theta}} \leftarrow \mathbf{\theta}+\alpha\textbf{v}$$
* Calculate the gradient based on the new weights
$$ \textbf{g} \leftarrow \frac{1}{m}\nabla_\theta\sum_{i=1}^m L(\textit{f} (\mathbf{x}^{(i)};\mathbf{\widetilde{\theta}}),\mathbf{y}^{(i)})$$
* Compute the velocity update:
$$ \textbf{v} \leftarrow \alpha\textbf{v}-\epsilon\textbf{g}$$
* Apply update:
$$ \mathbf{\theta} \leftarrow \mathbf{\theta}+\textbf{v}$$


## Comparison with the simple SGD for the Quadratic Regression

You can find the **notebook** [here](https://github.com/DavideDaz/TokyoDataScience/blob/master/Assignments/Gradient%20Descent%20Assignment/Basis%20Neural%20Network%20-%20Quadratic%20-%20SGD%20with%20Nesterov%20momentum.ipynb). Right click on the Colab button and open in a new tab.

The effect of the Nesterov momentum is to quickly push down the loss function after 200 iteration avoiding the initial oscillations.

<img src="{{ site.url }}{{ site.baseurl }}/OptimizerComparison/lossNesterov.png" alt="loss Nesterov" class="align-center">
<figcaption>Figure 2: Loss function over the epochs</figcaption>

## References

+ [1] https://ruder.io/optimizing-gradient-descent/index.html#gradientdescentoptimizationalgorithms
+ [2] http://www.deeplearningbook.org/
