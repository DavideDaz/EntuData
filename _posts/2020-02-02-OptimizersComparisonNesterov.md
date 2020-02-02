---
title: "Machine Learning Theory Comprehension: Optimizers Comparison on a Quadratic Regression, SGD with Nesterov Momentum"                
date: 2020-01-28
tags: [machine learning, data science, optimizers, machine learning theory]
header:
  image: "/OptimizerComparison/loss-landscapeNesterovRes.jpg"
excerpt: "Machine Learning Projects, Optimizer Comparison, Data Science"
mathjax: "true"
toc: true
toc_label: "Contents"
toc_icon: "cog"
---

In the previous post I anticipated that the gradient descent with Momentum alone show a limitation in case of changes of the loss function. Because of its update method the step keeps 'accelerating' even though the loss function starts sloping up again, leading therefore the parameters away from the target.

The idea behind the Nesterov momentum is to sense the change in the loss function and apply as correction to the velocity term in order to predict the loss function trend.


## SGD with Nesterov Momentum Rationale
The idea behind the Nesterov momentum is to sense the change in the loss function and apply a correction to the velocity term in order to adapt the step to this sudden change.

Therefore, the Nesterov method applies an initial estimate of the parameters with the velocity calculated in the previous iteration. Starting from this position, the gradient is calculated and based on this a further update to the weights is applied in order to introduce a correction. If the initial update was in the direction of the local minima, no or small correction is applied. In case the loss calculated with the initial update increases, the correction tends to adjust the direction again towards the minima.

We can see from Figure 1 the difference of the two approaches:

* The simple momentum optimization first calculates the gradient (small blue vector) and after add the contribute of the previous velocity (long blue vector)

* The Nesterov momentum initially updates the weights based on the previous velocity (brown vector) and based on the gradient estimation it applies a correction (red vector). The overall update is given by the green vector.

<img src="{{ site.url }}{{ site.baseurl }}/OptimizerComparison/nesterovarrow.png" alt="Nesterov Arrows">
<figcaption>Figure 1: SGD with Momentum and SGD with Nesterov Momentum weights update (Source: G. Hinton's lecture 6c)</figcaption>


## SGD with Nesterov Momentum implementation

Its implementation differs from the other just for the instant at which the gradient is calculated:

* Update the weights with previously calculated velocity:
$$ \mathbf{\thetatilde} \leftarrow \mathbf{\theta}+\textbf{v}$$
* Calculate the gradient based on the new weights
$$ \textbf{g} \leftarrow \frac{1}{m}\nabla_\theta\sum_{i=1}^m L(\textit{f} (\mathbf{x}^{(i)};\mathbf{\thetatilde}),\mathbf{y}^{(i)})$$
* Compute the velocity update:
$$ \textbf{v} \leftarrow \alpha\textbf{v}-\epsilon\textbf{g}$$
* Apply update:
$$ \mathbf{\theta} \leftarrow \mathbf{\theta}+\textbf{v}$$


## Comparison with the simple SGD for the Quadratic Regression

You can find the notebook [here](https://github.com/DavideDaz/TokyoDataScience/blob/master/Assignments/Gradient%20Descent%20Assignment/Basis%20Neural%20Network%20-%20Quadratic%20-%20SGD%20with%20Nesterov%20momentum.ipynb). Right click on the Colab button and open in a new tab.

The effect of the Nesterov momentum is to quickly push down the loss function after 200 iteration avoiding the initial oscillations.

<img src="{{ site.url }}{{ site.baseurl }}/OptimizerComparison/lossNesterov.png" alt="loss Nesterov" class="align-center">
<figcaption>Figure 2: Loss function over the epochs</figcaption>
