---
title: "Machine Learning Theory Comprehension: Optimizers Comparison"                
date: 2020-01-28
tags: [machine learning, data science, optimizers, machine learning theory]
header:
  image: "/OptimizerComparison/loss-landscapeR.jpg"
excerpt: "Machine Learning Projects, Optimizer Comparison, Data Science"
---

During my study at Tokyo Data Science I am constantly exploring new topics of Machine Learning and the elated theoretical bases of the technique employed to develop the models. If from one hand it comes easy to get the general idea of a particular method, on the other the understanding of the implementation on the code and its real effect on the model bear a certain level of abstraction.This is especially true if dealing with deep neural network, in which the choice of a different parameter or setup is integrated in a elaborated architecture and the real perception of a modification is often clouded by its complexity.

Rather than a very complex net and taking advantage of some assignment and similar examples on various topics taken from the Web, I therefore decided to break down the problem starting from a very simple case and apply the techniques of a particular subject to clearly unveil the effect of each method.

In this post I will start from a simple quadratic regression to compare the most common Optimizers currently employed to train ML models:

* SGD with Momentum
* SGD with Nesterov momentum
* AdaGrad and its evolution in RSMprop
* Adam   

Please note that some of these techniques, in particular the last three, are particularly conceived for applications to deep learning in cases with complex landscape loss functions (Like the one at the head of this post taken from [losslandscape](https://losslandscape.com/), which I strongly suggest to take a look at). Their use on a concave function may therefore lead to the expected result but still gives the clear idea of the rationale behind them.


## SGD with Momentum

The learning rate is a crucial parameter that determines the dimension of step to upload the weights after each iteration during the gradient descent. The simple SGD employs a constant learning rate. This means that all the parameters are updated with the same step independently from the "shape" of the loss function. In this situation, this can result in a slow journey to the local minima as shown in the case of a quadratic loss function with a poorly conditioned Hessian matrix of Figure 1, considering a simple case of two weights. Note that the loss is represented with contour lines as a function of the two parameters.

<img src="{{ site.url }}{{ site.baseurl }}/OptimizerComparison/MomentumandSGD.png" alt="SGD and SGD with momentum">
<figcaption>Figure 1: SGD and SGD with momentum</figcaption>

It is clear that to reach faster the local minima, the update on the horizontal direction has to move with a "faster peace". Now, given that the actual step is evaluated by multiplying the calculated gradient to the learning rate, it is reasonable to think that if the value of the gradient had a major increase on a particular direction, this trend will continue for the next step, unless sudden changes in the loss function.

We can think of the gradient as a sort of velocity, with its magnitude and a direction, somehow pointing with a certain intensity to the local minima. It can be therefore advantageous to take advantage of the momentum given by the previous iteration to speed up the learning process. At the same time we don't want to accelerate indefinitely and amplify too much the time step as we can overshoot our minima target. Hence, without going too much in deep with the physical parallel of momentum (whose clear explanation you can find [here](http://www.deeplearningbook.org/contents/optimization.html) for the SGD with momentum implementation we mainly need two ingredients:

- A variable $$ v $$ that plays the role of velocity
- A friction coefficient that introduces a gradual (exponential) decay of the velocity term in order to dump the oscillations in the weights updates, reaching a terminal velocity close to the minima.

The result it is a faster descent towards the local minima with a dumped oscillation of the weights as in Figure 2.

<img src="{{ site.url }}{{ site.baseurl }}/OptimizerComparison/Momentum.png" alt="SGD with momentum">
<figcaption>Figure 2: SGD with momentum</figcaption>
