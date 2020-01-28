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

* SGD (Given the few amount of generated data I neglected the minibatch implementation, considering a unique batch of 100 elements)
* SGD with momentum
* SGD with Nesterov momentum
* AdaGrad and its evolution in RSMprop
* Adam   

Please note that some of these techniques, in particular the last three, are particularly conceived for applications to deep learning in cases with complex landscape loss functions (Like the one at the head of this post taken from [losslandscape](https://losslandscape.com/[]), which I strongly suggest to take a look at). Their use on a concave function may therefore lead to the expected result but still gives the clear idea of the idea behind them.
