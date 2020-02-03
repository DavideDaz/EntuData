---
title: "Machine Learning Theory Comprehension: Optimizers Comparison on a Quadratic Regression, SGD with Momentum"                
date: 2020-01-28
tags: [Machine Learning Theory Comprehension]
header:
  image: "/OptimizerComparison/loss-landscapeR.jpg"
excerpt: "Machine Learning, Optimizer Comparison"
mathjax: "true"
toc: true
toc_label: "Contents"
toc_icon: "cog"
---

During my study at Tokyo Data Science I am constantly exploring new topics of Machine Learning. We start the foundation of each subject by firstly covering the theoretical bases of the technique employed to develop the different architectures. Even though getting the general idea of a particular method may come easy, the understanding of the implementation on the code and its real effect on the model often bear a certain level of abstraction.This is especially true if dealing with deep neural network in which the choice of a particular parameter or setup is integrated in a elaborated architecture and the real perception of its effect is often clouded by the complexity of the net.

I therefore decided to break down each topic by supporting the theory with a very simple exercise or application (that often correspond to the school assignment) in order to effectively represent the idea of each technique.

In this series of posts I will employ the gradient descent on a simple quadratic regression to compare the most common Optimizers currently employed to train ML models:

* SGD with Momentum
* SGD with Nesterov momentum
* AdaGrad and its evolution in RSMprop
* Adam   

Please note that some of these techniques, in particular the last two, are particularly conceived for applications to deep learning in cases with complex landscape loss functions (Like the one at the head of this post taken from [losslandscape](https://losslandscape.com/), which I strongly suggest to take a look at. Therefore, their use on a the quadratic function may not lead to the expected result but still can clarify their rationale.


## SGD with Momentum Rationale

The learning rate is a crucial parameter that determines the dimension of step to upload the weights after each iteration during the gradient descent. The simple SGD employs a constant learning rate. This means that all the parameters are updated with the same learning rate independently from the "shape" of the loss function. This can result in a slow journey to the local minima as shown in the case of Figure 1, considering a simple case of two weights. Note that the loss is represented with contour lines as a function of the two parameters.
<img src="{{ site.url }}{{ site.baseurl }}/OptimizerComparison/MomentumandSGD.png" alt="SGD and SGD with momentum">
<figcaption>Figure 1: SGD and SGD with momentum</figcaption>

It is clear that to quickly reach the local minima, the update on the horizontal direction has to move with a "faster peace". Now, given that the actual step is evaluated by multiplying the calculated gradient to the learning rate, it is reasonable to think that if the value of the gradient had a relevant increase on a particular direction, this trend will continue for the next iteration (unless sudden changes in the loss function).

We can think of the gradient as a sort of vector, with its magnitude and a direction pointing with a certain intensity to the local minima. It can be therefore advantageous to take advantage of the momentum given by the previous iteration to speed up the learning process. At the same time we don't want to accelerate indefinitely and amplify too much the step size, as we can overshoot our minima target. Hence, without going too much in deep with the physical parallel of momentum (whose clear explanation is in the [deep learning book](http://www.deeplearningbook.org/contents/optimization.html)), for the SGD with momentum implementation we mainly need two ingredients:

- A variable $$v$$ that plays the role of velocity
- A friction factor that introduces a gradual (exponential) decay of the velocity term in order to damp the oscillations in the weights updates.

The result it is a faster descent towards the local minima with a dumped oscillation of the weights as in Figure 2.

<img src="{{ site.url }}{{ site.baseurl }}/OptimizerComparison/Momentum.png" alt="SGD with momentum" class="align-center">
<figcaption>Figure 2: SGD with momentum</figcaption>

## SGD with Momentum implementation

Its implementation is pretty straightforward. Please note that given the few data and the simple case we are studying, a unique batch including all the samples has been considered.

* Compute the gradient estimate:
$$ \textbf{g} \leftarrow \frac{1}{m}\nabla_\theta\sum_{i=1}^m L(\textit{f} (\mathbf{x}^{(i)};\mathbf{\theta}),\mathbf{y}^{(i)})$$
* Compute the velocity update:
$$ \textbf{v} \leftarrow \alpha\textbf{v}-\epsilon\textbf{g}$$
* Apply update:
$$ \mathbf{\theta} \leftarrow \mathbf{\theta}+\textbf{v}$$

As specified in the Machine Learning text, "the step size depends on how large and how aligned a sequence of gradients are". If the gradient keeps pointing on the same direction its value contributes to amplify the step on the next iteration, "if the moment algorithm always observe gradient **g**, then it will accelerate in the direction **-g**".

## Comparison with the simple SGD for the Quadratic Regression

We can now finally get into the exercise on the quadratic regression. In the **notebook** linked [here](https://github.com/DavideDaz/TokyoDataScience/blob/master/Assignments/Gradient%20Descent%20Assignment/Basis%20Neural%20Network%20-%20Quadratic%20-%20SGD%20with%20momentum.ipynb) I compared the performance of the simple SGD (using a unique batch of inputs) with the SGD with the momentum applied. You can also open the notebook on Google Colab by right-clicking on the Colab button on top and opening it in a new tab. Two values of friction factor have been also employed to compare the performance.The two implementation have been initialized with the same weight values and run for the same number of epochs for a clear comparison.

<img src="{{ site.url }}{{ site.baseurl }}/OptimizerComparison/momentum_reg.png" alt="Regression" class="align-center">
<figcaption>Figure 3: Regression curves for the three implementations</figcaption>

The three methods reach a reasonable approximation of the quadratic function (Figure 3) but the trend of the loss function and its final value after the 1000 epochs unveil the change introduced by each method. Indeed, after an initial oscillation the loss for the SGD with momentum case is quickly pushed down, reaching the minima after only 100 iterations. The descent for the simple SGD is instead more gradual and ended at stays at a higher value after the end on the epochs Figure 4).

<figure class="half full">
<img src="{{ site.url }}{{ site.baseurl }}/OptimizerComparison/loss_mom.png" alt="SGD with momentum loss">
<img src="{{ site.url }}{{ site.baseurl }}/OptimizerComparison/loss_mom_mag.png" alt="SGD with momentum loss mag">
<figcaption>Figure 4: Loss function over the epochs</figcaption>
</figure>

From Figure 5 we can also observe the effect of the friction factor that in case of 0.9 introduces a 10% decay for each v term, until reaching a steady velocity given by $$\epsilon g/(1-\alpha)$$.

<img src="{{ site.url }}{{ site.baseurl }}/OptimizerComparison/velocities.png" alt="Velocity terms decay">
<figcaption>Figure 5: Velocity terms decay, alpha=0.9</figcaption>


## Case with a friction factor of 0.99

In case we set a higher friction factor we are actually "decreasing the friction" as we are storing a bigger portion of the previous gradient to evaluate the new step and the decay is reduced down to 1% over the iterations (Figure 6).

<img src="{{ site.url }}{{ site.baseurl }}/OptimizerComparison/velocities099.png" alt="Velocity terms decay099">
<figcaption>Figure 6: Velocity terms decay, alpha=0.99</figcaption>

The use of a high value of alpha from the beginning of the cycle can introduce a high oscillation of the loss function. In the initial step the gradient is still far to be aligned towards the local minima, and this error is not damped enough by the friction factor.

<figure class="half full">
<img src="{{ site.url }}{{ site.baseurl }}/OptimizerComparison/loss099.png" alt="SGD with momentum loss099">
<img src="{{ site.url }}{{ site.baseurl }}/OptimizerComparison/loss099_mag.png" alt="SGD with momentum loss mag099mag">
<figcaption>Figure 7: Loss function over the epochs, alpha=0.9,0.99</figcaption>
</figure>
As a result the effect of the excessive amplification is propagated over all the next iterations (Figure 7), resulting some times in a higher loss at the end of the 1000 iterations.


## Tuning the friction factor during the cycle

Reversely, we can leverage a higher friction factor after that the consecutive gradients are sufficiently aligned in order to further speed up the learning process. In the example below I run again the SGD with momentum using a friction factor of 0.9 for the first 200 iterations and switch to 0.99 for the remaining 800 with a very simple modification in the code:

    for i in range(1000):
        if i<=200:
            alpha_friction99 = 0.9
        else:
            alpha_friction99 = 0.99


In Figure 8 is shown the comparison between the case with $$\alpha=0.9$$ and the one with $$\alpha$$ switching from 0.9 to 0.99. We can clearly see how the loss decrease faster after 200 epochs.

<img src="{{ site.url }}{{ site.baseurl }}/OptimizerComparison/loss09-099.png" alt="Friction factor tuning ">
<figcaption>Figure 8: Effect of the friction factor tuning</figcaption>
