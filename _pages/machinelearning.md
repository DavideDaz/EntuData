---
layout: single
permalink: /machine-learning/
title: "Machine Learning Projects:"
author_profile: true
header:
  image: "/pictures/rainbridgeResize.jpg"
mathjax: "true"
toc: true
toc_label: "Contents"
toc_icon: "cog"
toc_sticky: true
---

## GTSRB - German Traffic Sign Recognition Benchmark - Multi-class, single-image classification challenge from Kaggle dataset

<img src="{{ site.url }}{{ site.baseurl }}/GTSRB/Header.png" alt="Traffic signs">

The German Traffic Sign Benchmark is a multi-class, single image classification challenge held at the International Joint Conference of Neural Networks (IJCNN) in 2011. The Dataset is composed by:

* More than 40 Classes
* More than 50'000 images in total
* Large, lifelike database

I trained the Resnet50_Wide Architecture on Pytorch to reach an accuracy over 99%. Discover the **notebook** here:
* [GTSRB-RESNET50_WIDE](https://github.com/DavideDaz/TokyoDataScience/blob/master/Machine%20Learning%20Projects/GTSRB/gtsrb_resnet50Wide_WAdam_10%25val.ipynb)
Right Click and open in a new tab on the Colab button to open the project in Google Colab.

Here version of the same architecture that employs an initial normalization of the dataset:
* [GTSRB-RESNET50_WIDE](https://github.com/DavideDaz/TokyoDataScience/blob/master/Machine%20Learning%20Projects/GTSRB/gtsrb_resnet50Wide_WAdam_10%25val_Norm.ipynb)
Right Click and open in a new tab on the Colab button to open the project in Google Colab.

I also modified the Resnet Architecture to obtain a slim version to run faster on Colab. Discover the **notebook** here:
* [GTSRB-RESNET_MODIFIED](https://github.com/DavideDaz/TokyoDataScience/blob/master/Machine%20Learning%20Projects/GTSRB/gtsrb_resnetModified_WAdam_10%25val.ipynb)
Right Click and open in a new tab on the Colab button to open the project in Google Colab.
