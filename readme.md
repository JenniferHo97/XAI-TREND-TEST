# Trend-based Faithfulness Test

This repo includes the implementation of the local explanation evaluation framework and extensive experiments. The frameworks integrates ten popular explanation methods and conducts faithfulness assessment on various tasks, including image classification, sentiment classification, and vulnerability detection.

## Table of Contents

- [Trend-based Faithfulness Test](#trend-based-faithfulness-test)
  - [Table of Contents](#table-of-contents)
  - [Dependency](#dependency)
  - [Parameters](#parameters)
  - [Getting Started](#getting-started)

## Dependency

We use `python3.7` to run this code with some necessary packages:

```text
captum==0.4.0
pytorch==1.9.0+cu111
torchtext==0.10.0
torchvision==0.10.0+cu111
gensim==4.1.2
numpy==1.20.3
```

## Parameters

We use default parameters for most explanation methods, some custom parameters are as follows.

```text
SG, SG-SQ, VG, SG-SQ-IG: stdevs=0.2
Occlusion: sliding_window_shapes=(1, 3, 3) (MNIST), (3, 3, 3) (CIFAR-10), (3, 6, 6) (Tiny-ImageNet)
LIME, KS: n_samples=500
LIME: n_segment=70(MNIST, CIFAR-10),100(Tiny-ImageNet)
```

Detailed parameters are provided in the source code.

## Getting Started

Running EMBT on MNIST: First, open the MNIST folder. Then fill out the paths to the datasets and models. Run the following files.

```text
cd mnist
python train_test_mnist.py
python train_test_backdoor_mnist.py
python exp_EMBT.py
```

## Full Version with Appendix

<https://arxiv.org/abs/2309.05679>
