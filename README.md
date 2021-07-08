# Machine-Learning

This repository contains some of my code in Machine Learning.

## Course Assignments
The .cpp files were assignments in my Machine Learning course. I implemented these in C++ for obtaining higher speed.
- Linear_Reg_Newton.cpp contains the code for simple linear regression, where I implemented everything from scratch including the transpose and matrix multiplication function. I computed the inverse using Cholesky Decomposition.
- MNIST_Naive_Bayes.cpp contains the code for Naive Bayes classifier on MNIST dataset. I also wrote the code for importing the dataset in C++.
- Online_Learning.cpp contains the code for online learning.

# Deep Learning
Iterated_FGSM.py and TF2_FGSM.py contain the FGSM and iterated FGSM adversarial attacks on a simple MLP in Tensorflow 1.0 and Tensorflow 2.0, respectively.
### Dependencies for Deep Learning
``` python >= 3.0, Tensorflow >= 1.0 (for Iterated_FGSM), and Tensorflow >= 2.0 (for TF2_FGSM) ```

Conv_FGSM.py contains the implementation of a basic convolutional neural network (CNN) and an FGSM attack on that network. It drops the accuracy of the CNN from ~99% to ~55%.

# Extreme Learning Machine
Vanilla_Extreme contains the code for the vanilla version of Extreme Learning Machines. It uses random input layer weights, and can quickly learn output layer weights using linear regression.
I am currently working on using Genetic Algorithm to learn the input layer weights. Coming soon.

### Dependencies for Extreme Learning
``` python >= 3.0```
