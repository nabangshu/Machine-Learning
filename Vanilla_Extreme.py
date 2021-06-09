import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import tensorflow as tf

hidden = 1000
def input_to_hidden(x):
    a = np.maximum(np.dot(x, W_in), 0)
    return a

def predict(x):
    y = np.dot(input_to_hidden(x), Wout)
    return y
		
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#Data preprocessing
#One Hot Encoding
n_classes = len(np.unique(y_train))
y_t = np.zeros((len(y_train),n_classes))
for i in range(len(y_train)):
    y_t[i][int(y_train[i])] = 1

y_train = y_t
y_t = np.zeros((len(y_test),n_classes))
for i in range(len(y_test)):
    y_t[i][int(y_test[i])] = 1
y_test = y_t
## Flattening images
x_t = []
for i in x_train:
    x_t.append(i.flatten())
x_t = np.array(x_t)
x_train = x_t

x_t = []
for i in x_test:
    x_t.append(i.flatten())
x_t = np.array(x_t)
x_test = x_t

# Vanilla ELM algorithm
W_in = np.random.normal(size=[x_train.shape[1], hidden])
X = input_to_hidden(x_train)
Wout = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y_train))

y = predict(x_test)
correct = 0
for i in range(len(y)):
    correct = correct + (1 if np.argmax(y[i]) == np.argmax(y_test[i]) else 0)
print('Accuracy: {:f}'.format(correct/len(y)))