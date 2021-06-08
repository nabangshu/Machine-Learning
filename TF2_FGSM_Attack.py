import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

from tensorflow.keras.layers import Flatten, Dense, Dropout
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=5)

model.evaluate(x_test, y_test, verbose=2)

loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
labels = tf.one_hot(y_test,10)
loss(model(x_test),labels)

x=tf.convert_to_tensor(x_test)
labels = tf.one_hot(y_test,10)
with tf.GradientTape() as tape:
    tape.watch(x)
    prediction = model(x)
    loss = loss(labels, prediction)
grad=tape.gradient(loss, x)

adv_x = x + 0.05*tf.sign(grad)
model.evaluate(adv_x, y_test, verbose=2)