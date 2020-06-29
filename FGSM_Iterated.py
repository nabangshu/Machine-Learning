import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os

DATA_DIR = '/tmp/data'
NUM_STEPS = 10000
MINIBATCH_SIZE = 100
data = input_data.read_data_sets(DATA_DIR, one_hot=True)

x_train=data.train.images
y_train=data.train.labels
x_test=data.test.images
y_test=data.test.labels
NUM_STEPS = 10000
X = tf.placeholder(tf.float32, [None, 784], name='features')
Y = tf.placeholder(tf.float32, [None, 10], name='labels')
training_epochs = 10000
n_neurons_in_h1 = 30
NN_name = 'attacked'
learning_rate = 0.001
W1 = tf.Variable(tf.truncated_normal([784, n_neurons_in_h1], mean=0, stddev=1 / np.sqrt(2)), name='weights1')
b1 = tf.Variable(tf.truncated_normal([n_neurons_in_h1],mean=0, stddev=1 / np.sqrt(2)), name='biases1')
y1 = tf.nn.relu(tf.add(tf.matmul(X, W1),b1), name='activationLayer1')

Wo = tf.Variable(tf.random_normal([n_neurons_in_h1, 10], mean=0, stddev=1/np.sqrt(2)), name='weightsOut')
bo = tf.Variable(tf.random_normal([10], mean=0, stddev=1/np.sqrt(2)), name='biasesOut')
#activation function(softmax)
logits=tf.add(tf.matmul(y1, Wo),bo,name='logits')
a = tf.nn.softmax(tf.add(tf.matmul(y1, Wo),bo), name='activationOutputLayer')
file_path= './'+ NN_name + '/'
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    #optimizer
train_step = tf.train.GradientDescentOptimizer(0.4).minimize(cross_entropy)
    
correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(Y, 1))
#accuracy determination
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

#FGSM attack
NN_name = 'attacked'
x_adv = X+ 0.01*tf.sign(tf.gradients(cross_entropy,X)[0])
fgsm = X+ 0.1*tf.sign(tf.gradients(cross_entropy,X)[0])
x_adv = tf.clip_by_value(x_adv,-1.0,1.0)
x_adv = tf.stop_gradient(x_adv)

with tf.Session() as sess:
# Train
    sess.run(tf.global_variables_initializer())
    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE, shuffle=False)
        sess.run(train_step, feed_dict={X: batch_xs, Y: batch_ys})
# Test
    ans = sess.run(accuracy, feed_dict={X: data.test.images,Y: data.test.labels})
    y_pred = sess.run(tf.argmax(a, 1), feed_dict={X: x_test})
    #acc= sess.run(accuracy, feed_dict={x: x_adv2,y_true: data.test.labels}) 
    print ("Accuracy for clean examples: {:.4}%".format(ans*100))
    fg=sess.run(fgsm, feed_dict={X: data.test.images,Y:data.test.labels})
    x_adv2=sess.run(x_adv, feed_dict={X: data.test.images,Y:data.test.labels})
		
# Iterative gradient attack
    for count in range(9):
        x_adv3=x_adv2
        x_adv2=sess.run(x_adv, feed_dict={X: x_adv3,Y:data.test.labels})
    y_pred_adv = sess.run(tf.argmax(a, 1), feed_dict={X: x_adv2})
    y_pred_adv2 = sess.run(tf.argmax(a, 1), feed_dict={X: fg})
    acc= sess.run(accuracy, feed_dict={X: x_adv2,Y: data.test.labels})
    acc2= sess.run(accuracy, feed_dict={X: fg,Y: data.test.labels})
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    saver = tf.train.Saver()
    saver.save(sess, file_path+ 'model.checkpoint')
    print('Model saved')
print ("Adversarial Accuracy (Iterated Gradient): {:.4}%".format(acc*100))
print ("Adversarial Accuracy (FGSM): {:.4}%".format(acc2*100))