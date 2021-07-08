import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.001
num_steps = 500
batch_size = 128
display_step = 10

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

wc1= tf.Variable(tf.truncated_normal([5, 5, 1, 32],stddev=0.1))
bc1= tf.Variable(tf.truncated_normal([32],stddev=0.1))

wc2= tf.Variable(tf.truncated_normal([5, 5, 32, 64],stddev=0.1))

bc2= tf.Variable(tf.truncated_normal([64],stddev=0.1))

wd1= tf.Variable(tf.truncated_normal([7*7*64, 1024],stddev=0.1))

bd1= tf.Variable(tf.truncated_normal([1024],stddev=0.1))

out= tf.Variable(tf.truncated_normal([1024, num_classes],stddev=0.1))

bout= tf.Variable(tf.truncated_normal([num_classes],stddev=0.1))

x = tf.reshape(X, shape=[-1, 28, 28, 1])

conv1 = tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='SAME')
conv1 = tf.nn.bias_add(conv1, bc1)
conv1 = tf.nn.relu(conv1)

conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

conv2 = tf.nn.conv2d(conv1, wc2, strides=[1, 1, 1, 1], padding='SAME')
conv2 = tf.nn.bias_add(conv2, bc2)
conv2 = tf.nn.relu(conv2)

conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

fc1 = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]])
fc1 = tf.add(tf.matmul(fc1, wd1), bd1)
fc1 = tf.nn.relu(fc1)
fc1 = tf.nn.dropout(fc1, dropout)
logits = tf.add(tf.matmul(fc1, out), bout)

prediction = tf.nn.softmax(logits)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#adversarial attack
x_adv = X + 0.1*tf.sign(tf.gradients(loss_op,X)[0])
x_adv = tf.clip_by_value(x_adv,-1.0,1.0)
x_adv = tf.stop_gradient(x_adv)

init = tf.global_variables_initializer()

x_train=mnist.train.images
y_train=mnist.train.labels
x_test=mnist.test.images
y_test=mnist.test.labels
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
        '''
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
    '''
    print("Optimization Finished!")
    x_adv2=sess.run(x_adv, feed_dict={X: mnist.test.images[:256],Y: mnist.test.labels[:256],keep_prob: 1.0})
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images[:256],Y: mnist.test.labels[:256],keep_prob: 1.0}))    
    print("Adversarial Accuracy (FGSM):", \
        sess.run(accuracy, feed_dict={X: x_adv2,
                                      Y: mnist.test.labels[:256],
                                      keep_prob: 1.0}))