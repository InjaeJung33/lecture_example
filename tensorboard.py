import tensorflow as tf
import numpy as np

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

global_step =10

#From TF graph, decide which tensors you want to log
with tf.variable_scope('layer1') as scope:
        W1 = tf.Variable(tf.random_normal([2, 10]), name='weight1')
        b1 = tf.Variable(tf.random_normal([10]), name='bias1')
        L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
        L1 = tf.nn.dropout(L1, keep_prob=0.5)

        layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
        tf.summary.histogram("X", X)
        tf.summary.histogram("weights", W1)
        tf.summary.histogram("bias", b1)
        tf.summary.histogram("layer", L1)

with tf.variable_scope('layer2') as scope:
        W2 = tf.Variable(tf.random_normal([10, 10]), name='weight2')
        b2 = tf.Variable(tf.random_normal([10]), name='bias2')
        L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
        L2 = tf.nn.dropout(L2, keep_prob=0.5)
        tf.summary.histogram("weights", W2)
        tf.summary.histogram("bias", b2)
        tf.summary.histogram("layer", L2)

with tf.variable_scope('layer2') as scope:
        W3 = tf.Variable(tf.random_normal([10, 10]), name='weight3')
        b3 = tf.Variable(tf.random_normal([10]), name='bias3')
        L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
        L3 = tf.nn.dropout(L3, keep_prob=0.5)
        tf.summary.histogram("weights", W3)
        tf.summary.histogram("bias", b3)
        tf.summary.histogram("layer", L3)

with tf.variable_scope('layer3') as scope:
    W4 = tf.Variable(tf.random_normal([10, 1]), name='weight4')
    b4 = tf.Variable(tf.random_normal([1]), name='bias4')
    L4 = tf.nn.relu(tf.matmul(L3, W3) + b4)
    L4 = tf.nn.dropout(L4, keep_prob=0.5)
    tf.summary.histogram("weights", W4)
    tf.summary.histogram("bias", b4)
    tf.summary.histogram("layer", L4)
    hypothesis = tf.sigmoid(tf.matmul(L4, W4) + b4)

    # cost/loss function
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

    #Merge all summaries
    summary = tf.summary.merge_all()
#Create writer and add graph
    # initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Create summary writer
    writer = tf.summary.FileWriter("mnist_logs1")
    writer.add_graph(sess.graph)
    feed_dict={X:x_data,Y:y_data}
    s, _ = sess.run([summary, train], feed_dict=feed_dict)
    writer.add_summary(s, global_step=global_step)
    global_step += 1

#tensorboard --logdir=/tmp/mnist_logs
