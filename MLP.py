import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()  # 默认的session


# 初始化权重
in_units = 784
h1_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])

keep_prob = tf.placeholder(tf.float32)  # dropout参数

# model
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y_hat = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)


# Define loss and optimizer
y = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat), reduction_indices=[1]))
# train_step = tf.train.AdagradOptimizer().minimize(cross_entropy)
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

# Train
tf.global_variables_initializer().run()
for i in range(3000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  train_step.run({x: batch_xs, y: batch_ys, keep_prob: 0.75})


# Test
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
