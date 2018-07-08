import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

session = tf.InteractiveSession()

# 每张图片是28*28像素，转为一个行向量为1*784
x = tf.placeholder(tf.float32, shape=[None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.placeholder(tf.float32, shape=[None, 10])

y_hyp = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hyp), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()

for i in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)
    train_step.run({x: batch_x, y: batch_y})

correct_predicition = tf.equal(tf.argmax(y_hyp, 1), tf.argmax(y, 1))  # 验证准确率

accuracy = tf.reduce_mean(tf.cast(correct_predicition, tf.float32))

# 分别打印训练集、测试集、验证集的准确率
print(accuracy.eval({x: mnist.train.images, y: mnist.train.labels}))
print(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
print(accuracy.eval({x: mnist.validation.images, y: mnist.validation.labels}))
