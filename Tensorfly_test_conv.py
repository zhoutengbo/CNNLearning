import tensorflow as tf
from tensorflow.python.client import device_lib
import input_data
import os

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


with tf.device('/gpu:0'):
	x = tf.placeholder("float", [None, 784])


	#卷积在每个5x5的patch中算出32个特征。卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。 而对于每一个输出通道都有一个对应的偏置量
	#[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
	#strides****为卷积时在图像每一维的步长，这是一个一维的向量，长度为4，对应的是在input的4个维度上的步长
	#adding****是****string****类型的变量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式，SAME代表卷积核可以停留图像边缘，VALID表示不能
	W_conv1 = weight_variable([5, 5, 1, 32])

	b_conv1 = bias_variable([32])

	#x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)
	#[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
	x_image = tf.reshape(x, [-1,28,28,1])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)


	print('tf.shape(h_conv1):',h_conv1.shape)

	h_pool1 = max_pool_2x2(h_conv1)

	print('tf.shape(h_pool1):',h_pool1.shape)


	W_conv2 = weight_variable([5, 5, 32, 64])

	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)


	h_pool2 = max_pool_2x2(h_conv2)

	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

	print('tf.shape(x1):',h_conv2.shape)

	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder("float")

	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([1024, 10])

	b_fc2 = bias_variable([10])

	y_ = tf.placeholder("float", [None,10])

	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

	cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	init = tf.global_variables_initializer()

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
wirter = tf.summary.FileWriter('/path/to/log-directory',sess.graph)

sess.run(init)

for i in range(20000):
	batch = mnist.train.next_batch(50)
	if i%100 == 0:
		train_accuracy = accuracy.eval(session=sess,feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
		print (batch[1])
		print ("step %d, training accuracy %g"%(i, train_accuracy))
	train_step.run(session=sess,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
	

print ("test accuracy %g"%accuracy.eval(session=sess,feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))