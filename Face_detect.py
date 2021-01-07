import tensorflow as tf
from tensorflow.python.client import device_lib
import face_input_data

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.001)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.001, shape=shape)
	return tf.Variable(initial)


face_data = face_input_data.read_data_sets()


with tf.device('/gpu:0'):
	x = tf.placeholder("float", [None,40, 40,1],name='xs')

	W_conv1 = weight_variable([5, 5, 1, 32])

	b_conv1 = bias_variable([32])

	#x_image = x#tf.reshape(x, [-1,128,128,1])

	h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

	h_pool1 = max_pool_2x2(h_conv1)

	W_conv2 = weight_variable([5, 5, 32, 64])

	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

	h_pool2 = max_pool_2x2(h_conv2)

	W_fc1 = weight_variable([10 * 10 * 64, 1024])

	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 10*10*64])

	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder("float")

	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([1024, 2])

	b_fc2 = bias_variable([2])

	y_ = tf.placeholder("float", [None,2],name='ys')

	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2,name='output')

	cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

	train_step = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.5).minimize(cross_entropy)

#	grads = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.5).compute_gradients(cross_entropy)
#	for i, (g, v) in enumerate(grads):
#		if g is not None:
#			grads[i]=(tf.clip_by_norm(g, 5), v)
#	train_op = train_step.apply_gradients(grads)

	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))


	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"),name='accuracy')

	init = tf.global_variables_initializer()

	
saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
wirter = tf.summary.FileWriter('/path/to/log-directory',sess.graph)

sess.run(init)
flat = 0

for i in range(1000):
	#if flat == 0:
	#	batch = face_data.train.next_batch(50)
	#	flat = 1
	#else:
	batch = face_data.dataSetRandom.next_batch(50)
	#	flat = 0
	#print (batch[1])
	#print (i)
	if i%100 == 0:
		test_batch=face_data.test.next_batch(50)
		train_accuracy = accuracy.eval(session=sess,feed_dict={x:test_batch[0], y_: test_batch[1], keep_prob: 1.0})

		tt = tf.argmax(test_batch[1],1)
		print (sess.run(y_conv,feed_dict={x:test_batch[0], y_: test_batch[1], keep_prob: 1.0}))
		print ("step %d, training accuracy %g"%(i, train_accuracy))

	train_step.run(session=sess,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


save_path = saver.save(sess, "./model/model.ckpt")
print ("Model saved in file: ", save_path)
