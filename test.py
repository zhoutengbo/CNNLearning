import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import face_input_data

import cv2
import os



class Detector(object):
	def __init__(self, net_factory, data_size, batch_size, model_path):
		graph = tf.Graph()
		with graph.as_default():
			self.image_op = tf.placeholder(tf.float32, shape=[batch_size, data_size, data_size, 3], name='input_image')
			#self.cls_prob, self.bbox_pred, self.landmark_pred = net_factory(self.image_op, training=False)
			self.sess = tf.Session(
				config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
			#saver = tf.train.Saver()
			#model_dict = '/'.join(model_path.split('/')[:-1])
			#ckpt = tf.train.get_checkpoint_state(model_dict)
			#print(model_path)
			#readstate = ckpt and ckpt.model_checkpoint_path
			#assert  readstate, "the params dictionary is not valid"
			#print("restore models' param")
			#saver.restore(self.sess, model_path)
		self.data_size = data_size
		self.batch_size = batch_size

	def predict(self, databatch):
		scores = []
		batch_size = self.batch_size

		minibatch = []
		cur = 0
		n = databatch.shape[0]
		while cur < n:
			print (cur,n,batch_size)
			minibatch.append(databatch[cur:min(cur + batch_size, n), :, :, :])
			cur += batch_size
		
		cls_prob_list = []
		bbox_pred_list = []
		landmark_pred_list = []
		for idx, data in enumerate(minibatch):

			print (idx)
			m = data.shape[0]
			print (m)
			real_size = self.batch_size
			print (real_size)
			if m < batch_size:
				keep_inds = np.arange(m)
				print (keep_inds)
				gap = self.batch_size - m
				while gap >= len(keep_inds):
					gap -= len(keep_inds)
					print ("======>1",keep_inds.shape)
					keep_inds = np.concatenate((keep_inds, keep_inds))
				if gap != 0:
					print ("======>2",keep_inds.shape)
					keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))

				#print (keep_inds.)
				print (data.shape)
				data = data[keep_inds]
				print (data.shape)
				real_size = m







def processed_image(img,scale):
	height, width, channels = img.shape
	new_height = int(height * scale)  # resized new height
	new_width = int(width * scale)  # resized new width

	new_dim = (new_width, new_height)
	img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
	img_resized = (img_resized - 127.5) / 128
	#print (new_width,new_height,new_dim,img_resized)
	return img_resized

def read_single_tfrecord(tfrecord_file, batch_size, net):
	# generate a input queue
	# each epoch shuffle
	filename_queue = tf.train.string_input_producer([tfrecord_file],shuffle=True)
	# read tfrecord
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	image_features = tf.parse_single_example(
		serialized_example,
		features={
			'image/encoded': tf.FixedLenFeature([], tf.string),#one image  one record
			'image/label': tf.FixedLenFeature([], tf.int64),
			'image/roi': tf.FixedLenFeature([4], tf.float32),
			'image/landmark': tf.FixedLenFeature([10],tf.float32)
		}
	)
	if net == 'PNet':
		image_size = 12
	elif net == 'RNet':
		image_size = 24
	else:
		image_size = 48
	image = tf.decode_raw(image_features['image/encoded'], tf.uint8)
	image = tf.reshape(image, [image_size, image_size, 3])
	image = (tf.cast(image, tf.float32)-127.5) / 128

	# image = tf.image.per_image_standardization(image)
	label = tf.cast(image_features['image/label'], tf.float32)
	roi = tf.cast(image_features['image/roi'],tf.float32)
	landmark = tf.cast(image_features['image/landmark'],tf.float32)
	image, label,roi,landmark = tf.train.batch(
		[image, label,roi,landmark],
		batch_size=batch_size,
		num_threads=2,
		capacity=1 * batch_size
	)
	label = tf.reshape(label, [batch_size])
	roi = tf.reshape(roi,[batch_size,4])
	landmark = tf.reshape(landmark,[batch_size,10])
	return image, label, roi,landmark


def prelu(inputs):
    alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs-abs(inputs))*0.5
    return pos + neg

def _activation_summary(x):
    '''
    creates a summary provides histogram of activations
    creates a summary that measures the sparsity of activations

    :param x: Tensor
    :return:
    '''

    tensor_name = x.op.name
    print('load summary for : ',tensor_name)
    tf.summary.histogram(tensor_name + '/activations',x)
    #tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def cls_ohem(cls_prob, label):

	num_keep_radio=0.7
    # 建立一个和label相同shape的全0数组
	zeros = tf.zeros_like(label)
	# 移除非法数据
	label_filter_invalid = tf.where(tf.less(label,0), zeros, label)

	# 包含列表中整个数据
	num_cls_prob = tf.size(cls_prob)

	# 按行重排整张数据表
	cls_prob_reshape = tf.reshape(cls_prob,[num_cls_prob,-1])



	# 类型转换
	label_int = tf.cast(label_filter_invalid,tf.int32)
	# get the number of rows of class_prob
	num_row = tf.to_int32(cls_prob.get_shape()[0])
	#row = [0,2,4.....]
	row = tf.range(num_row)*2

	# 这里算出为人脸概率的下标
	indices_ = row + label_int

	#tf.gather：用一个一维的索引数组，将张量中对应索引的向量提取出来

	label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
	with tf.Session() as sess:
		print("label_prob:%s"%(sess.run(label_prob)))

	#计算标记的概率损失
	loss = -tf.log(label_prob+1e-10)
	with tf.Session() as sess:
		print("loss:%s"%(sess.run(loss)))
	zeros = tf.zeros_like(label_prob, dtype=tf.float32)

	ones = tf.ones_like(label_prob,dtype=tf.float32)

	# set pos and neg to be 1, rest to be 0
	#这里做一次矫正，如果标号为负数，那么置0
	valid_inds = tf.where(label < zeros,zeros,ones)

	#计算数量
	# get the number of POS and NEG examples
	num_valid = tf.reduce_sum(valid_inds)


	keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
	#FILTER OUT PART AND LANDMARK DATA
	#如果有不合法的数据，这里会去掉
	loss = loss * valid_inds
	#返回头部的num_keep_radio% 数据
	loss,_ = tf.nn.top_k(loss, k=keep_num)
	with tf.Session() as sess:
		print("loss===>:%s"%(sess.run(loss)))
	with tf.Session() as sess:
		print(sess.run(tf.gather(cls_prob_reshape, indices_)))
		print(sess.run(label_prob))
		print("Loss:%s"%(sess.run(tf.reduce_mean(loss))))
	return tf.reduce_mean(loss)

def P_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
	with slim.arg_scope([slim.conv2d],
			activation_fn=prelu,
			weights_initializer=slim.xavier_initializer(),
			biases_initializer=tf.zeros_initializer(),
			weights_regularizer=slim.l2_regularizer(0.0005), 
			padding='valid'):
		print(inputs.shape)

#	 conv2d(
#    input,
#    filter,
#    strides,
#    padding,
#    use_cudnn_on_gpu=None,
#    data_format=None,
#    name=None
#	)
		#@input 
		#	[batch_size, in_height, in_width, in_channels]
		#
		#@filter
		#	[filter_height, filter_width, in_channels, out_channels]
		# 	[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]


		#inputs同样是指需要做卷积的输入图像
		#num_outputs指定卷积核的个数（就是filter的个数）
		#kernel_size用于指定卷积核的维度（卷积核的宽度，卷积核的高度）
		#stride为卷积时在图像每一维的步长
		#padding为padding的方式选择，VALID或者SAME

		net = slim.conv2d(inputs, 10, 3, stride=1,scope='conv1')
		_activation_summary(net)
		print(net.shape)
		net = slim.max_pool2d(net, kernel_size=[2,2], stride=2, scope='pool1', padding='SAME')
		_activation_summary(net)
		print(net.shape)
		net = slim.conv2d(net,num_outputs=16,kernel_size=[3,3],stride=1,scope='conv2')
		_activation_summary(net)
		print(net.shape)
		net = slim.conv2d(net,num_outputs=32,kernel_size=[3,3],stride=1,scope='conv3')
		_activation_summary(net)
		print(net.shape)
		conv4_1 = slim.conv2d(net,num_outputs=2,kernel_size=[1,1],stride=1,scope='conv4_1',activation_fn=tf.nn.softmax)
		_activation_summary(conv4_1)
		print(conv4_1.shape)
		
		bbox_pred = slim.conv2d(net,num_outputs=4,kernel_size=[1,1],stride=1,scope='conv4_2',activation_fn=None)
		_activation_summary(bbox_pred)
		print (bbox_pred.shape)

		cls_prob = tf.squeeze(conv4_1,[1,2],name='cls_prob')
		#cls_loss = cls_ohem(cls_prob, label)
		



face_data = face_input_data.read_data_sets()

batch = face_data.my_tran.next_batch(2)
#print(batch[0])
#P_Net(batch[0])
#float_arr = batch[0].astype(np.float32)
#test_arry = tf.zeros( [1,12, 12,3])
#print(float_arr.shape)
#P_Net(test_arry)
batch_size = [2048, 64, 16]

PNet = Detector("P_NET", 12, batch_size[0], "MODULE")


im_resize = processed_image(batch[0].reshape(40,40,1),12/20)

PNet.predict(im_resize.reshape(1,24,24,1))

data=np.array([[1],[34,3],[3,5]])
data1=np.array([4],)
data1 = np.concatenate((data1, data1))
#data = data[data1]
print(data.shape)


#image_batch, label_batch, bbox_batch,landmark_batch = read_single_tfrecord("", 50, 'PNet')


#cls_prob = tf.random_uniform([10,2],0,1,seed = 100)

#with tf.Session() as sess:
#	print("cls_prob:%s"%(sess.run(cls_prob)))

#label = np.array([1,0,0,0,0,0,0,0,0,0])
#cls_ohem(cls_prob, label)


#test=tf.less(label,0)
#with tf.Session() as sess:
#	print("test:%s"%(sess.run(test)))