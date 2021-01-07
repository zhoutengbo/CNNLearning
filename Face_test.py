import tensorflow as tf
from tensorflow.python.client import device_lib
import face_input_data
import numpy
import cv2
from PIL import Image
from scipy import misc

#sess = tf.Session()

#print ("Model restored.")
#with tf.device('/gpu:0'):
#	saver = tf.train.import_meta_graph('./model/model.ckpt.meta')
#saver.restore(sess, tf.train.latest_checkpoint('./model'))

#graph = tf.get_default_graph()

#xs = graph.get_tensor_by_name('input/xs:0')

#ys = graph.get_tensor_by_name('input/ys:0')




meta_path='./model/model.ckpt.meta'
model_path='./model/model.ckpt'

saver = tf.train.import_meta_graph(meta_path) 
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
	saver.restore(sess, model_path)
	graph = tf.get_default_graph()
	xs = graph.get_tensor_by_name('xs:0')
	ys = graph.get_tensor_by_name('ys:0')
	out = graph.get_tensor_by_name('output:0')
	accuracy = graph.get_tensor_by_name('accuracy:0')
	keep_prob= graph.get_tensor_by_name('Placeholder:0')
	face_data = face_input_data.read_data_sets()
	batch = face_data.my_tran.next_batch(6)
	train_accuracy = accuracy.eval(session=sess,feed_dict={xs:batch[0], ys: batch[1], keep_prob: 1.0})
	tt = tf.argmax(batch[1],1)
	print (sess.run(accuracy,feed_dict={xs:batch[0], ys: batch[1], keep_prob: 1.0}))
	print ("training accuracy %g"%(train_accuracy))
	#im = Image.fromarray(list(batch[0][0]))
	#im.save("test_result.png")

	print(sess.run(out,feed_dict={xs:batch[0],keep_prob:1}))
	


#face_data = face_input_data.read_data_sets()
#batch = face_data.test.next_batch(2)

#print (batch[0].shape)

#x_image = tf.reshape(batch[0], [-1,128,128,1])

#print (x_image)



#A = numpy.zeros((2,5))
#A[0] = 1
#print (A,"         ==")

#B=numpy.reshape(A,(2,5,1))
#print(B[1][1][0])



#B = numpy.resize(A,(3,5))

#print (A[:,0])
#print (B)







