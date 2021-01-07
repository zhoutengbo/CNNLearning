import os
import cv2
import numpy
import random

TEST_DATA="./Data/Test"
TEST_OUT_DATA_PATH="./Data/Test/tmp"

TRAN_DATA="./Data/Tran"
TRAN_OUT_DATA_PATH="./Data/Tran/tmp"

TEST_NO_FACE_DATA="./Data/no_face"
TEST_NO_FACE_DATA_PATH="./Data/no_face/tmp"

MY_TEST_DATA="./Data/my_tran"
MY_TEST_DATA_PATH="./Data/my_tran/tmp"

def init_images(filename):
  print('Extracting', filename)
  files=os.popen('ls -l '+filename+" |awk '{print $9}'").read()
  images = files.split('\n')
  return images

def handle_images(images,int_dir,out_dir):
	if os.path.exists(out_dir) == False:
		os.makedirs(out_dir)
	result_images=[]
	print (type(images))
	for x in images:
		file_out=out_dir+"/"+x
		if os.path.exists(file_out) == False:
			file_in=int_dir+"/"+x
			try:
				img = cv2.imread(file_in)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				img_new = cv2.resize(img, (40, 40))
				dst = cv2.equalizeHist(img_new)
				cv2.imwrite(file_out,dst)
				result_images.append(x)
			except IOError:
				print ("bad image",x)
			except:
				print ("bad image",x)
		else:
			result_images.append(x)

	return result_images
class DataSet(object):
	def __init__(self,int_dir,out_dir,is_negtive=False):
		images=init_images(int_dir)
		self._images=handle_images(images,int_dir,out_dir)
		self._num_examples=len(self._images)
		self._labels = numpy.zeros((len(self._images),2)) 
		self._int_dir=int_dir
		self._out_dir=out_dir
		self._idx=0
		self._epochs_completed = 0

		if is_negtive == False :
			self._labels[:,0] = 1
		else:
			self._labels[:,1] = 1
	def next_batch(self, batch_size):
		start = self._idx
		self._idx += batch_size
		if self._idx > self._num_examples:
			self._epochs_completed += 1
			start = 0
			self._idx = batch_size
			assert batch_size <= self._num_examples

		end = self._idx
		out_result_images=[]
		out_result_labels=[]
		count=0
		for x in self._images[start:end]:
			file_in=self._out_dir+"/"+x
			img = cv2.imread(file_in)
			if img is not None:
				img = cv2.split(img)[0]
				out_result_images.append(numpy.array(img).reshape(40,40,1))
				#print (self._labels[count,:])
				out_result_labels.append(self._labels[count,:])
				count=count+1
		return numpy.array(out_result_images), numpy.reshape(numpy.array(out_result_labels),(count,2))
class DateSetRandom(object):
	def __init__(self,train,noface):
		self.train = train
		self.noface = noface

	def next_batch(self,batch_size):
		out_result_images=[]
		out_result_labels=[]
		flat = 0
		count=0
		for x in range(1,batch_size):
			if flat == 0:
				flat = 1
				idx=random.randint(1, self.train._num_examples)
				file_in=self.train._out_dir+"/"+self.train._images[idx]
				img = cv2.imread(file_in)
				if img is not None:
					img = cv2.split(img)[0]
					out_result_images.append(numpy.array(img).reshape(40,40,1))
					out_result_labels.append(self.train._labels[idx])
					count=count+1
				#self.train.
			else:
				flat = 0
				idx=random.randint(1, self.noface._num_examples)
				file_in=self.noface._out_dir+"/"+self.noface._images[idx]
				img = cv2.imread(file_in)
				if img is not None:
					img = cv2.split(img)[0]
					out_result_images.append(numpy.array(img).reshape(40,40,1))
					out_result_labels.append(self.noface._labels[idx])
					count=count+1
		#print ("out")
		return numpy.array(out_result_images), numpy.reshape(numpy.array(out_result_labels),(count,2))
			

def read_data_sets():
	class DataSets(object):
		pass
	data_sets = DataSets()

	data_sets.train = DataSet(TEST_DATA,TEST_OUT_DATA_PATH)
	data_sets.test = DataSet(TRAN_DATA,TRAN_OUT_DATA_PATH)
	data_sets.noface = DataSet(TEST_NO_FACE_DATA,TEST_NO_FACE_DATA_PATH,True)
	data_sets.dataSetRandom=DateSetRandom(data_sets.train,data_sets.noface)
	data_sets.my_tran= DataSet(MY_TEST_DATA,MY_TEST_DATA_PATH)
	return data_sets

#DataSet(TEST_DATA,TEST_OUT_DATA_PATH)
#DataSet(TRAN_DATA,TRAN_OUT_DATA_PATH)
#print (random.randint(12, 20))
