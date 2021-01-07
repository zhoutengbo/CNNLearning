
'''
	https://zhuanlan.zhihu.com/p/58825924
	该文件生成四种数据
	1、Positive face数据：图片左上右下坐标和label的IOU>0.65的图片
	2、part face数据：图片左上右下坐标和label的0.65>IOU>0.4的图片
	3、negative face 数据：图片左上右下坐标和lable的IOU<0.3的图片
	4、landmark face数据：图片带有landmark label的图片
'''

#原始图片lab
ORIGIN_LAB_DATA_PATH="../Pnet/Data/wider_face_split/wider_face_train_bbx_gt.txt"
#原始图片地址
ORIGIN_IMG_DATA_PATH="../Pnet/Data/WIDER_train/images/"

class pictrueInfo(object):
	def __init__(self):
		self.path = "" 
		self.labSize = -1
		self.labList = []

	def setInfo(self,path,labSize):
		self.path=path
		self.labSize=labSize

	def insertLab(self,x1,y1,w,h):
		labInfo = {}
		labInfo['x1']=x1
		labInfo['y1']=y1
		labInfo['w']=w
		labInfo['h']=h
		self.labList.append(labInfo)

	def clear(self):
		self.path=""
		self.labSize=-1
		self.labList.clear()

	def show(self):
		print("path:",self.path)
		print("labSize:",self.labSize)
		print("labList_size:",len(self.labList))



class DataOrigin(object):
	def __init__(self,lab_file_path,origin_data_path):
		self.lab_file_path = lab_file_path
		self.origin_data_path = origin_data_path


		self.date=self.parseLabFile()
		self.date[8].show()

	#解析原始文件
	def parseLabFile(self):
		reseult=[]
		picture=pictrueInfo()
		num_flat=False
		path=""
		labSize=0
		cnt=0
		with open(self.lab_file_path,"r") as f:
			for line in f.readlines():
				pos=line.find('/')
				start_flat=False

				if(pos>0):
					picture.clear()
					start_flat=True
					cnt=0
					labSize=0

				if(start_flat==False and num_flat==False):
					tmpList=line.split(' ')
					picture.insertLab(tmpList[0],tmpList[1],tmpList[2],tmpList[3])
					cnt=cnt+1

				if(cnt>=labSize and labSize>0):
					reseult.append(picture)

				if(num_flat==True):
					num_flat=False
					labSize=int(line)
					picture.setInfo(path,labSize)

				

				
				if(start_flat == True):
					num_flat=True
					path=line

			return reseult

		#截取
		def cutPcture(self,index):
			pass
				


def generalPositiveFace():
	pass




DataOrigin(ORIGIN_LAB_DATA_PATH,ORIGIN_IMG_DATA_PATH)