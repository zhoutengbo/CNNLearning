
#人脸检测负样本生成
from PIL import Image
import os
#一个rec—width上滑动的步数
w_step=5
h_step=5

rec_width_list=[60,100,150,200]
rec_height_list=[70,120,180,240]

#裁剪图像并保存
def crop_and_save(rec_width,rec_height,k):
    w_step_num=round((width-rec_width)//(rec_width//w_step))
    h_step_num=round((height-rec_height)//(rec_height//h_step))
    y1=0
    for i in range(h_step_num):
    	x1=0
    	i=i+1
    	for j in range(w_step_num):
    		j=j+1
    		k=k+1
    		img.crop((x1,y1,x1+rec_width,y1+rec_height)).resize((50,50)).save(r'./Data/no_face_new/{}.jpg'.format(k))
    		x1=x1+rec_width//w_step
    	y1=y1+rec_height//h_step
    return k

m=0
n=0
for dir,folder,file in os.walk(r'./Data/Tran/'):
	for i in file:
		m=m+1
#输出当前正在裁剪的图片名及已裁剪张数
		print(i)
		print(m)
		try:
			img = Image.open(os.path.join(r'./Data/Tran/',i))
#        转为灰度图
			img=img.convert('L')
			height=img.size[1]
			width=img.size[0]
			for i in range(len(rec_width_list)):
				rec_width=rec_width_list[i]
				rec_height=rec_height_list[i]
				n=crop_and_save(rec_width,rec_height,n)
		except IOError:
			print ("bad image")
		except:
			print ("bad image")
