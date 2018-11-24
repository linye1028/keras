#20181124
#keras mnist预处理和图片显示函数代码
import matplotlib.pyplot as plt
from keras.datasets import mnist
import pandas as pd
from keras.utils import np_utils
#C:\Users\linye_home\.keras\datasets

#显示image
def plot_image(image):
	fig=plt.gcf()
	fig.set_size_inches(2,2)
	plt.imshow(image,cmap="binary")
	plt.show()

#显示图像，标签，预测结果
def plot_image_labels_prediction(images,labels,prediction,idx,num=10):
	fig=plt.gcf()
	fig.set_size_inches(12,14)
	if num>25:
		num=25
	for i in range(0,num):
		ax=plt.subplot(5,5,1+i) #建立subgraph子图行，5行5列
		ax.imshow(images[idx],cmap='binary')
		title="label="+str(labels[idx])
		if len(prediction)>0 :
			title+=",prediction="+str(prediction[idx])
		ax.set_title(title,fontsize=10)
		ax.set_xticks([])
		ax.set_yticks([])
		idx+=1
	plt.show()
	
(x_train_image,y_train_label),(x_test_image,y_test_label)=mnist.load_data()
x_Train=x_train_image.reshape(60000,784).astype('float32')
x_Test=x_test_image.reshape(10000,784).astype('float32')
x_Train_normalize=x_Train/255
x_Test_normalize=x_Test/255
y_TrainOneHot=np_utils.to_categorical(y_train_label)
y_TestnOneHot=np_utils.to_categorical(y_test_label)
