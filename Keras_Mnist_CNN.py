#数据预处理用
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
from keras.utils import np_utils
#建模用
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
#画图用
import matplotlib.pyplot as plt
from keras.models import load_model
np.random.seed(10)

def show_train_history(train_history,train,validation):
	plt.plot(train_history.history[train])
	plt.plot(train_history.history[validation])
	plt.title('Train History')
	plt.ylabel(train)
	plt.xlabel('Epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()

def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx], cmap='binary')
        title= "label=" +str(labels[idx])
        if len(prediction)>0:
            title+=",predict="+str(prediction[idx]) 
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()	
	
#提取mnist并转化成四位数组
(x_train,y_train),(x_test,y_test)= mnist.load_data()
x_train4D=x_train.reshape(x_train.shape[0],28,28,1).astype('float32')
x_test4D=x_test.reshape(x_test.shape[0],28,28,1).astype('float32')
#归一化
x_train4D_normalize=x_train4D/255
x_test4D_normalize=x_test4D/255
#label on-hot encoding
y_train_OneHot = np_utils.to_categorical(y_train) 
y_test_OneHot = np_utils.to_categorical(y_test)

#建模
model=Sequential()
#建立卷积层1
model.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',input_shape=(28,28,1),activation='relu'))
#建立池化层1
model.add(MaxPooling2D(pool_size=(2,2)))
#建立卷积层2
model.add(Conv2D(filters=36,kernel_size=(5,5),padding='same',activation='relu'))
#建立池化层2
model.add(MaxPooling2D(pool_size=(2,2)))
#加入Dropout 每次训练迭代随机放弃25%神经元，避免过拟合
model.add(Dropout(0.25))
#建立平坦层
model.add(Flatten())
#建立隐藏层
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
#建立输出层
model.add(Dense(10,activation='softmax'))
#查看CNN摘要
print(model.summary())

#进行训练
#定义训练方式
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#开始训练
train_history=model.fit(x=x_train4D_normalize,y=y_train_OneHot,validation_split=0.2,epochs=10,batch_size=300,verbose=2)
#画出训练过程
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')


model.save("mnist_keras_cnn_model.h5") #保存模型
del model #删除原模型

model1 = load_model("mnist_keras_cnn_model.h5") #加载模型
prediction=model1.predict_classes(x_test4D_normalize)

plot_images_labels_prediction(x_test,y_test,prediction,idx=100,num=25)