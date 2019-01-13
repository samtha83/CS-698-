
import os
from PIL import Image
from PIL import ImageFilter
import numpy as np
import keras
import glob
import sys
	
import pandas as pd
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D,GlobalAveragePooling2D
from keras.models import Sequential
from keras.optimizers import SGD

#os.environ['CUDA_VISIBLE_DEVICES'] = ''
num_classes = 2
epochs = 50
batch_size = 5
img_rows, img_cols = 32, 32

#os.chdir("/home/s/sr852/project")
#path = sys.argv[1]
path = "/home/s/sr852/project"
os.chdir(path)
data_dir ='4blocks_200k1/imgs'

data = []
labels =[]
import re
numbers = re.compile(r'(\d+)')
print("loading started")
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


for r, d, files in os.walk(data_dir):
    #print("r:",r)
    #print("d:",d)
    #print("files:", files)
    for filename in sorted(glob.glob(os.path.join(r, '*.png')),key=numericalSort):
        #print(filename)
        img = Image.open(filename)#shape is 256 X 256 X 3
        #print("file opened")
        img = img.resize((img_rows, img_cols))#resize image into fixed size 32x32x3
        img = np.array(img)[np.newaxis, :, :, :3]#add new axis and new size is 1x32x32x3
        data.append(img)
    for filename in glob.glob(os.path.join(r, '*.txt')):
        #print("this is label:",filename)
        lines = [line.rstrip('\n') for line in open(filename)]
        #print(lines)
        labels.append(lines)

print("loading ended")
#df = pd.read_csv('block_labels.csv')#load images' names and labels
#names = df['file'].values
#labels = df['label'].values.reshape(-1,1)# converting to 1-D Matix

#data = []

data_arr = np.asarray(data)
labels_arr = np.array(labels).reshape(-1,1)
data_arr = np.concatenate(data_arr)#concatenate images, shape is 209x128x128x3

x_train = data_arr[10:].astype(np.float32)
y_train = labels_arr[10:]
x_test = data_arr[:10].astype(np.float32)# set the last 32 images as test dataset
y_test = labels_arr[:10]
x_train /= 255 #fit the pixel intensity to range between 0 and 1
x_test /= 255

print("rows of x_train:",len(x_train))
print("rows of y_train:",len(x_train))
print("rows of x_test:",len(x_test))
print("rows of y_test:",len(y_test))

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)#this is one hot encoding an prevents misleading that labels are ordered number

model = Sequential()


model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', input_shape=[32,32,3])) #Convolution

model.add(Activation('relu')) #Activation function

#model.add(AveragePooling2D(pool_size=(2, 2))) #2x2 average pooling

#model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='valid'))

model.add(Activation('relu')) #Activation function

#model.add(GlobalAveragePooling2D())

#model.add(AveragePooling2D(pool_size=(2,2)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) #shape equals to [batch_size, 32] 32 is the number of filters # converts 2-D array to 1-D for classification

model.add(Dense(256,activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(num_classes,activation ='softmax')) #Fully connected layer

#model.add(Activation('softmax'))

#opt = keras.optimizers.rmsprop(lr=0.01, decay=1e-6)
opt='adam'

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


def train():

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

train()





