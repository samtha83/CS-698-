import os
from PIL import Image
from PIL import ImageFilter
import numpy as np
import keras
import glob
import sys	
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score
#from sklearn import svmcrossvalidate
from sklearn.model_selection import train_test_split


#os.environ['CUDA_VISIBLE_DEVICES'] = ''
num_classes = 2
#epochs = 50
#batch_size = 4
#img_rows, img_cols = 32, 32
path =sys.argv[1]
#path = "/home/s/sr852/project"
os.chdir(path)
data_dir ='4blocks_200k/imgs'
#data_dir ='4blocks_200k1/imgs'
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
        #img = img.resize((img_rows, img_cols))#resize image into fixed size 32x32x3
        #img = np.array(img)[np.newaxis, :, :, :3]#add new axis and new size is 1x32x32x3
        img = np.array(img)
        data.append(img.flatten())
    for filename in glob.glob(os.path.join(r, '*.txt')):
        #print("this is label:",filename)
        #lines = [int(line.rstrip('\n')) for line in open(filename)]
        #print(lines)
	#lines=np.array
        #lines=[]
        for line in open(filename):
            labels.append(int(line.rstrip('\n')))
        #labels.append(lines)

print("loading ended")

print("size of data, no of features",len(data[0]))
#data_arr = np.asarray(data)
#data_arr = data_arr.flatten()
#labels_arr = np.array(labels).reshape(-1,1)
#data_arr = np.concatenate(data_arr)#concatenate images, shape is 209x128x128x3

#X=data
#y=labels
#split_test_size =0.10

#x_train, x_test, y_train, y_test=train_test_split(X,y, test_size =split_test_size, random_state=45)


x_train = data[1000:]
y_train = labels[1000:]
x_test = data[:1000]
y_test = labels[:1000]

print("number of  x_train:", len(x_train))
print("number of y_train-labels:", len(y_train))
print("number of x_test: ",len(x_test))
print("number of y_test: ",len(y_test))


#------------checking if any missing values-------------------------
x_train_df = pd.DataFrame(x_train)
y_train_df = pd.DataFrame(y_train)

x_test_df =  pd.DataFrame(x_test)
y_test_df = pd.DataFrame(y_test)

print("Missing values in x_train:", x_train_df.isnull().values.any())
print("Missing values in y_train:", y_train_df.isnull().values.any())
print("Missing values in x_test:", x_test_df.isnull().values.any())
print("Missing values in y_test:", y_test_df.isnull().values.any())

#------------------------------------------------------------


print("starting modeling")
c=[.1,1]
for ct in c:
	clf = svm.LinearSVC(C=ct)
#clf = svm.SVC(kernel="poly",degree=2)
#clf = svm.SVC(kernel="linear",degree=2)
#scores = cross_val_score(clf, data, labels, cv=5)
#scores = cross_val_score(clf, x_train, y_train, cv=5)
#print(scores)
#print(sum(scores)/len(scores))
#svmcrossvalidate.getbestC(data, labels)

#bestC = svmcrossvalidate.getbestC(x_train, y_train)

	m=clf.fit(x_train, y_train)
	acc =m.score(x_test, y_test)
	print("for c= ",ct,"acc= ",acc)
