#Name : Apoorva Arya
#Roll number: 2020032 

from google.colab import drive
drive.mount("/content/drive")
! cp drive/MyDrive/SML/cifar-10-python.tar.gz .
! cp drive/MyDrive/SML/fminst.zip .
! cp drive/MyDrive/SML/mnist.zip .
! tar -xf cifar-10-python.tar.gz 
! unzip fminst.zip -d fminst/
! unzip mnist.zip -d mnist/
#---------------------------------

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

#Answer1

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def LDA(x, y, test):
    X = np.array(x)
    Y = np.array(y)
    clf = LinearDiscriminantAnalysis()
    clf.fit(X,Y)
    return clf.predict(test)


def getAccuracy(original, toCompare):
    matches = 0
    for i in range(len(original)):
        if (original[i] == toCompare[i]):
            matches += 1
    return (matches/len(original))

def getClassWiseAccuracy(original, toCompare, param):
    actualParam = 0
    matchParam = 0
    for i in range(len(original)):
        if(original[i] == param):
            actualParam += 1
            if(toCompare[i] == param):
                matchParam += 1
    return matchParam/actualParam


batch_metadata = unpickle("cifar-10-batches-py/batches.meta")
batch_1 = unpickle("cifar-10-batches-py/data_batch_1")
batch_2 = unpickle("cifar-10-batches-py/data_batch_2")
batch_3 = unpickle("cifar-10-batches-py/data_batch_3")
batch_4 = unpickle("cifar-10-batches-py/data_batch_4")
batch_5 = unpickle("cifar-10-batches-py/data_batch_5")
test_batch = unpickle("cifar-10-batches-py/test_batch")
train_x = []
train_x.extend(batch_1[b'data'])
train_x.extend(batch_2[b'data'])
train_x.extend(batch_3[b'data'])
train_x.extend(batch_4[b'data'])
train_x.extend(batch_5[b'data'])
train_x = np.array(train_x)
train_x = train_x.reshape(train_x.shape[0],3,32,32)
train_y = []
train_y.extend(batch_1[b'labels'])
train_y.extend(batch_2[b'labels'])
train_y.extend(batch_3[b'labels'])
train_y.extend(batch_4[b'labels'])
train_y.extend(batch_5[b'labels'])
train_y = np.array(train_y)
#-------------------------------------------------------------------

print(train_x.shape,train_y.shape)
test_x = test_batch[b'data']
test_y = test_batch[b"labels"]
test_y = np.array(test_y)

train_x = train_x.reshape(50000,3*32*32)
test_x = test_x.reshape(10000,3*32*32)
resLDA = LDA(train_x, train_y, test_x)
#--------------------------------------------------------------------

print(getAccuracy(test_y, resLDA))
print("Class-Wise accuracies:- ")
for i in range(10):
    print(i, end=" - ")
    print(getClassWiseAccuracy(test_y, resLDA, i))

def visualization(class_num):
  x_reshaped = train_x.reshape(train_x.shape[0], 3, 32, 32)

  L = []
  x = len(x_reshaped)
  for i in range(x):
    if(train_y[i]==class_num):
      L.append(x_reshaped[i])

    if len(L)==5:
      return L

def plotImg(Lists):
  for i in range(5):
    plt.imshow(np.transpose(Lists[i], (1,2,0)))
    plt.figure()

for i in range(10):
  lists = visualization(i)
  plotImg(lists)

#-------------------------------------------------------------------#
#-------------------------------------------------------------------#
#-------------------------------------------------------------------#

#Answer2

import gzip
def images_file_read(file_name):
    with gzip.open(file_name, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count, column_count))
        return images

def labels_file_read(file_name):
    with gzip.open(file_name, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels

def func_PCA(n_components, data):
    pca = PCA(n_components)
    X_pca = pca.fit_transform(data)
    return pca, X_pca

train_x1 = images_file_read("mnist/mnist/train-images-idx3-ubyte.gz")
print(train_x1.shape)
train_y1 = labels_file_read("mnist/mnist/train-labels-idx1-ubyte.gz")
test_x1 = images_file_read("mnist/mnist/t10k-images-idx3-ubyte.gz")
print(test_x1.shape)
test_y1 = labels_file_read("mnist/mnist/t10k-labels-idx1-ubyte.gz")

train_x1 = train_x1.reshape(60000,28*28)
test_x1 = test_x1.reshape(10000,28*28)
resPCA, resPCA_t  = func_PCA(15, train_x1)
test_x_t = resPCA.transform(test_x1)
resLDA1 = LDA(resPCA_t, train_y1, test_x_t)
print(getAccuracy(test_y1, resLDA1))
print("Class-Wise accuracies:- ")
for i in range(10):
    print(i, end=" - ")
    print(getClassWiseAccuracy(test_y1, resLDA1, i))

#-------------------------------------------------------------------

resPCA, resPCA_t  = func_PCA(8, train_x1)
test_x_t = resPCA.transform(test_x1)
resLDA1 = LDA(resPCA_t, train_y1, test_x_t)
print(getAccuracy(test_y1, resLDA1))
print("Class-Wise accuracies:- ")
for i in range(10):
    print(i, end=" - ")
    print(getClassWiseAccuracy(test_y1, resLDA1, i))

#-------------------------------------------------------------------

resPCA, resPCA_t  = func_PCA(3, train_x1)
test_x_t = resPCA.transform(test_x1)
resLDA1 = LDA(resPCA_t, train_y1, test_x_t)
print(getAccuracy(test_y1, resLDA1))
print("Class-Wise accuracies:- ")
for i in range(10):
    print(i, end=" - ")
    print(getClassWiseAccuracy(test_y1, resLDA1, i))

#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------

#Answer3
import pandas as pd
df_train = pd.read_csv("fminst/fashion-mnist_train.csv")
df_test = pd.read_csv("fminst/fashion-mnist_test.csv")
label = {0:"T-shirt/top",
1 :"Trouser",
2 :"Pullover",
3 :"Dress",
4 :"Coat",
5 :"Sandal",
6 :"Shirt",
7 :"Sneaker",
8 :"Bag",
9 :"Ankle boot"}

train_x2 = np.array(df_train.iloc[:,1:]).reshape(df_train.shape[0],28,28)
train_y2 = np.array(df_train.iloc[:,0])
test_x2 = np.array(df_test.iloc[:,1:]).reshape(df_test.shape[0],28,28)
test_y2 = np.array(df_test.iloc[:,0])
print(train_x2.shape,train_y2.shape)
print(test_x2.shape,test_y2.shape)
train_x2 = train_x2.reshape(60000,28*28)
test_x2 = test_x2.reshape(10000,28*28)

#arr = train_x, arr1 = train_y
def Mean(arr):
  mean = []
  arr = arr.T
  for i in range(len(arr)):
        value = np.array(arr[i])
        mean.append(value.mean())
  mean = np.asarray(mean)
  return mean

def classwiseMean(arr, arr1):
  mean_vectors = []
  cl = np.unique(arr1)
  for c in cl: 
    mean_vectors.append(np.mean(arr[arr1==c], axis= 0))
  return mean_vectors

def FDA(arr, arr1, vals):
    mean = Mean(arr)
    cmean = classwiseMean(arr, arr1)
    sb = np.zeros((arr.shape[1], arr.shape[1]))
    
    for cl, cm in enumerate(cmean):
        s = arr[arr1==cl]
        size = s.shape[0]
        
        cm = cm.reshape(1, arr.shape[1])
        centr = cm-mean
        sb += size*np.dot(centr.T, centr)
   
    sw = []
    for cl, cm in enumerate(cmean):
        si = np.zeros((arr.shape[1], arr.shape[1]))
        s = arr[arr1==cl]
        for s in s:
            temp = s-mean
            temp = temp.reshape(1, arr.shape[1])
            si += np.dot(temp.T, temp)
        sw.append(si)
        
    S = np.zeros((arr.shape[1], arr.shape[1]))
    for si in sw:
        S += si
    
    np.fill_diagonal(S, S.diagonal()+0.0001)
    SI = np.linalg.inv(S)
    W = SI.dot(sb)
    
    eigen_values, temp, eigen_vectors = np.linalg.svd(W)

    eigen_values = eigen_values[-vals:]
    eigen_vectors = eigen_vectors[:,-vals:]
    return eigen_vectors

W = FDA(train_x2, train_y2, 9)

#-------------------------------------------------------------------

print(W.shape)
print(test_x2.shape)
Y = np.zeros((60000, 9))
temp = W.T

for i in range(len(train_x2)):
  Y[i] = (temp.dot(train_x2[i]))

print(Y.shape)
Test_x2 = np.zeros((10000, 9))
for i in range(10000):
  Test_x2[i] = temp.dot(test_x2[i])

print(train_y2.shape)
resLDA2 = LDA(Y, train_y2, Test_x2)

print(getAccuracy(test_y2, resLDA2)*100)
print("Class-Wise accuracies:- ")
for i in range(10):
    print(i, end=" - ")
    print(getClassWiseAccuracy(test_y2, resLDA2, i)*100)

#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------

#Answer4

#Best value of Q2 --> 15, since it gave the max. accuracy
resPCA, resPCA_t  = func_PCA(15, train_x1)
test_x_t = resPCA.transform(test_x1)

#Applying FDA
resFDA1 = FDA(resPCA_t, train_y1, 9)
temp2 = resFDA1.T

Y1 = np.zeros((60000, 9))
for i in range(len(resPCA_t)):
  Y1 = (temp2.dot(resPCA_t[i]))

Test_x_t = np.zeros((10000, 9))
for i in range(10000):
  Test_x_t[i] = temp2.dot(test_x_t[i])

#Applying LDA
resLDA3 = LDA(Y1, train_y1, Test_x_t)

print(getAccuracy(test_y1, resLDA3)*100)
print("Class-Wise accuracies:- ")
for i in range(10):
    print(i, end=" - ")
    print(getClassWiseAccuracy(test_y1, resLDA3, i)*100)