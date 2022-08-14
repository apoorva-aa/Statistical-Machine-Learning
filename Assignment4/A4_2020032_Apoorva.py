#Name : Apoorva Arya
#Roll number: 2020032 

from google.colab import drive
drive.mount("/content/drive")
! cp drive/MyDrive/SML/fminst.zip .
! cp drive/MyDrive/SML/mnist.zip .
! unzip fminst.zip -d fminst/
! unzip mnist.zip -d mnist/

import numpy as np
import gzip
import sys
import pickle as cPickle
from keras.datasets import mnist
from keras.layers import Dropout
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping

from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from keras import initializers
from tensorflow.keras import optimizers
from matplotlib import pyplot
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from tensorflow.keras.layers import BatchNormalization
from array import array

#Answer 1

from sklearn import tree
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

def getAccuracy(original, toCompare):
    matches = 0
    for i in range(len(original)):
        if (original[i] == toCompare[i]):
            matches += 1
    return (matches/len(original))

train_x1 = images_file_read("mnist/mnist/train-images-idx3-ubyte.gz")
train_y1 = labels_file_read("mnist/mnist/train-labels-idx1-ubyte.gz")
test_x1 = images_file_read("mnist/mnist/t10k-images-idx3-ubyte.gz")
test_y1 = labels_file_read("mnist/mnist/t10k-labels-idx1-ubyte.gz")

learning_rate = 0.1
final_pred=0
train_x1_f = train_x1
test_x1_f = test_x1
train_x1_f = train_x1_f.reshape(60000,28*28)
test_x1_f = test_x1_f.reshape(10000,28*28)
#iteration 1
clf = tree.DecisionTreeRegressor(max_depth=1)
clf.fit(train_x1_f, train_y1)
y_hat = clf.predict(train_x1_f)
final_pred += y_hat
r1 = train_y1 - y_hat
final_pred += r1*(learning_rate)
#iteration 2
clf.fit(train_x1_f, r1)
y_hat_1 = clf.predict(train_x1_f)
r2 = r1 - y_hat_1
final_pred += r2*(learning_rate)
#iteration 3
clf.fit(train_x1_f, r2)
y_hat_2 = clf.predict(train_x1_f)
r3 = r2 - y_hat_2
final_pred += r3*(learning_rate)
#iteration 4
clf.fit(train_x1_f, r3)
y_hat_3 = clf.predict(train_x1_f)
r4 = r3 - y_hat_3
final_pred += r4*(learning_rate)
#iteration 5
clf.fit(train_x1_f, r4)
y_hat_4 = clf.predict(train_x1_f)
r5 = r4 - y_hat_4
final_pred += r5*(learning_rate)
for i in range(len(final_pred)):
  final_pred[i] = round(final_pred[i])

print(getAccuracy(test_y1, final_pred)*100)

#---------------------------------------------------------------------------------------------------#

#Answer 2
#Reference taken from ther FeedForward neural network code provided in lecture 18 by prof. AV Subramanyam
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

val_x2 = test_x2[0:500,:,:]
val_y2 = test_y2[0:500]
test_images = test_x2[500:,:,:]
test_labels = test_y2[500:]

# Normalization
train_x2 = (train_x2 / 255) - 0.5
test_images = (test_images / 255) - 0.5
val_x2 = (val_x2 / 255) - 0.5

# Flattening
train_x2 = train_x2.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))
val_x2 = val_x2.reshape((-1, 784))

# Build the model
model = Sequential() # create model
model.add(Dense(512, input_dim=784, trainable=True,activation='relu', use_bias=True, kernel_initializer=initializers.he_normal(seed=None))) 
model.add(Dropout(0.5))

model.add(Dense(256, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Dense(128, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Dense(64, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Dense(10, trainable=True, activation='softmax')) # output layer

#optimizer
sgd = optimizers.SGD(learning_rate=0.01, momentum=0.9)

# Compile the model.
model.compile(
  optimizer=sgd,
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# obtain the summary
model.summary()

#  early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 5)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', 
                     verbose=1, save_best_only=True)

# Train the model.
history=model.fit(
  train_x2,
  to_categorical(train_y2),
  validation_data=(val_x2, to_categorical(val_y2)),  
  epochs=30,
  batch_size=512,
  shuffle = True,
  callbacks=[es,mc],
)

# Evaluate the model.
saved_model = load_model('best_model.h5')
scores = saved_model.evaluate(
  test_images,
  to_categorical(test_labels)
)

print('Test accuracy:', scores[1])

pyplot.plot(history.history['loss'], label='training loss')
pyplot.legend()
pyplot.show()
from sklearn.metrics import classification_report
y_pred = np.argmax(model.predict(test_images), axis=-1)
print(classification_report(test_labels, y_pred))

#---------------------------------------------------------------------------------------------------#

#Answer 3
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 5)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', 
                     verbose=1, save_best_only=True)
val_x1 = test_x1[0:500,:]
val_y1 = test_y1[0:500]
test_images1 = test_x1[500:,:]
test_labels1 = test_y1[500:]

# Normalization
train_x1 = (train_x1 / 255) - 0.5
test_images1 = (test_images1 / 255) - 0.5
val_x1 = (val_x1 / 255) - 0.5

# Flattening
train_x1 = train_x1.reshape((-1, 784))
test_images1 = test_images1.reshape((-1, 784))
val_x1 = val_x1.reshape((-1, 784))

# Build the model
model1 = Sequential() # create model
model1.add(Dense(512, input_dim=784, trainable=True,activation='relu', use_bias=True, kernel_initializer=initializers.he_normal(seed=None)))
model1.add(Dropout(0.5))
model1.add(Dense(128, input_dim=512, trainable=True,activation='relu', use_bias=True, kernel_initializer=initializers.he_normal(seed=None))) 
model1.add(Dense(64, input_dim=128, trainable=True,activation='relu', use_bias=True, kernel_initializer=initializers.he_normal(seed=None))) 
model1.add(Dense(128, input_dim=64, trainable=True,activation='relu', use_bias=True, kernel_initializer=initializers.he_normal(seed=None))) 
model1.add(Dense(512, input_dim=128, trainable=True,activation='relu', use_bias=True, kernel_initializer=initializers.he_normal(seed=None))) 
model1.add(Dense(784, input_dim=512, trainable=True,activation='relu', use_bias=True, kernel_initializer=initializers.he_normal(seed=None))) 

#optimizer
adam = optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

# Compile the model.
model1.compile(
  optimizer=adam,
  loss='mean_squared_error'
)

# obtain the summary
model1.summary()

history1 = model1.fit(train_x1, train_x1, epochs=10,batch_size=512)
pyplot.plot(history1.history['loss'], label='training loss')
pyplot.legend()
pyplot.show()

from keras.models import Model
model2= Sequential()
model2.add(model1.layers[0])
model2.add(model1.layers[1])
model2.add(model1.layers[2])
model2.add(model1.layers[3])
model2.add(Dense(32, input_dim=64, trainable=True,activation='relu', use_bias=True, kernel_initializer=initializers.he_normal(seed=None))) 
model2.add(Dense(10, input_dim=32, trainable=True,activation='softmax', use_bias=True, kernel_initializer=initializers.he_normal(seed=None))) 
model2.compile(
  optimizer=adam,
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)
history2=model2.fit(
  train_x1,
  to_categorical(train_y1),
  validation_data=(val_x1, to_categorical(val_y1)),  
  epochs=10,
  batch_size=512,
  shuffle = True,
)

# Evaluate the model.
saved_model = load_model('best_model.h5')
scores1 = saved_model.evaluate(
  test_images1,
  to_categorical(test_labels1)
)

print('Test accuracy:', scores1[1])
y_pred1 = np.argmax(model2.predict(test_images1), axis=-1)
print(classification_report(test_labels1, y_pred1))

#---------------------------------------------------------------------------------------------------#

#Answer 4
import random
def getClassWiseAccuracy(original, toCompare, param):
    actualParam = 0
    matchParam = 0
    for i in range(len(original)):
        if(original[i] == param):
            actualParam += 1
            if(toCompare[i] == param):
                matchParam += 1
    return matchParam/actualParam


clf1 = tree.DecisionTreeClassifier()
clf1 = clf1.fit(train_x1_f, train_y1)
dtc = clf1.predict(test_x1_f)
print("Accuracy before bagging:")
print("Testing accuracy = " + str(getAccuracy(test_y1, dtc)*100))
print("Classwise accuracies:")
for i in range(10):
    print(i, end=" - ")
    print(getClassWiseAccuracy(test_y1, dtc, i))


bag1 = np.zeros((60000, 784))
bag1_y = np.zeros(60000)
bag2= np.zeros((60000, 784))
bag2_y = np.zeros(60000)
bag3= np.zeros((60000, 784))
bag3_y = np.zeros(60000)
for i in range(len(train_x1_f)):
  i1 = random.randint(0,59999)
  i2 = random.randint(0,59999)
  i3 = random.randint(0,59999)
  bag1[i] = train_x1_f[i1]
  bag1_y[i] = train_y1[i1]
  bag2[i] = train_x1_f[i2]
  bag2_y[i] = train_y1[i2]
  bag3[i] = train_x1_f[i3]
  bag3_y[i] = train_y1[i3]

clf2 = tree.DecisionTreeClassifier()
a1 = clf2.fit(bag1, bag1_y)
dtc1 = clf2.predict(test_x1_f)
clf3 = tree.DecisionTreeClassifier()
a2 = clf3.fit(bag2, bag2_y)
dtc2 = clf3.predict(test_x1_f)
clf4 = tree.DecisionTreeClassifier()
a3 = clf4.fit(bag3, bag3_y)
dtc3 = clf4.predict(test_x1_f)

majority_arr = np.zeros(10000)
for i in range(len(dtc1)):
  v1 = dtc1[i]
  v2 = dtc2[i]
  v3 = dtc3[i]
  if (v1==v2 or v1==v3):
    majority_arr[i] = v1
  elif (v2==v3):
    majority_arr[i] = v2
  else:
    majority_arr[i] = max(v1, v2, v3)

print("Accuracy after bagging:")
print("Testing accuracy - " + str(getAccuracy(test_y1, majority_arr)*100))
print("Classwise accuracies:")
for i in range(10):
    print(i, end=" - ")
    print(getClassWiseAccuracy(test_y1, majority_arr, i))