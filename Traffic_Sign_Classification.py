#Importing the required Libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow
import keras
from keras import models, layers
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

data = []
labels = []
cur_path = os.getcwd()
classes = 43 #We know that there are 43 clasess in this dataset


#Retrieving the images and their labels

#The dataset has folders from 0–42 i.e. 43 classes
for i in range(classes):
  path = os.path.join(cur_path,'Train',str(i))
  images = os.listdir(path)

#iterating on all the images of the index folder
  for img in images:
    try:
      image = Image.open(path + '/'+ img)
      image = image.resize((32,32))
      image = np.array(image)
      data.append(image)
      labels.append(i)
    except:
        print("Error loading image")

#Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

# #Just visualising the data
# plt.figure(figsize = (12,12))
#
# for i in range(4) :
#     plt.subplot(1, 4, i+1)
#     plt.imshow(data[i], cmap='gray')
#
# plt.show()

print(data.shape, labels.shape)
#Splitting training and testing dataset
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#Converting the labels into one hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

#Building a model
model = models.Sequential() #Sequential Model
model.add(layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=x_train.shape[1:]))
model.add(layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Dropout(rate=0.25))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Dropout(rate=0.25))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(43, activation='softmax'))
model.summary()

#Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 15
history = model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_data=(x_test, y_test))
model.save("Traffic_Sign_Model.h5")

#plotting graphs for accuracy
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


#model testing
y_test = pd.read_csv('Test.csv')
labels = y_test["ClassId"].values
img_paths = y_test["Path"].values
test_data=[]
for path in img_paths:
  image = Image.open(path)
  image = image.resize((32,32))
  test_data.append(np.array(image))
test_data = np.array(test_data)
predict_x=model.predict(test_data)
classes_x=np.argmax(predict_x,axis=1)

#Accuracy with the test data
accuracy_score(labels, classes_x)