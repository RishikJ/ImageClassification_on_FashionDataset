# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 17:33:08 2022

@author: rishi
"""

import keras
import numpy
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.layers import BatchNormalization
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History

image_directory = 'train/'
df_train = pd.read_csv('train_label.csv')    
print(df_train.head())     # printing first five rows of the file
print(df_train.columns)


SIZE = 224
X_dataset = [] 
X_DATASET = "xdataset.npy"
for i in tqdm(range(df_train.shape[0])):
    img = image.load_img(image_directory +df_train['file_name'][i], target_size=(SIZE,SIZE,3))
    img = image.img_to_array(img)
#    img = img/255.
    X_dataset.append(img)  
X = np.array(X_dataset)

numpy.save(X_DATASET, X)

numpy.load(X_DATASET)

from tensorflow.keras.applications.resnet50 import preprocess_input
X = preprocess_input(X)

y = df_train["label"]
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.2)

#Other model to try...
#VGG model with 3 blocks + dropout + batch normalization
#model3_drop_norm = Sequential()
#model3_drop_norm.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(100, 100, 3)))
#model3_drop_norm.add(BatchNormalization())
#
#model3_drop_norm.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
#model3_drop_norm.add(BatchNormalization())
#model3_drop_norm.add(MaxPooling2D((2, 2)))
#
#model3_drop_norm.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
#model3_drop_norm.add(BatchNormalization())
#
#model3_drop_norm.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
#model3_drop_norm.add(BatchNormalization())
#model3_drop_norm.add(MaxPooling2D((2, 2)))
#
#model3_drop_norm.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
#model3_drop_norm.add(BatchNormalization())
#
#model3_drop_norm.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
#model3_drop_norm.add(BatchNormalization())
#model3_drop_norm.add(MaxPooling2D((2, 2)))
#
#model3_drop_norm.add(Flatten())
#model3_drop_norm.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
#model3_drop_norm.add(BatchNormalization())
#model3_drop_norm.add(Dense(11, activation='softmax'))


#model = Sequential()
#model.add(Conv2D(32, 3, activation = "sigmoid", padding = 'same', input_shape = (100, 100, 3)))
#model.add(BatchNormalization())
#
#model.add(Conv2D(32, 3, activation = "sigmoid", padding = 'same', kernel_initializer = 'he_uniform'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D())
#
#model.add(Conv2D(64, 3, activation = "sigmoid", padding = 'same', kernel_initializer = 'he_uniform'))
#model.add(BatchNormalization())
#
#model.add(Conv2D(64, 3, activation = "sigmoid", padding = 'same', kernel_initializer = 'he_uniform'))
#model.add(BatchNormalization()) 
#model.add(MaxPooling2D())
#
#model.add(Flatten())
#model.add(Dense(128, activation = "sigmoid", kernel_initializer = 'he_uniform'))
#model.add(Dense(11, activation = 'softmax'))

from keras.applications.resnet50 import ResNet50

model = ResNet50(include_top = False, weights = "imagenet", input_shape = (SIZE,SIZE,3))

x = Flatten()(model.layers[-1].output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(11, activation='softmax')(x)
model = Model(inputs = model.inputs, outputs = output)

for layers in model.layers[:143]:
    layers.trainable = False
model.summary()
# compile model
#from keras.optimizers import SGD
#opt = SGD(lr=0.001, momentum=0.9)

history = History()
callbacks = [history, 
             EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=1, cooldown=0, min_lr=1e-7, verbose=1),
             ModelCheckpoint(filepath='fashionResNet50.best.hdf5', 
             monitor='val_accuracy', verbose=1, 
             save_best_only=True, save_weights_only=True, mode='auto')]
  
opt = Adam(lr=1e-4)           
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(X_train, y_train, epochs=1,verbose = 1, validation_data=(X_test, y_test), batch_size=64, callbacks = callbacks)

model.save("NormalModel_ADAM_fashion.h5")

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


from keras.models import load_model
new_model = load_model("NormalModel_ADAM_fashion.h5")
history = new_model.fit(X_train, y_train, epochs=8, validation_data=(X_test, y_test), batch_size=32)

test_image_directory = 'test/'
df = pd.read_csv('sample_submission.csv') 

SIZE = 100
X_dataset_test = [] 
X_DATASET_TEST = "xdataset_test.npy"

for i in tqdm(range(df.shape[0])):
    img = image.load_img(test_image_directory +df['file_name'][i], target_size=(SIZE,SIZE,3))
    img = image.img_to_array(img)
#    img = img/255.
    X_dataset_test.append(img)
    
X_dataset_test = np.array(X_dataset_test)

numpy.save(X_DATASET_TEST, X_dataset_test)
X_dataset_test = numpy.load(X_DATASET_TEST)

y_dataset_test = model.predict(X_dataset_test)
y_dataset_test_label = (np.argmax(y_dataset_test, axis=1)).reshape(-1, 1)

df_test = pd.DataFrame(y_dataset_test_label, columns = ["label"])
df["label"] = df_test["label"]

df.to_csv("result.csv",index = False)

print("hi")



