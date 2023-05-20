from google.colab import drive
drive.mount('/content/drive')

!pip install streamlit

#Importing the libraries

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import load_model

data_dir = '/content/drive/MyDrive/dataset/Multi-class Weather Dataset'
img_size = (224, 224)
batch_size = 24
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(data_dir, target_size=img_size, batch_size=batch_size, subset='training')
val_generator = datagen.flow_from_directory(data_dir, target_size=img_size, batch_size=batch_size, subset='validation')

lb = LabelBinarizer()
lb.fit(train_generator.classes)
num_classes = train_generator.num_classes

num_epochs = 10
learning_rate = 0.0001
model = Sequential()
model.add(Flatten(input_shape=train_generator.image_shape))
model.add(Dense(10, activation='relu'))
model.add(Dense(4, activation='sigmoid'))
optimizer = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(train_generator, epochs=num_epochs, validation_data=val_generator)

test_generator = datagen.flow_from_directory(data_dir, target_size=img_size, batch_size=batch_size, subset='validation')

y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)

print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

#number of epochs
num_epochs = 10
learning_rate = 0.0001

#model creation
model = Sequential()
model.add(Flatten(input_shape=train_generator.image_shape))
model.add(Dense(10, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
optimizer = Adam(lr=learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model_path = '/content/drive/MyDrive/Colab Notebooks/Multi-class Weather Dataset Model/best_model.h5'

checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
history = model.fit(train_generator, epochs=num_epochs, validation_data=val_generator, callbacks=[checkpoint])

img = Image.open('/content/drive/MyDrive/dataset/Multi-class Weather Dataset/Cloudy/cloudy1.jpg').resize((224, 224))

x = np.array(img) / 255.0
x = np.expand_dims(x, axis=0)
predictions = model.predict(x)
class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
predicted_class = class_names[np.argmax(predictions)]
probability = np.max(predictions)

print(f'Predicted class: {predicted_class}')
print(f'Probability: {probability}')

plt.imshow(img)
plt.show()

img = Image.open('/content/drive/MyDrive/dataset/Multi-class Weather Dataset/Rain/rain10.jpg').resize((224, 224))

x = np.array(img) / 255.0
x = np.expand_dims(x, axis=0)
predictions = model.predict(x)
class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
predicted_class = class_names[np.argmax(predictions)]
probability = np.max(predictions)

print(f'Predicted class: {predicted_class}')
print(f'Probability: {probability}')

plt.imshow(img)
plt.show()
