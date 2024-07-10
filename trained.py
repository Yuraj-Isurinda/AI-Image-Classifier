import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import models
from keras import layers
from keras import datasets

(training_images, training_labels), (testing_images,testing_labels) = datasets.cifar10.load_data()
training_images,testing_images = training_images / 255, testing_images / 255

class_names =['Plane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']


   # Optional for resource management

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

model = models.load_model('image_classifier.keras')

#image load
img = cv.imread('object3.jpg')

#chnage color format
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img])/255)
index = np.argmax(prediction)
print(f'Prediction is {class_names[index]}')
plt.show()
