import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
import logging

logging.basicConfig(filename='face_mask_detection.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DIR1 = r'train'
#DIR2 = r'dataset\valid'
CATEGORY = ['with_mask', 'without_mask']


def generate(directories, cat, size):
    logging.info(f"Generating data from {directories} with categories: {cat} and size: {size}")
    data_gen = []
    for category in cat:
        folder = os.path.join(directories, category)
        i = 0
        if category == 'without_mask':
            label = 0
        else:
            label = 1
        for paths in os.listdir(folder):
            if i <= size:
                img_paths = os.path.join(folder, paths)
                img = cv2.imread(img_paths)
                if img is None:
                    continue
                else:
                    img = cv2.resize(img, (100, 100))
                    img = img / 255  # scaling the values to a range of 0 to 1
                    # img_arr= preprocess_input(img_arr)
                data_gen.append([img, label])
            else:
                break
            i = i + 1
    return data_gen


data_train = generate(DIR1, CATEGORY, 5000)
#data_valid = generate(DIR2, CATEGORY, 1416)

np.random.RandomState(seed=42).shuffle(data_train)
#np.random.RandomState(seed=42).shuffle(data_valid)


def separate(dat):
    x = []
    y = []

    for f, l in dat:
        x.append(f)
        y.append(l)

    x = np.array(x)
    y = np.array(y)

    return x, y


X_data_train, y_data_train = separate(data_train)
#X_data_valid, y_data_valid = separate(data_valid)

# creating the model

from keras.applications.vgg16 import VGG16

vgg = VGG16(include_top=False, input_shape=(100, 100, 3))

# vgg.summary()
model = Sequential()

for layer in vgg.layers:
    layer.trainable = False

for layer in vgg.layers:
    model.add(layer)

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

logging.info("Training the model")
history = model.fit(X_data_train, y_data_train, epochs=10, validation_split=0.2)

# plotting of the losses
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.figure()

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()

# predicting using the model

#new_model = load_model('Face_mask_model.h5')

#directory = r'C:\Users\DIPYAMAN GOSWAMI\OneDrive\Desktop'
file = 'without_mask_7.jpg'

#img_path = os.path.join(directory, file)
img_arr = cv2.imread(file)
cv2.imshow('image', img_arr)
img_arr = cv2.resize(img_arr, (100, 100))
img_arr = img_arr / 255

img_arr = img_arr.reshape((1,) + img_arr.shape)

logging.info("Making predictions using the model")
prediction = model.predict(img_arr)

if prediction >= 0.5:
    logging.info('Prediction: person with mask')
    print('person with mask')
else:
    logging.info('Prediction: person without mask')
    print('person without mask')

cv2.waitKey(0)
cv2.destroyAllWindows()

logging.info("Saving the model")
model.save('Face_mask_model.h5')
