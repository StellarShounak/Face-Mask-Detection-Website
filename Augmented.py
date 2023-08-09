import os
import cv2
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

DIR = r'train'
CAT = ['with_mask', 'without_mask']

for category in CAT:
    folder = os.path.join(DIR, category)
    for path in os.listdir(folder):
        img_path = os.path.join(folder, path)
        img = cv2.imread(img_path)
        if img is None:
            continue
        else:
            img = img.reshape((1,) + img.shape)
            i = 0
            for image in datagen.flow(img, batch_size=32, save_to_dir=folder, save_format='jpg'):
                i = i + 1
                if i > 15:
                    break
