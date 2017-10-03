# import matplotlib.pyplot as plt
import keras
import logging
import numpy as np
import os

from data import DataLoader

from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

N_TRAIN_PER_CATEGORY = 2000
N_TEST_PER_CATEGORY = 200
TRAIN_BATCH_SIZE = 50
EPOCHS = 50
NUM_CLASSES = 10
MODEL_NAME = 'quickdraw10'

logging.basicConfig(level=logging.DEBUG)

training_data_loader = DataLoader(dataset_type='train', 
                                  batch_size=N_TRAIN_PER_CATEGORY, 
                                  count_per_category=N_TRAIN_PER_CATEGORY,
                                  categories_file=os.path.join('categories', MODEL_NAME + '.txt'))
testing_data_loader = DataLoader(dataset_type='test',
                                 batch_size=N_TEST_PER_CATEGORY, 
                                 count_per_category=N_TEST_PER_CATEGORY, 
                                 categories_file=os.path.join('categories', MODEL_NAME + '.txt'))

X_train, y_train = training_data_loader.next_batch()
for i in range(NUM_CLASSES - 1):
    X, y = training_data_loader.next_batch()
    X_train = np.concatenate((X_train, X))
    y_train = np.concatenate((y_train, y))
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)

X_test, y_test = testing_data_loader.next_batch()
for i in range(NUM_CLASSES - 1):
    X, y = testing_data_loader.next_batch()
    X_test = np.concatenate((X_test, X))
    y_test = np.concatenate((y_test, y))
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

model = Sequential()

model.add(Conv2D(64, (15, 15), strides=3, padding='valid', input_shape=(225, 225, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

model.add(Conv2D(128, (5, 5), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

model.add(Conv2D(512, (7, 7), padding='valid'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv2D(512, (1, 1), padding='valid'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv2D(NUM_CLASSES, (1, 1), padding='valid'))
model.add(Flatten())
model.add(Activation('softmax'))

save_dir='models'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
plot_model(model, to_file=os.path.join(save_dir, MODEL_NAME + '.png'), show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=TRAIN_BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(X_test, y_test),
          shuffle=True)

model_path = os.path.join(save_dir, MODEL_NAME + '.h5')
model.save(model_path)