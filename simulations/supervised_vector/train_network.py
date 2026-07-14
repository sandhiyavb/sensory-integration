import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, Dense, Flatten, Dropout

#load training data 
input_images = np.load("inputs/input_images_guidance.npy")
output_vectors = np.load("inputs/output_vectors_guidance.npy")

#flatten training data
input_images = input_images.reshape(-1, *input_images.shape[-3:])
output_vectors = output_vectors.reshape(-1, output_vectors.shape[-1])

INPUT_SHAPE = input_images[0].shape


#split and shuffle data
train_x, test_x, train_y, test_y = train_test_split(input_images,output_vectors,
                                                    test_size=0.2, random_state=12)

#build neural network
model = Sequential()
model.add(Conv2D(32, (5,5), input_shape=INPUT_SHAPE,name='cnn_0'))
model.add(Activation('relu',name='activation_0'))
model.add(Conv2D(64, (4,4), strides=(2,2),name='cnn_1'))
model.add(Activation('relu',name='activation_1'))
model.add(Conv2D(64,(3,3),strides=(2,2),name='cnn_2'))
model.add(Activation('relu',name='activation_2'))
model.add(Flatten())
model.add(Dense(64, activation='relu',name='analysis'))
model.add(Dropout(0.1,name='dropout'))
model.add(Dense(3,activation='linear',name='output'))
model.summary()

model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=5)
model.save('vector_model.h5')

#testing data 
input_images_test = np.load("inputs/input_images_test.npy")
output_vectors_test = np.load("inputs/output_vectors_test.npy")

#flatten test data
input_images_test = input_images_test.reshape(-1, *input_images_test.shape[-3:])
output_vectors_test = output_vectors_test.reshape(-1, output_vectors_test.shape[-1])

#evaluate model on test data
model.evaluate(input_images_test, output_vectors_test)
