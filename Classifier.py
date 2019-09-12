"Generating the classifier"

# Import neccessary libraries 
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import training and test data
Data_a = np.load('class_a.npy')
label_a = np.zeros(Data_a.shape[0])         #Label class [A] as zeros
Data_b = np.load('class_b.npy')
label_b = np.ones(Data_b.shape[0])          #Label class [B] as ones


# Data preperation
# Data normalization and noise reduction
def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.float)    
    for i, image_file in enumerate(images):
        data[i] = image_file[:,:,None] > 0.5    # Remove noise in test data                   
        data[i] -= data[i].mean()               # Normalize data
        data[i] /= data[i].std()
    
    return data

# Prepare train and validation data
ROWS, COLS, CHANNELS = 40,60,1
Data_X = np.concatenate((Data_a, Data_b), axis=0)
Data_X = prep_data(Data_X)
Data_y = np.concatenate((label_a, label_b), axis=0)
Data_y = to_categorical(Data_y)

#Train on all data
X_train, X_dummy, y_train, y_dummy = train_test_split(Data_X, Data_y, test_size=0.0, shuffle = True)

#Use Keras image generator to augment the training data by introducing horizontal and vertical shifts
datagen = ImageDataGenerator(width_shift_range=0.5, height_shift_range=0.5)
datagen.fit(X_train)

#The following architecture is based on mnist keras tutorial 
#https://keras.io/examples/mnist_cnn/

#Define basic model parameters
batch_size = 8
num_classes = 2
epochs = 12
kernel_size = (3,3)
pool_size = (2,2)

#The model is created in Keras
#CNN architecture is adopted
#Maxpooling, Dense, Conv2D and dropout layers were used 
model = Sequential()

model.add(MaxPooling2D(pool_size=pool_size,input_shape=(ROWS, COLS, CHANNELS)))

model.add(Conv2D(16, kernel_size = kernel_size, padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Conv2D(32, kernel_size = kernel_size, padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size = kernel_size, padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Conv2D(128, kernel_size = kernel_size, padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),epochs=epochs)

#Save model to be used later
model.save('CR_classifier.h5')  
del model  # deletes the existing model
