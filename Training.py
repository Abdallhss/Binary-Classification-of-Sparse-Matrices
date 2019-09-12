"Training and validation of the classifier"
# Import neccessary libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
Field = np.load('field.npy')

#Uncomment to visualize the data
'''
i = 74
image_a = Data_a[i]
image_b = Data_b[i]

plt.figure(1)
plt.imshow(image_a,cmap="gray")
plt.figure(2)
plt.imshow(image_b,cmap="gray")

# class A seems to be a circle // class B seems to be a square
plt.figure(3)
test_img = Field[108]
plt.imshow(test_img,cmap="gray")

'''

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

X_train, X_valid, y_train, y_valid = train_test_split(Data_X, Data_y,test_size=0.33, shuffle = True)

#Use Keras image generator to augment the training data by introducing horizontal and vertical shifts
datagen = ImageDataGenerator(width_shift_range=0.5,height_shift_range=0.5)
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

history  = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),epochs=epochs)

# plot learning curve
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(history.history['acc'], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Loss', color=color)  
ax2.plot(history.history['loss'], color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  
plt.show()

#Evaluate the accuracy on the validation data
score = model.evaluate(X_valid, y_valid, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



