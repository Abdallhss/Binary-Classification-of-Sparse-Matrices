"Evaluate the model on test data"
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

#import test data
Field = np.load('field.npy')

ROWS, COLS, CHANNELS = 40,60,1
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

#Load saved model
model = load_model('CR_classifier.h5')

#Classify unlabeled data
Test_data = prep_data(Field)
predictions = model.predict(Test_data)

#Create a dataframe with predicted labels and confidence level
Test_labels = np.argmax(predictions,axis = 1)
Test_labels_conf = np.array([predictions[i,Test_labels[i]] for i in range(len(Test_labels))])
Data = {'ML Label': Test_labels, 'Confidence': Test_labels_conf}
Classification_result = pd.DataFrame(data=Data)

#Compare with manual labeling
#Comment for other tests
manual_labels = [1,0,0,0,0,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,0,1,0,0,1,1,0,1,1,1,1,0,1,1,0,0,0,1,1,1,1,1,1,0,1,1,0,0,1,1,
                 0,1,1,0,1,1,0,0,0,0,1,0,1,0,1,0,1,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,0,1,1,0,1,0,1,1,0,1,
                 0,0,1,1,0,0,1,1,0,1,0,1,1,0,0,0,0,1,0,1,0,1,1,0,0,0,0,1,1,0,1,0,1,0,1,0,1,1,1,0,0,1,0,0,1,1,1,0,1,0,
                 0,0,1,1,1,1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,1,1,1,0,1,0,1,1,1,1,0,0,0,1,1,1,0,0,0,0,1,1,0]

Classification_result['Manual_labels'] = np.array(manual_labels)

