import tensorflow
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2

model = tensorflow.keras.models.load_model('cnn_dogscats.hdf5')

def predict(file):
    #test_img = image.load_img(file,target_size=(64,64))
    #test_img = image.img_to_array(test_img)
    #test_img = cv2.imread(file)
    test_img = cv2.resize(file,(64,64))
    test_img = np.expand_dims(test_img,axis=0)
    test_img = test_img
    result = model.predict(test_img)
    return result[0][0]




