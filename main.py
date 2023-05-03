import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

model = load_model('model.h5')

st.title('Small Images Classification')
st.markdown('Upload Image')


def numbers_to_strings(argument):
    switcher = {
        0:	'airplane',
        1:	'automobile',
        2:	'bird',
        3:	'cat',
        4:	'deer',
        5:	'dog',
        6:	'frog',
        7:	'horse',
        8:	'ship',
        9:	'truck'
    }
    return switcher.get(argument, "nothing")


image = st.file_uploader("Upload image")
submit = st.button('Predict')
if submit:
    if image is not None:
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(opencv_image, channels="BGR")
        opencv_image = cv2.resize(opencv_image, (32, 32))
        opencv_image.shape = (1, 32, 32, 3)
        Y_pred = model.predict(opencv_image)
        ypred1 = np.argmax(Y_pred)
        predict = ""
        predict = numbers_to_strings(ypred1)
        st.title(predict)
