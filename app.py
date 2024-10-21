import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np 
st.header('Image Classification Model')

model = load_model(r'C:\Users\jacki\Downloads\image_classification\Image_classify.keras')

data_cate = ['cat', 'dog']

img_height = 180
img_width = 180
image = st.text_input('Enter Image name','9527.jpg')


image_load = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_batch = tf.expand_dims(img_arr,0)

predict = model.predict(img_batch)

score = tf.nn.softmax(predict)
st.image(image)
st.write('Animal in the image is ' + data_cate[np.argmax(score)])
st.write('With accuracy of ' + str(np.max(score)*100))  