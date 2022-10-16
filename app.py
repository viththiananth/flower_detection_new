import streamlit as st
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image, ImageOps
import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

def load_model():
    model = tf.keras.models.load_model("my_model.h5")
    return model

model=load_model()

st.write("""
         # Welcome to Flower Classification
         """)

file = st.file_uploader("Please upload a Flower Image" , type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (150,150)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction


if file==None:
    st.text("Please Upload a file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['daisy’, ‘dandelion’, ‘roses’ , ‘sunflowers’, ‘tulips']
    string = "The Flower in the Image is most likely is : " + class_names[np.argmax(predictions) ]
    st.success(string)
    
