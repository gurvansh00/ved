import streamlit as st
import tensorflow as tf
from PIL import Image
import os
import numpy as np
from tensorflow.keras.preprocessing import image
import skimage
import tempfile
import pandas as pd

st.markdown('''
<style>
.stApp{
background-color:#98FF98;
}
</style>
''',unsafe_allow_html=True)
# Load your TensorFlow model here
MODEL2 = tf.keras.models.load_model("inception_model.h5")
MODEL2 = tf.keras.models.load_model("mobilenet_model.h5")
MODEL3 = tf.keras.models.load_model("nasnet_model.h5")

# Define the categories your model predicts on
categories = ["Healthy", "Bacterial Leaf Blight", "Brown Spot", "Leaf Smut"]

# Streamlit app title and description
st.title('Rice Disease Detection System')
st.write('Upload an image in "jpg/png/jpeg" format')
# Upload image through Streamlit's file uploader
uploaded_file = st.file_uploader("Choose an image...")
if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    img.save('ok.jpg')
    image_to_predict = image.load_img('ok.jpg', target_size=(224, 224))
    image_to_predict=image.img_to_array(image_to_predict,data_format="channels_last")
    image_to_predict = np.expand_dims(image_to_predict, axis=0)
    image_to_predict=image_to_predict/255
    images_array=[]
    images_array.append(image_to_predict)
    p1= MODEL1.predict(images_array)
    p2= MODEL2.predict(images_array)
    p3=MODEL3.predict(images_array)
    accu1=round(np.max(p1)*100,2)
    accu2=round(np.max(p2)*100,2)
    accu3=round(np.max(p3)*100,2)
    totaccu=accu2+accu3+accu1
    healthaccu=0
    blightaccu=0
    brownaccu=0
    smutaccu=0
    list=[p2,p3]
    for i in list:
        if categories[np.argmax(i)] =="Healthy":
            healthaccu += round(np.max(i)*100, 2)
        elif categories[np.argmax(i)] == "Bacterial Leaf Blight":
            blightaccu += round(np.max(i)*100, 2)
        elif categories[np.argmax(i)] == "Brown Spot":
            brownaccu += round(np.max(i)*100, 2)
        elif categories[np.argmax(i)] == "Leaf Smut":
            smutaccu += round(np.max(i)*100, 2)
    healthaccu=round((healthaccu/totaccu)*100,2)
    blightaccu = round((blightaccu/totaccu)*100,2)
    brownaccu =round((brownaccu/totaccu)*100,2)
    smutaccu = round((smutaccu/totaccu)*100,2)
    predictions = [healthaccu,blightaccu,brownaccu,smutaccu]
    st.write("### Classification Results:")
    df = pd.DataFrame({'Diseases':categories,'Confidence':predictions})
    p2 = st.empty()
    p2.markdown(df.style.hide(axis="index").to_html(), unsafe_allow_html=True)


st.markdown(
'''
            Resources:
            <a href="http://www.agritech.tnau.ac.in/expert_system/paddy/cpdisbrownspot.html" target="_blank" title="Brownspot">
                Brownspot</a> |
            <a href="http://www.agritech.tnau.ac.in/expert_system/paddy/cpdisblb.html" target="_blank" title="Bacterial Leaf Blight">
                Bacterial Leaf Blight</a> |
            <a href="https://plantclinic.tamu.edu/factsheets/leaf-smut-on-grasses/" target="_blank" title="Leaf Smut">Leaf
                Smut</a>
            <br>
            <br>
            <p>This Web API is a part of Project work Made by
                <br>
                Vedanshee Agrawal
                            </p>
''',unsafe_allow_html=True)
