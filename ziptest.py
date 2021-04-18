import streamlit as st
import zipfile
from tensorflow.keras.preprocessing import image
import tensorflow
from tensorflow.keras.models import load_model
import numpy as np
import os
uploaded_file = st.file_uploader("Upload Files",type=['zip'])
if uploaded_file is not None:
    with open("abc.zip","wb") as f:
        f.write(uploaded_file.getbuffer())
zip_ref = zipfile.ZipFile('abc.zip','r')
zip_ref.extractall()
zip_ref.close()
import numpy as np
model=load_model('crackmodelworking.h5')
dirs = os.listdir('crackimgs')
res=''
for file in dirs:
    filename='crackimgs\\'+file
    img1 = image.load_img(filename, target_size=(227,227))
    img = image.img_to_array(img1)
    img = img/255
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img, batch_size=None,steps=1)
    if(prediction[:,:]>0.5):
        value ='Iam {:2.1f}% sure that image has no cracks'.format(100*prediction[0,0])
    else:
        value ='iam {:2.1f}% sure that image has cracks'.format((1.0-prediction[0,0])*100)
    res=res+'\n'+file+'----'+value+'\n'
st.text(res)
