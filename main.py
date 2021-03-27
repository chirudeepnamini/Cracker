from flask import Flask,render_template,request
import os 
import tensorflow
app=Flask(__name__)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
model=load_model('crackmodelworking.h5')
@app.route('/imghandler',methods=['POST','GET'])
def imghandler():
    global model
    filename=None
    if request.method=="POST":
        file=request.files['file']
        filename=file.filename
        print(type(file))
        imgpath=os.path.join('\\cracker\\static\\',filename)
        relpath='\\static\\'+filename
        file.save(imgpath)
        img1 = image.load_img(imgpath, target_size=(227,227))
        img = image.img_to_array(img1)
        img = img/255
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img, batch_size=None,steps=1)
        if(prediction[:,:]>0.5):
            value ='Iam {:2.1f}% sure that image has no cracks'.format(100*prediction[0,0])
        else:
            value ='iam {:2.1f}% sure that image has cracks'.format((1.0-prediction[0,0])*100)
        return render_template('result.html',value=value,imgpath=relpath)
        
        
    
    


@app.route('/')
def fn():
    return render_template('home.html')
@app.route('/about')
def about():
    return "<h1>hello</h1>"
if __name__=='__main__':
    app.run(debug=True)
    