from flask import Flask,render_template,request,jsonify
from commons import get_tensor
from infrence import get_class_name
import torch
import os
import io
import base64
from PIL import Image
from base64 import decodestring
app = Flask(__name__)
@app.route('/',methods=['GET','POST'])
def hello_world() :
    if request.method == 'GET':
        return render_template('index.html', value="hello")
    if  request.method == 'POST':
        if 'file' not in request.files :
            print("File Not Uploaded")
            return
        file = request.files['file']
        image=file.read()
        category=get_class_name(image_bytes=image)
        #print(get_tensor(image_bytes=image))
        #print(tensor.shape)
        return render_template('result.html',flower=category)

@app.route('/predict_api/', methods=['GET', 'POST'])
def predict_api():
    nq = request.json
    print(nq['mytext'])
    bytestr=nq['mytext'].encode('utf-8')
    f = io.BytesIO(base64.b64decode(bytestr))
    image=f.read()
    category=get_class_name(image_bytes=image)
    print(category)
    return jsonify({"prediction":str(category)})

if __name__=='__main__':
    app.run(debug=True,port=os.getenv('PORT',5000))
