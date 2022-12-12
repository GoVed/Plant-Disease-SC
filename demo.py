# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:11:49 2022

@author: vedhs
"""

from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import ml
import iproc

import glob

files = glob.glob('static/highlight/*')
for f in files:
    os.remove(f)
    
files = glob.glob('static/raw/*')
for f in files:
    os.remove(f)
    
files = glob.glob('static/segment/*')
for f in files:
    os.remove(f)
    
files = glob.glob('static/swmad/*')
for f in files:
    os.remove(f)
    
files = glob.glob('static/swmadAfterSegment/*')
for f in files:
    os.remove(f)
    
if not os.path.isdir("/static"):
    os.makedirs("/static")
    
app = Flask(__name__)
userImgs={}
userFileName={}
plantPred={}
diseasePred={}
segmentedImgs={}
swmad={}
swmadAfterSegment={}
highlight={}

plantModel=ml.plantML()
plantModel.loadModel('model/plant.h5')

segmentModel=ml.SegmentML()


@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def upload():

    if 'uploadimg' in request.form:
        file = request.files['img']
        fname = secure_filename(file.filename)
        
        imagefile = request.files.get('img', '')
        img = Image.open(imagefile.stream)
        img.save('static/raw/' + fname)
        userImgs[request.remote_addr]=np.asarray(img)
        userFileName[request.remote_addr]=fname
        
        fname_after_processing = fname
        return jsonify({'result_image_location': url_for('static',filename= 'raw/'+fname_after_processing)})

@app.route('/predictPlant',methods=['POST'])
def predictPlant():    
    plantPred[request.remote_addr],swmad[request.remote_addr]=plantModel.predictPath('static/raw/'+userFileName[request.remote_addr],returnswmad=True)
    plantPred[request.remote_addr]=plantPred[request.remote_addr].tolist()
    Image.fromarray(swmad[request.remote_addr]).save('static/swmad/' + userFileName[request.remote_addr])
    folders=iproc.getFolders('data/test/raw')
    res=[x for _, x in sorted(zip(plantPred[request.remote_addr], folders),reverse=True)]
    return jsonify({"res":res,"values":sorted(plantPred[request.remote_addr],reverse=True),'result_image_location': url_for('static',filename= 'swmad/'+userFileName[request.remote_addr])})

@app.route('/segmentImage',methods=['POST'])
def segmentImage():    
    segmentModel.loadModels('model/segment/'+request.json['plant'])
    segmentedImgs[request.remote_addr]=segmentModel.predict(userImgs[request.remote_addr])[0]
    segmentedImgs[request.remote_addr]=segmentedImgs[request.remote_addr].astype(np.float32)/255
    segmentedImgs[request.remote_addr]=userImgs[request.remote_addr]*np.reshape(segmentedImgs[request.remote_addr].astype(np.uint8),(segmentedImgs[request.remote_addr].shape[0],segmentedImgs[request.remote_addr].shape[1],1)) 
    Image.fromarray(segmentedImgs[request.remote_addr]).save('static/segment/' + userFileName[request.remote_addr])
    return jsonify({'result_image_location': url_for('static',filename= 'segment/'+userFileName[request.remote_addr])})
        
@app.route('/predictDisease',methods=['POST'])
def predictDisease():   
    diseaseModel=ml.plantML('data/train/raw/'+request.json['plant'])
    diseaseModel.loadModel('model/disease'+request.json['plant']+'.h5')
    diseasePred[request.remote_addr],swmadAfterSegment[request.remote_addr]=diseaseModel.predictPath('static/segment/'+userFileName[request.remote_addr],returnswmad=True)
    diseasePred[request.remote_addr]=diseasePred[request.remote_addr].tolist()
    Image.fromarray(swmadAfterSegment[request.remote_addr]).save('static/swmadAfterSegment/' + userFileName[request.remote_addr])
    folders=iproc.getFolders('data/test/raw/'+request.json['plant'])
    res=[x for _, x in sorted(zip(diseasePred[request.remote_addr], folders),reverse=True)]
    return jsonify({"res":res,"values":sorted(diseasePred[request.remote_addr],reverse=True),'result_image_location': url_for('static',filename= 'swmadAfterSegment/'+userFileName[request.remote_addr])}) 

@app.route('/calculateDisease',methods=['POST'])
def calculateDisease():
    perc,highlight[request.remote_addr]=iproc.calculateDiseasePart(segmentedImgs[request.remote_addr],plant=request.json['plant'],returnProcImg=True)
    highlight[request.remote_addr]=highlight[request.remote_addr].astype(np.uint8)
    Image.fromarray(highlight[request.remote_addr]).save('static/highlight/' + userFileName[request.remote_addr])
    return jsonify({"res":str(perc),'result_image_location': url_for('static',filename= 'highlight/'+userFileName[request.remote_addr])})

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=False)