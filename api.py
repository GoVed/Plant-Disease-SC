# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:25:56 2022

@author: vedhs
"""

from flask import Flask,send_file,request
from PIL import Image
import numpy as np
import io 
import iproc
import ml

app = Flask(__name__)

class user:
    def __init__(self):
        self.img=None
        self.ip=None

userImgs={}
segmentedImgs={}
segmentModel=ml.SegmentML()
segmentModel.loadModels('D:\\vedhs\\Projects\\Plant-Disease-SC-copy\\model\\segment\\Apple')

@app.route('/segment', methods=['POST'])
def segment():

    if request.remote_addr in userImgs:
        if request.remote_addr not in segmentedImgs:
            
            segmentedImgs[request.remote_addr]=segmentModel.predict(userImgs[request.remote_addr])[0]
            
        data=Image.fromarray(iproc.overlayMask(userImgs[request.remote_addr], segmentedImgs[request.remote_addr]))
        # data=Image.fromarray(userImgs[request.remote_addr])
        rawBytes = io.BytesIO()
        data.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        
        return send_file(rawBytes,mimetype='image/gif')
    else:
        return '0'

@app.route('/calculateDisease', methods=['POST'])
def calcDisease():
    if request.remote_addr in userImgs:        
        return str(iproc.calculateDiseasePart(iproc.overlayMask(userImgs[request.remote_addr], segmentedImgs[request.remote_addr])))
        
    else:
        return '0'

    
@app.route('/upload', methods=['POST'])
def imageDown():
    imagefile = request.files.get('img', '')
    img = np.asarray(Image.open(imagefile.stream))    
    userImgs[request.remote_addr]=img
    if request.remote_addr in segmentedImgs:
        del segmentedImgs[request.remote_addr]
    return '1'

if __name__ == '__main__':
    app.run(debug=False)
    
    
