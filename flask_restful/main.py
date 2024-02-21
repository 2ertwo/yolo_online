from flask_restful import reqparse, Resource, Api
from werkzeug.datastructures import FileStorage
from flask import Flask
from yolo import loadcv2dnnNetONNX, preprocess4cv2dnn, gen_result
import cv2
import os
import json

n = loadcv2dnnNetONNX(os.path.join('/','www','fg','models','best.onnx'))

class UploadImg(Resource):
    def __init__(self):
        # 创建一个新的解析器
        self.parser = reqparse.RequestParser()
        # 增加imgFile参数，用来解析前端传来的图片。
        self.parser.add_argument('imgFile', required=True, type=FileStorage, location='files', help="imgFile is wrong.")

    def post(self):
        img_file = self.parser.parse_args().get('imgFile')
        img_file.save(img_file.filename)
        blobImage, image = preprocess4cv2dnn(img_file.filename)
        outNames = n.getUnconnectedOutLayersNames()
        n.setInput(blobImage)
        outs = n.forward(outNames)
        names = [ 'overflow', 'garbage', 'garbage_bin' ]
        r = gen_result(outs, names)
        print(r)
        return json.dumps(r), 201


if __name__ == '__main__':
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(UploadImg, '/uploadimg')
    app.run(host='0.0.0.0',port=2024)
