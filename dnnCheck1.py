
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import numpy as np
import cv2 
from tensorflow.keras.models import load_model


import json
from datetime import datetime
import onnx
import onnxruntime
from onnx import numpy_helper

def getDNNOutput(inputImage):
    print("dnn started")
    # print(str(datetime.now()))
    model = load_model('saved_models/3_2')
    dnnOutput = 0

    image = cv2.imread(inputImage)    
    image = cv2.resize(image, (49,49)).copy()

    a,b,c = image.shape
    image = image.reshape(1,a,b,c)
    image = image.astype(np.float32) / 255.0
    image2 = tf.convert_to_tensor(image)
    dnnOutput  = np.argmax(model.predict(image2))

    print("dnnOutput = ", dnnOutput)
    return dnnOutput 






def getDNNOutput_onnx(inputImage):
    
    print("dnn started")
    print(str(datetime.now()))
    model = onnx.load('OGmodel_pb_converted.onnx')


    dnnOutput = 1

    image = cv2.imread(inputImage)    
    image = cv2.resize(image, (49, 49)).copy()

    a, b, c = image.shape
    image = image.reshape(1, a, b, c)
    print(image.shape)
    # print(image[0][0])

    image = image.astype(np.float32)  / 255.0
    # image2 = tf.convert_to_tensor(image)

    session = onnxruntime.InferenceSession('OGmodel_pb_converted.onnx')
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(input_name)
    print(output_name)

    result = session.run([output_name], {input_name: image})
    print(result)
    dnnOutput  = np.argmax(np.array(result).squeeze(), axis=0)

    print("dnnOutput = ", dnnOutput)
    return dnnOutput 

# getDNNOutput("images/minmax_0.ppm")
# getDNNOutput("images/minmax_1.ppm")


# getDNNOutput("images/collisionImage_00.ppm")

# print("\n\n\n")
# getDNNOutput_onnx("images/collisionImage_00.ppm")


getDNNOutput("images/collisionImage_00.ppm")

print("\n\n\n")
getDNNOutput_onnx("images/collisionImage_00.ppm")