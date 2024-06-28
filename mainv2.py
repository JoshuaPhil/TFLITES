import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
import numpy as np
from math import floor
import cv2 as cv
import string

alphabets = string.ascii_letters+string.digits
blank_index = len(alphabets)

def run_tflite_model(image_path, quantization):
    idata = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    idata = cv.resize(idata, (200, 31))
    idata = idata[np.newaxis]
    idata = idata.astype('float32')/255
    path = f'ocr_{quantization}.tflite'
    interpreter = tf.lite.Interpreter(model_path="textRecognition.tflite")
    interpreter.allocate_tensors()

    idetails = interpreter.get_input_details()
    odetails = interpreter.get_output_details()

    ishape = idetails[0]['shape']
    interpreter.set_tensor(idetails[0]['index'], idata)

    interpreter.invoke()

    output = interpreter.get_tensor(odetails[0]['index'])
    return output

image_path = "temp.png"

tfloutput = run_tflite_model(image_path, 'float16')
foutput = "".join(alphabets[index] for index in tfloutput[0] if index not in [blank_index, -1])
print(foutput)
cv.imshow(cv.imread(image_path))