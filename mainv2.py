import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
import numpy as np
from math import floor
import cv2 as cv
import string

alphabets = string.digits+string.ascii_lowercase
blank_index = len(alphabets)
 
def detect_text(mat):
    idata = mat
    (imageHeight, imageWidth) = idata.shape[:2]

    (newImageHeight, newImageWidth) = (320, 320)
    ratioWidth = imageWidth / newImageWidth
    ratioHeight = imageHeight / newImageHeight
    
    idata = cv.resize(idata, (newImageWidth, newImageHeight))
    (imageHeight, imageWidth) = idata.shape[:2]

    idata = idata.astype("float32")
    mean = np.array([123.68, 116.779, 103.99][::-1], dtype="float32")
    idata -= mean
    idata = np.expand_dims(idata, 0)
    return idata, mat, ratioWidth, ratioHeight

def recognise_text(mat):
    idata = mat
    idata = cv.resize(idata, (200, 31))
    idata = idata[np.newaxis]
    idata = np.expand_dims(idata, 3)
    idata = idata.astype('float32')/255
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


#Tracking variables and constants.
running : bool = False
FRAME_RATE : int = 10
DELAY_IN_MS : int = floor(1000/FRAME_RATE)
TEMP_FILENAME = "temp.png" 

#Safe to call twice.
#Starts the program.
def start():
    global running
    if not running: 
        running = True
    else:
        pass

#Safe to call twice.
#Stops the program.
def stop():
    global running
    if running:                                                        
        running = False
    else: 
        pass
    

def write_to_file(txt):
    file = open("found_text.txt", "a")
    file.write(txt+"\n")
    file.close()

camera = cv.VideoCapture(0)

#Start the program
start()

while running:
    frame = None 
  
    if camera.isOpened():
        
        rval, frame = camera.read()
        assert rval==True, "Camera is not plugged in"
    
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imwrite(TEMP_FILENAME, gray)

        tfloutput = recognise_text(gray)
        foutput = "".join(alphabets[index] for index in tfloutput[0] if index not in [blank_index, -1])
        print(foutput)


        print("Interpreted Text:", foutput)
        write_to_file(foutput)

        cv.imshow("Camera Feed", frame)
        cv.imshow("Output", gray)

    #Get the value of the key that is pressed.
    keyValue : int = cv.waitKey(DELAY_IN_MS)
    
    #Determine if the escape key is pressed.
    if keyValue == 27:
        #stop the program
        stop()
camera.release()
cv.destroyAllWindows()