#USE THIS VERSION OF MAIN


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
import numpy as np
from math import floor
import cv2 as cv
import string

#alphabet for ocr
alphabets = string.digits+string.ascii_lowercase
blank_index = len(alphabets)

#function for text detection inferrence
def detect_text(mat):
    #create a copy of the matrix for changing
    idata = mat
    #get height and width of matrix
    (imageHeight, imageWidth) = idata.shape[:2]

    #set new height and width for the model
    (newImageHeight, newImageWidth) = (320, 320)
    #calculate scale factor for both width and height
    ratioWidth = imageWidth / newImageWidth
    ratioHeight = imageHeight / newImageHeight
    
    #resize the image based on the new width and height
    idata = cv.resize(idata, (newImageWidth, newImageHeight))
    #store new dimensions into variables
    (imageHeight, imageWidth) = idata.shape[:2]

    #change the cast to float32
    idata = idata.astype("float32")
    #take the mean of the following array
    mean = np.array([123.68, 116.779, 103.99][::-1], dtype="float32")
    #subtract values in the matrix by the mean
    idata -= mean
    idata = np.expand_dims(idata, 0)
    #return new matrix, orginal and the scale factors on both dimensions
    return idata, mat, ratioWidth, ratioHeight

#function for text recognition inference
def recognise_text(mat):
    #create copy of matrix for manipulation
    idata = mat
    #resize the image for inference
    idata = cv.resize(idata, (200, 31))
    idata = idata[np.newaxis]
    idata = np.expand_dims(idata, 3)
    #normalize values in the matrix
    idata = idata.astype('float32')/255
    #load tflite
    interpreter = tf.lite.Interpreter(model_path="textRecognition.tflite")
    interpreter.allocate_tensors()

    #get io details of the model
    idetails = interpreter.get_input_details()
    odetails = interpreter.get_output_details()

    ishape = idetails[0]['shape']
    #enter information into model
    interpreter.set_tensor(idetails[0]['index'], idata)

    #execute model
    interpreter.invoke()

    #get output from model
    output = interpreter.get_tensor(odetails[0]['index'])
    return output

#image name
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
    
#write found text to file
def write_to_file(txt):
    file = open("found_text.txt", "a")
    file.write(txt+"\n")
    file.close()

#get webcam, change number in constructor if wrong camera is used
camera = cv.VideoCapture(0)

#Start the program
start()

#when running is true
while running:
    frame = None 
    
    #read from camera feed and use input for inference
    if camera.isOpened():
        
        rval, frame = camera.read()
        assert rval==True, "Camera is not plugged in"

        #onvert frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #create a temporary file
        cv.imwrite(TEMP_FILENAME, gray)

        #infer text
        tfloutput = recognise_text(gray)
        foutput = "".join(alphabets[index] for index in tfloutput[0] if index not in [blank_index, -1])
        print(foutput)

        #print output and write to file
        print("Interpreted Text:", foutput)
        write_to_file(foutput)

        #show frames
        cv.imshow("Camera Feed", frame)
        cv.imshow("Output", gray)

    #Get the value of the key that is pressed.
    keyValue : int = cv.waitKey(DELAY_IN_MS)
    
    #Determine if the escape key is pressed.
    if keyValue == 27:
        #stop the program
        stop()
#close program
camera.release()
cv.destroyAllWindows()