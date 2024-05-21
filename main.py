import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
import numpy as np
from math import floor
import cv2 as cv

#Tracking variables and constants.
running : bool = False
FRAME_RATE : int = 13
DELAY_IN_MS : int = floor(1000/FRAME_RATE)
TEMP_FILENAME : str = "temp.png"


interpreter_1 = tf.lite.Interpreter(model_path="1.tflite")
interpreter_2 = tf.lite.Interpreter(model_path="2.tflite")
interpreter_1.allocate_tensors()

input_details_1 = interpreter_1.get_input_details()
output_details_1 = interpreter_1.get_output_details()



interpreter_2.allocate_tensors()

input_details_2 = interpreter_2.get_input_details()
output_details_2 = interpreter_2.get_output_details()

print(output_details_1)
print(input_details_2)


#Safe to call more than once.
#Starts the program.
def start():
    global running
    if not running:
        running = True
    else:
        pass

#Safe to call more than once. 
#Stops the program.
def stop():
    global running
    if running:
        running = False
    else:
        pass 



camera : cv.VideoCapture = cv.VideoCapture(0)
#Start the program
start()
while running:
    frame = None
 
    if camera.isOpened():
        rval, frame = camera.read()
        assert rval==True, "Camera is not plugged in."
    
    #Get the value of the key that is pressed.
    keyValue : int = cv.waitKey(DELAY_IN_MS)
    
    #Determine if the escape key is pressed.
    if keyValue == 27:
        #stop the program
        stop()
camera.release()
cv.destroyAllWindows()