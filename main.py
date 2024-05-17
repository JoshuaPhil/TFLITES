import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
import numpy as np
from math import floor
import cv2 as cv

#Tracking variables and constants.
running : bool = False
FRAME_RATE : int = 30
DELAY_IN_MS : int = floor(1000/FRAME_RATE)
TEMP_FILENAME = "temp.png"


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


camera = cv.VideoCapture(0)
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
