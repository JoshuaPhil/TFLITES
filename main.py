import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
import numpy as np
from math import floor
import cv2 as cv

Number = int | float

#Tracking variables and constants.
running : bool = False
FRAME_RATE : Number = 13
DELAY_IN_MS : Number = floor(1000/FRAME_RATE)
TEMP_FILENAME : str = "temp.png"



running : Number = 0

text_detection = tf.lite.Interpreter(model_path="1.tflite")
text_detection.allocate_tensors()

text_detection_input_details = text_detection.get_input_details()
text_detection_output_details = text_detection.get_output_details()


i_data = cv.imread("temp.png")


im = cv.cvtColor(i_data, cv.COLOR_BGR2RGB)

model_size = (320, 320)
resized = cv.resize(im, model_size, interpolation=cv.INTER_CUBIC)
resized = resized.astype(np.float32)
resized /= 255

i_data = np.expand_dims(resized, 0)

text_detection.set_tensor(text_detection_input_details[0]['index'], i_data)

text_detection.invoke()
 
text_detection_output = text_detection.get_tensor(text_detection_output_details[0]['index'])

print(text_detection_output)



cv.imshow("output", text_detection_output)

text_recognition = tf.lite.Interpreter(model_path="2.tflite")     
text_recognition.allocate_tensors()

text_recognition_input_details = text_recognition.get_input_details() 
text_recognition_output_details = text_recognition.get_output_details()

text_recognition.set_tensor(text_recognition_input_details[0]['index'], text_detection_output)

text_recognition.invoke()

text_recognition_output = text_recognition.get_tensor(text_recognition_output_details[0]['index'])

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
while running:
    frame = None
 
    if camera.isOpened():
        rval, frame = camera.read()
        assert rval==True, "Camera is not plugged in."
    
    #Get the value of the key that is pressed. 
    keyValue : Number = cv.waitKey(DELAY_IN_MS)
    
    #Determine if the escape key is pressed.
    if keyValue == 27:
        #stop the program
        stop()
camera.release()
cv.destroyAllWindows() 