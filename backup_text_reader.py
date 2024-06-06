from PIL import Image
from math import floor
import cv2 as cv
import pytesseract
import os

#Path to tesseract program.
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

#Tracking variables and constants.
running : bool = False
FRAME_RATE : int = 13
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

        image = Image.open(TEMP_FILENAME)
        text = pytesseract.image_to_string(image)
        image.close()
        os.remove(TEMP_FILENAME)

        print("Interpreted Text:", text)
        write_to_file(text)


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
