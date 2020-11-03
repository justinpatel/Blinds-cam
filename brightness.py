import cv2
import numpy as np

def Tell_surrounding(cam, engine):
    ret, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg = np.sum(frame)/(frame.shape[0]*frame.shape[1])
    avg=avg/255
    if(avg > 0.6):
        #print ("Very bright", avg)
        engine.speak("There is very bright surrounding")
    elif(avg > 0.4):
        #print ("Bright", avg)
        engine.speak("There is bright surrounding")
    elif(avg>0.2 and avg<0.4):
        #print ("Dim", avg)
        engine.speak("There is dim surrounding")
    else:
        #print ("Dark",avg)
        engine.speak("There is dark surrounding")

