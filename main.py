import speech
import Object_detection
import cv2
import os
import brightness
import six.moves.urllib as urllib
import tarfile
<<<<<<< HEAD
import text_detection
import emergency
import final_currency_detection
from google_vision_object_detection import vision_detect
import time

=======
#import text_detection
#import emergency
#import final_currency_detection
#from google_vision_object_detection import vision_detect
import time

# What model to use.
>>>>>>> 7384dbeefced818805d096911edf23061af49aec
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

<<<<<<< HEAD
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'


=======
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
>>>>>>> 7384dbeefced818805d096911edf23061af49aec
PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


NUM_CLASSES = 90

desc = Object_detection.describe(PATH_TO_LABELS, MODEL_FILE, NUM_CLASSES, PATH_TO_CKPT)

listening = True

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'googleVisionAPI.json'

engine = speech.speech_to_text()
engine.speak("Welcome to Blind's Spectacles")
#engine.speak("What can I help you with?")
print("Listening...")

while True:
    if listening:
        engine.speak("What can I help you with?")
        resp = engine.recognize_speech_from_mic()
        #print(resp)
        if(resp!=None):
            if "describe" in resp or "navigate" in resp:
                engine.speak("got it, detecting objects")
                cam = cv2.VideoCapture(0)
                brightness.Tell_surrounding(cam, engine)
                '''timeout = 30
                timeout_start = time.time()
                while time.time() < timeout_start+timeout:
                    objects = vision_detect(cam)
                    for object_ in objects:
                        if object_.score > 0.7:
                            print('\n{} (confidence: {})'.format(object_.name, object_.score))
                            engine.speak(object_.name)'''
                            #for vertex in object_.bounding_poly.normalized_vertices:
                                #print(' - ({},{})'.format(vertex.x, vertex.y))
                desc.detect(cam, engine)
                engine.speak('description over')
                cv2.destroyAllWindows()
                cam.release()

            elif "search" in resp or "find" in resp:
                engine.speak('What are you looking for?')
                obj = engine.recognize_speech_from_mic()
                if(obj!=None):
                    cam = cv2.VideoCapture(0)
                    desc.search(cam, engine, obj)
                engine.speak('search over')
                cv2.destroyAllWindows()
                cam.release()
<<<<<<< HEAD
                
            elif "read" in resp or "detect text" in resp:
=======

            elif "stop" in resp or "bye" in resp:
                engine.speak('Bye! take care')
                
                break
            else:
                engine.speak('Sorry, I didnt get you. Can you please repeat?')

            '''elif "read" in resp or "detect text" in resp:
>>>>>>> 7384dbeefced818805d096911edf23061af49aec
                engine.speak("got it, detecting text")
                cam = cv2.VideoCapture(0)
                text_detection.text_detect(cam, engine)
                engine.speak('Text detection over')
                cv2.destroyAllWindows()
                cam.release()

<<<<<<< HEAD
=======
            

>>>>>>> 7384dbeefced818805d096911edf23061af49aec
            elif "emergency" in resp:
                emergency.sendMsg()
                engine.speak('message sent')

<<<<<<< HEAD
=======

            

>>>>>>> 7384dbeefced818805d096911edf23061af49aec
            elif "detect currency" in  resp:
                engine.speak("got it, detecting currency")
                cam = cv2.VideoCapture(0)
                final_currency_detection.currency_detect(cam, engine)
                engine.speak('currency detection over')
                cv2.destroyAllWindows()
<<<<<<< HEAD
                cam.release() 

            elif "stop" in resp or "bye" in resp:
                engine.speak('Bye! take care')
                
                break
            else:
                engine.speak('Sorry, I didnt get you. Can you please repeat?')

                        
=======
                cam.release()  '''              
>>>>>>> 7384dbeefced818805d096911edf23061af49aec
          
                          
            
                
            
        
