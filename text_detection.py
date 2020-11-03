import os
import io
from google.cloud import vision
from google.cloud.vision import types
import cv2
import time

def text_detect(cam, engine):

    client = vision.ImageAnnotatorClient()

    #credentials = service_account.Credentials.from_service_account_file('googleVisionAPI.json')
    #client = vision.ImageAnnotatorClient(credentials=credentials)

    engine.speak('Ready to detect text in 5 seconds')
    timeout = 5
    timeout_start = time.time()
    
    while time.time() < timeout_start+timeout:
        ret, img = cam.read()    
        #cv2.imshow("Capturing", img)
        cv2.imwrite('test\image.jpg', img)

    #engine.speak('saving photo')
    
    cv2.destroyAllWindows()
    cam.release()
    engine.speak('The detected text is: ')  
    with io.open('test\image.jpg', 'rb') as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    #print(len(texts))
    #print('Text:')
    if (len(texts)!=0):
        print(texts[0].description)
        engine.speak(texts[0].description)
    else:
        engine.speak("couldn't find text")
    '''textm = ""
    for i, text in enumerate(texts):
        #print(text.description)
        textm += text.description
        textm = textm + " "
    
    print(textm) '''
