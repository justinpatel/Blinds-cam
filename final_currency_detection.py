import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2
import time

# Load the model
model = tensorflow.keras.models.load_model('model\keras_currency_model.h5', compile=False)
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

def currency_detect(cam, engine):
    
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    engine.speak('Detecting currency in 5 seconds')
    timeout = 5
    timeout_start = time.time()
    while time.time() < timeout_start+timeout:
        ret, img = cam.read()    
        #cv2.imshow("Capturing", img)
        cv2.imwrite('test\currency_detect.jpg', img)

    #engine.speak('photo captured')
    
    cv2.destroyAllWindows()
    cam.release()
    engine.speak('detected currency banknote')
    
    # Replace this with the path to your image
    image = Image.open('test\currency_detect.jpg')

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    #image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    CATEGORIES = ["10","20","50","100","200","500","2000","0"]

    # run the inference
    prediction = model.predict(data)

    max_val = 0
    index = 0
    
    for i in range(0, len(prediction[0])):
        
        if prediction[0][i] > max_val:
            index = i
            max_val = prediction[0][i]

    #print(CATEGORIES[index])
    if CATEGORIES[index] == "0":
        engine.speak("Sorry Couldnt find any banknote")
    else:
        print("it is a "+ CATEGORIES[index] +" rupee note")
        engine.speak("it is a "+ CATEGORIES[index] +" rupee note")
    #print(prediction)
