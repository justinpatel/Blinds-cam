import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import time

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("../blind's specs/object_detection/")

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class describe():
  def __init__(self, PATH_TO_LABELS, MODEL_FILE, NUM_CLASSES, PATH_TO_CKPT):
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    self.category_index = label_map_util.create_category_index(categories)
    self.detection_graph = tf.Graph()
    with self.detection_graph.as_default():
      od_graph_def = tf.compat.v1.GraphDef()
      with tf.compat.v1.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
      file_name = os.path.basename(file.name)
      if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())
                
  def detect(self,cap, engine):
    frame_count = 2
    timeout = 20
    timeout_start = time.time()
    #engine.speak('detecting')

    '''(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)  < 3 :
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = cap.get(cv2.CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))'''

    # Number of frames to capture
    #num_frames = 120;
    #print "Capturing {0} frames".format(num_frames)
 
    
    with self.detection_graph.as_default():
      with tf.compat.v1.Session(graph=self.detection_graph) as sess:
        while time.time() < timeout_start+timeout:
          #print(self.frame_count)
          frame_count += 1
          ret, image_np = cap.read()
          img = cv2.flip(image_np, 1)
          (H, W) = (640, 480)
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(img, axis=0)
          image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
          # Each score represent how level of confidence for each of the objects.
          # Score is shown on the result image, together with the class label.
          scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
          classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
          # Actual detection.
          (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor : image_np_expanded})
          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
            img,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

          text = []
          detected = []
          for i, b in enumerate(boxes[0]):
            if classes[0][i] == 44 or classes[0][i] == 1:
              if scores[0][i] > 0.5:
                if frame_count % 1 == 0:
                  label = self.category_index[classes[0][i]].get("name")
                  if label not in detected:                    

                    mid_x = ((boxes[0][i][3] + boxes[0][i][1]) /2)*W
                    mid_y = ((boxes[0][i][2] + boxes[0][i][0]) /2)*H

                    W_pos,H_pos = '',''

                    if mid_x <= W/3:
                      W_pos = "right"
                    elif mid_x <= (W/3 *2):
                      W_pos = "center"
                    else:
                      W_pos = "left"

                    if mid_y <= H/3:
                      H_pos = "top"
                    elif mid_y <= (H/3 *2):
                      H_pos = "mid"
                    else:
                      H_pos = "bottom"
                    #text.append(H_pos+" "+ W_pos+" "+ label)
                    detected.append(label)
                    
                    #apx_distance = round((1-(boxes[0][i][3] - boxes[0][i][0])),1)
                    #cv2.putText(image_np,'hh{}'.format(apx_distance), (int(mid_x), int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    engine.speak(label+ " is at "+H_pos+" "+ W_pos)
                    #print(text)
          
          #print(frame_count)
          '''resp = engine.recognize_speech_from_mic()
          if resp == 'stop':
            cv2.destroyAllWindows()
            cap.release()
            break'''
          cv2.imshow('object detection',  cv2.resize(img, (800,600)))

          '''if frame_count == 50:
            cv2.destroyAllWindows()
            cap.release()
            break'''
          
          #return boxes, scores, classes
          #resp = engine.recognize_speech_from_mic()
          '''if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break'''

  def search(self, cap, engine, obj):
    frame_count = 0
    found = False
    with self.detection_graph.as_default():
      with tf.compat.v1.Session(graph=self.detection_graph) as sess:
        while True:
          #print(self.frame_count)
          frame_count += 1
          ret, image_np = cap.read()
          img = cv2.flip(image_np, 1)
          (H, W) = (640, 480)
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(img, axis=0)
          image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
          # Each score represent how level of confidence for each of the objects.
          # Score is shown on the result image, together with the class label.
          scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
          classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
          # Actual detection.
          (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor : image_np_expanded})
          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
            img,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

          text = []
          detected = []
          
          for i, b in enumerate(boxes[0]):
            if self.category_index[classes[0][i]].get("name") ==  obj:
              if scores[0][i] > 0.5:
                #if frame_count % 5 == 0:
                found = True
                label = self.category_index[classes[0][i]].get("name")
                  #if label not in detected:                    

                mid_x = ((boxes[0][i][3] + boxes[0][i][1]) /2)*W
                mid_y = ((boxes[0][i][2] + boxes[0][i][0]) /2)*H

                W_pos,H_pos = '',''

                if mid_x <= W/3:
                  W_pos = "left"
                elif mid_x <= (W/3 *2):
                  W_pos = "center"
                else:
                  W_pos = "right"

                if mid_y <= H/3:
                  H_pos = "top"
                elif mid_y <= (H/3 *2):
                  H_pos = "mid"
                else:
                  H_pos = "bottom"
                #text.append(H_pos+" "+ W_pos+" "+ label)
                #detected.append(label)
                    
                    #apx_distance = round((1-(boxes[0][i][3] - boxes[0][i][0])),1)
                    #cv2.putText(image_np,'hh{}'.format(apx_distance), (int(mid_x), int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                engine.speak(obj+" is found at "+H_pos+" "+ W_pos)
                    #print(text)
          
          #print(frame_count)
         
          cv2.imshow('object detection',  cv2.resize(img, (800,600)))

          if frame_count == 50:
            if not found:
              engine.speak("Sorry could not find what you are looking for")
            cv2.destroyAllWindows()
            cap.release()
            break
          
          #return boxes, scores, classes
          #resp = engine.recognize_speech_from_mic()
          if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break
    
    
          
        
