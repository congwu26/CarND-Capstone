from styx_msgs.msg import TrafficLight

import rospy
import tensorflow as tf
import os
import cv2
import numpy as np

from keras.models import load_model

#pretrained tf model for object generation
object_detection_graph_model = "frozen_inference_graph.pb"
traffic_light_classificaiton_model = "model.h5"
TRAFFIC_LIGHT_LABEL = 10

class TLClassifier(object):
    def __init__(self):
        # load classifier
        current_path = os.path.dirname(os.path.realpath(__file__))
        self.model = load_model(current_path + '/' + traffic_light_classificaiton_model, compile=False)
        self.class_graph =  tf.get_default_graph()

        # load object detection graph
        self.object_detection_graph = tf.Graph()
        with self.object_detection_graph.as_default():
            gdef = tf.GraphDef()
            with open(current_path + '/' + object_detection_graph_model, 'rb') as f:
                gdef.ParseFromString( f.read() )
                tf.import_graph_def( gdef, name="" ) 
            self.session = tf.Session(graph=self.object_detection_graph )
            self.image_tensor = self.object_detection_graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.object_detection_graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.object_detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.object_detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.object_detection_graph.get_tensor_by_name('num_detections:0')



    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        traffic_light_image = self.get_traffic_light_image(image)
        if (traffic_light_image is None):
            return TrafficLight.UNKNOWN

        #resize to model input
        traffic_light_image = cv2.resize(traffic_light_image, (48,96))
        img_resize = np.expand_dims(traffic_light_image, axis=0)
        
        #prediction
        with self.class_graph.as_default():
            prediction = np.argmax(self.model.predict(img_resize, batch_size=1))
        
        return prediction

    
    def get_traffic_light_image(self, image):
        with self.object_detection_graph.as_default():
            #switch from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            tf_image_input = np.expand_dims(image,axis=0)

            #run detection model
            (detection_boxes, detection_scores, detection_classes, num_detections) = \
                self.session.run([self.boxes, self.scores, self.classes, self.num_detections],
                                 feed_dict={self.image_tensor: tf_image_input})
            detection_boxes = np.squeeze(detection_boxes)
            detection_classes = np.squeeze(detection_classes)
            detection_scores = np.squeeze(detection_scores)

            tf_image = None
            detection_threshold = 0.2

            idx = -1
            for i, cl in enumerate(detection_classes.tolist()):
                if cl == TRAFFIC_LIGHT_LABEL:
                    idx = i
                    break

            if idx == -1 or detection_scores[idx] < detection_threshold:
                return None # traffic ligth wasn't found
            else:
                #transform to image coordinates
                img_shape = image.shape[0:2]
                detected_box = detection_boxes[idx]
                img_box = np.zeros_like(detected_box)
                img_box[0] = np.int(detected_box[0] * img_shape[0])
                img_box[1] = np.int(detected_box[1] * img_shape[1])
                img_box[2] = np.int(detected_box[2] * img_shape[0])
                img_box[3] = np.int(detected_box[3] * img_shape[1])
                img_box = img_box.astype(int)
                return image[img_box[0]:img_box[2], img_box[1]:img_box[3]]

