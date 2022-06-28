import os
import cv2
import numpy as np
from gtts import *

class MaskRCNN:
    def __init__(self):
        # Loading Mask RCNN
        self.net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
                                            "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

        # Generate random colors
        np.random.seed(2)
        self.colors = np.random.randint(0, 255, (90, 3))

        # Conf threshold
        self.detection_threshold = 0.7
        self.mask_threshold = 0.3

        self.classes = []
        with open("dnn/classes.txt", "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        self.obj_boxes = []
        self.obj_classes = []
        self.obj_centers = []
        self.obj_contours = []
        self.distances = []


    def detect_objects_mask(self, bgr_frame):
        blob = cv2.dnn.blobFromImage(bgr_frame, swapRB=True)
        self.net.setInput(blob)

        boxes, masks = self.net.forward(["detection_out_final", "detection_masks"])

        # Detect objects
        frame_height, frame_width, _ = bgr_frame.shape
        detection_count = boxes.shape[2]

        self.obj_boxes = []
        self.obj_classes = []
        self.obj_centers = []
        self.obj_contours = []

        for i in range(detection_count):
            box = boxes[0, 0, i]
            class_id = box[1]
            score = box[2]
            if score < self.detection_threshold:
                continue


            x = int(box[3] * frame_width)
            y = int(box[4] * frame_height)
            x2 = int(box[5] * frame_width)
            y2 = int(box[6] * frame_height)
            self.obj_boxes.append([x, y, x2, y2])
            cx = (x + x2) // 2
            cy = (y + y2) // 2
            #define object center
            self.obj_centers.append((cx, cy))
            # append class
            self.obj_classes.append(class_id)
            mask = masks[i, int(class_id)]
            roi_height, roi_width = y2 - y, x2 - x
            mask = cv2.resize(mask, (roi_width, roi_height))
            _, mask = cv2.threshold(mask, self.mask_threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.obj_contours.append(contours)
        return self.obj_boxes, self.obj_classes, self.obj_contours, self.obj_centers
    def draw_object_mask(self, bgr_frame):
        # loop through the detection
        for box, class_id, contours in zip(self.obj_boxes, self.obj_classes, self.obj_contours):
            x, y, x2, y2 = box
            roi = bgr_frame[y: y2, x: x2]
            roi_height, roi_width, _ = roi.shape
            color = self.colors[int(class_id)]
            roi_copy = np.zeros_like(roi)
            for cnt in contours:
                cv2.drawContours(roi, [cnt], - 1, (int(color[0]), int(color[1]), int(color[2])), 3)
                cv2.fillPoly(roi_copy, [cnt], (int(color[0]), int(color[1]), int(color[2])))
                roi = cv2.addWeighted(roi, 1, roi_copy, 0.5, 0.0)
                bgr_frame[y: y2, x: x2] = roi
        return bgr_frame
    def draw_object_info(self, bgr_frame, depth_frame):
        # loop through the detection
        for box, class_id, obj_center in zip(self.obj_boxes, self.obj_classes, self.obj_centers):
            x, y, x2, y2 = box

            color = self.colors[int(class_id)]
            color = (0,255,0)
            cx, cy = obj_center
            c1x = (cx + x)/2
            c2x = (cx + x2)/2
            # find the possotion of depthframe where the object located
            depth_mm = depth_frame[cy, cx]
            # setup text to speech
            langu = 'en'
            filename = "hello.mp3"
            #text to speech funtion
            def t_to_s():
                dist = int(depth_mm/10)
                audio = gTTS(text = class_name + str(dist) +'Centimeter', lang = langu,slow= False)
                audio.save(filename)
                os.system(f'start {filename}')
            # filtering object by detection area
            if (560<x<720 or 560<x2<720 or 560<(cx+x)/2<720 or 560<(cx+x2)/2<720 or 560< cx<720) and depth_mm<5000:
            #filtering object by controling dept
                class_name = self.classes[int(class_id)]
                cv2.rectangle(bgr_frame, (x, y), (x + 250, y + 70), color, -1)
                cv2.putText(bgr_frame, class_name.capitalize(), (x + 5, y + 25), 0, 0.8, (255, 255, 255), 2)

                cv2.putText(bgr_frame, "{} cm".format(depth_mm / 10), (x + 5, y + 60), 0, 1.0, (255, 255, 255), 2)
                cv2.rectangle(bgr_frame, (x, y), (x2, y2), color, 1)
                # print information and call function to convert text to speech
                print(class_name + str(depth_mm/1000)+'Meter')
                t_to_s()
            else:
                return False, None, None
        return bgr_frame



