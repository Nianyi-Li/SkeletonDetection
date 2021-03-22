# this module contains code for extracting relevant pose
# information that can be used for mask detection
import math
import cv2
import numpy as np
from natsort import natsorted

import pose_extract as pe


def readYolov4(labelsPath, weightsPath, configPath):
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    return net


def processJsonForFaces(json_path, image_dir, net, conf_thresh):
    data_dict = pe.get_pose_info(image_dir, json_path, compute_box_size=True)
    sortedKeys = natsorted(data_dict.keys())

    for key in sortedKeys:
        im_dict = data_dict[key]
        print(im_dict["full_im_path"])
        img = cv2.imread(im_dict["full_im_path"], cv2.IMREAD_COLOR)
        (H, W) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (608, 608), swapRB=True, crop=False)
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        net.setInput(blob)
        layerOutputs = net.forward(ln)  # list of 3 arrays, for each output layer

        confidences = []
        people = im_dict['people']
        for kk in range(len(people)):
            person = people[kk]
            if person['valid_sub_im']:
                person_classID = "unknown"
                head = person['head_pos']
                headx = head[0]
                heady = head[1]
                max_conf = 0
                for output in layerOutputs:
                    for detection in output:
                        # extract the class ID and confidence (i.e., probability) of
                        # the current object detection
                        scores = detection[5:]  # last 2 values in vector
                        classID = np.argmax(scores)
                        confidence = scores[classID]
                        # filter out weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        center_x = int(detection[0] * W)
                        center_y = int(detection[1] * H)
                        if confidence > conf_thresh and math.fabs(center_x - headx) < 30 and math.fabs(
                                center_y - heady) < 30:
                            if classID == 0 and confidence > max_conf:
                                person_classID = "no_mask"
                                max_conf = confidence
                            if classID == 1 and confidence > max_conf:
                                person_classID = "mask"
                                max_conf = confidence
                person['is_wearing_mask'] = person_classID
    return data_dict
