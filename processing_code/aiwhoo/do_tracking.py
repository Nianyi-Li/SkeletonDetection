# this module contains code for extracting relevant pose
# information that can be used for mask detection
import pdb
import math
from natsort import natsorted
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pose_extract as pe
from PIL import Image
import norfair
from norfair import Detection, Tracker, Video, draw_tracked_objects
import sys


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


def doTracking(data_dict, first_id):
    sortedKeys = natsorted(data_dict.keys())
    tracker = Tracker(distance_function=euclidean_distance, distance_threshold=700, point_transience=1,
                      hit_inertia_min=1, hit_inertia_max=75, init_delay=25)
    max_id = first_id
    first_frame = 0
    last_frame = 0
    if(len(sortedKeys) > 0):
        first_frame = int(sortedKeys[0].split('.')[0])
        last_frame = int(sortedKeys[-1].split('.')[0])
    for ii in range(first_frame, last_frame + 1):
        curr_key = '{0:05d}'.format(ii) + '.jpg'
        detections = []
        if curr_key in sortedKeys:
            im_dict = data_dict[curr_key]
            cv2.imread(im_dict["full_im_path"])
            people = im_dict['people']
            np.zeros((len(people), 2))
            for kk in range(len(people)):
                person = people[kk]
                if person['valid_sub_im']:
                    center = np.array(person['head_pos'])
                    detections.append(Detection(center))
            tracked_objects = tracker.update(detections=detections)
            # draw_tracked_objects(img, tracked_objects)
            people = im_dict['people']
            for kk in range(len(people)):
                person = people[kk]
                person['ID'] = -1

            sz = max(len(people), len(tracked_objects))
            all_dists = np.ones((sz,sz))*math.inf
            for kk in range(len(people)):
                person = people[kk]
                c = np.array(person['head_pos'])
                if(person['valid_sub_im'] == True):
                    for tt in range(len(tracked_objects)):
                        tracked_object = tracked_objects[tt]
                        ct = tracked_object.estimate
                        distance = math.sqrt(((c[0] - ct[0][0]) ** 2) + ((c[1] - ct[0][1]) ** 2))
                        all_dists[kk,tt] = distance

            for kk in range(len(people)):
                min_overall = np.amin(all_dists)
                if(min_overall == math.inf or min_overall > 75):
                    break
                min_idxs = np.where(all_dists == np.amin(all_dists))
                try:
                    min_person = int(min_idxs[0])
                    min_tracked_obj = int(min_idxs[1])
                    person = people[min_person]
                    all_dists[:, min_tracked_obj] = math.inf
                    all_dists[min_person, :] = math.inf
                    tracked_object = tracked_objects[min_tracked_obj]
                    person['ID'] = first_id + tracked_object.id - 1
                    if max_id < person['ID']:
                        max_id = person['ID']
                except:
                    print('No min dists? Skipping')


        else:
            tracker.update(detections=detections)
    return data_dict, max_id


def parseTrackingDict(data_dict, first_id, last_id, fps):
    SCOTT_JANK_FACTOR = 1.0
    outputDict = {}
    sortedKeys = natsorted(data_dict.keys())
    final_frame_in_keys = 0
    if(len(sortedKeys) > 0):
        final_frame_in_keys = int((sortedKeys[-1]).split(".")[0])
    for person_id in range(first_id, last_id + 1, 1):
        closest_distance = 0.0
        closest_distance_min = float('inf')
        avg_height = 0.0
        avg_speed = 0
        is_wearing_mask = 0.0
        count = 0
        count_dist = 0
        count_height = 0
        count_speed = 0.0
        prev_pos = []
        curr_pos = []
        curr_frame = -1
        prev_frame = -1
        first_frame = 0
        last_frame = 0
        for key in sortedKeys:
            im_dict = data_dict[key]
            people = im_dict['people']
            frame = int(key.split(".")[0])
            for kk in range(len(people)):
                person = people[kk]
                if (person['valid_sub_im'] is True) and (person['ID'] == person_id):
                    curr_pos = person['head_pos_3d']
                    curr_frame = frame
                    if(prev_frame == -1):
                        prev_frame = curr_frame
                        prev_pos = curr_pos
                    if (len(prev_pos) == 3 and len(curr_pos) == 3 and curr_frame > 0 and prev_frame > 0 and
                            (curr_frame - prev_frame > 10 or curr_frame == final_frame_in_keys)):
                        pos_dist = math.dist(curr_pos[0:2], prev_pos[0:2])
                        frame_diff = curr_frame - prev_frame

                        prev_frame = curr_frame
                        prev_pos = curr_pos
                        meters_per_second = 0
                        if(frame_diff > 0):
                            meters_per_second = (pos_dist / (frame_diff / 30.0)) / 100.0 / SCOTT_JANK_FACTOR
                        if (meters_per_second > 0 and meters_per_second < 10):
                            count_speed = count_speed + 1
                            avg_speed = avg_speed + meters_per_second
                    if (first_frame == 0):
                        first_frame = int(key.split(".")[0])
                    last_frame = int(key.split(".")[0])
                    min_dist = person['min_dist']
                    if min_dist > 0.1 and min_dist < 2000:
                        count_dist = count_dist + 1
                        closest_distance = closest_distance + (person['min_dist'] / 100.0)
                        closest_distance_min = min(closest_distance_min, (person['min_dist'] / 100.0))
                    if person['is_wearing_mask'] == 'mask' or person['is_wearing_mask'] == "no_mask":
                        is_wearing_mask = is_wearing_mask + int(person['is_wearing_mask'] == "mask")
                        count = count + 1
                    ht = person['height']
                    if (ht > 0 and ht < 2000):
                        avg_height = avg_height + ht
                        count_height = count_height + 1

        if (count_speed > 0):
            avg_speed = (avg_speed / count_speed)
        else:
            avg_speed = 1

        if (count_dist > 0):
            closest_distance = closest_distance / count_dist  # in meters
        else:
            closest_distance = float('inf')
            closest_distance_min = float('inf')
        if count > 5:
            is_wearing_mask = bool(round(is_wearing_mask / count))
        else:
            is_wearing_mask = "unknown"
        if count_height > 0:
            avg_height = (avg_height / count_height) / 100.0  # in meters
        else:
            avg_height = 168.4 / 100.0  # in meters
        row = {'closest_distance_avg': closest_distance, 'closest_distance_min': closest_distance_min,
               'is_wearing_mask': is_wearing_mask, "first_frame": first_frame, "last_frame": last_frame,
               "height": avg_height, "avg_speed": avg_speed}
        outputDict[person_id] = row

    return outputDict
