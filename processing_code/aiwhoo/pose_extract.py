# this module contains code for extracting relevant pose
# information that can be used for mask detection
import pdb
import json
import os
import math
import social_distance
from PIL import Image
import sys
import numpy as np
import csv




# finds the minimum distance this person is to any other person
def get_min_dist(index, matrix):
    min_dist = sys.float_info.max
    closest = -1
    for i in range(matrix.shape[0]):
        if i < index:
            dist = matrix[i, index]
            if dist < min_dist:
                min_dist = dist
                closest = i
        elif i > index:
            dist = matrix[i, index]
            if dist < min_dist:
                min_dist = dist
                closest = i
    return min_dist, closest


# network outputs 832×512. need to scale to fit right
def scale_2d(img):
    xfrac = img.width / 832
    yfrac = img.height / 512
    return xfrac, yfrac


def apply_image_scale(point, scalex, scaley):
    return [int(point[0] * scalex), int(point[1] * scaley)]


# get the sub image, and center point of the head.
# pose point of every point on skeleton not always returned
def head_sub_im(im, pose_2d, box_size, scalex, scaley):
    head = apply_image_scale(pose_2d[1], scalex, scaley)
    neck = apply_image_scale(pose_2d[0], scalex, scaley)
    l_shoulder = apply_image_scale(pose_2d[3], scalex, scaley)
    r_shoulder = apply_image_scale(pose_2d[9], scalex, scaley)

    width = im.width
    height = im.height
    pix_scale = get_pixel_head_scale(pose_2d, scalex, scaley)
    proportioned = pix_scale > 0

    cent = [0, 0]
    sub_im = None
    if non_zero(head):
        box = get_box(list(np.average([head] + [neck], axis=0)), width, height, box_size)
        cent = list(np.average([head] + [neck], axis=0))
        sub_im = im.crop((box[0], box[1], box[2], box[3]))
        head = cent
    elif non_zero(neck) and proportioned:
        cent = [neck[0], int(round(neck[1] + .5 * pix_scale))]  # sort of assumes person is upright
        box = get_box(cent, width, height, box_size)
        sub_im = im.crop((box[0], box[1], box[2], box[3]))
        head = cent
    elif (non_zero(l_shoulder) and non_zero(r_shoulder)) and proportioned:
        diff = [l_shoulder[0] - r_shoulder[0], l_shoulder[1] - r_shoulder[1]]
        new_neck = [round(l_shoulder[0] + 0.5 * diff[0]), round(l_shoulder[1] + 0.5 * diff[1])]
        cent = [new_neck[0], int(round(new_neck[1] + .5 * pix_scale))]  # sort of assumes person is upright
        box = get_box(cent, width, height, box_size)
        sub_im = im.crop((box[0], box[1], box[2], box[3]))
        head = cent
    return sub_im, cent


# using human proportions from general art principles (humans are ~7.5 headlengths tall)
def get_pixel_head_scale(pose_2d, scalex, scaley):
    l_foot = apply_image_scale(pose_2d[8], scalex, scaley)
    r_foot = apply_image_scale(pose_2d[14], scalex, scaley)
    l_knee = apply_image_scale(pose_2d[7], scalex, scaley)
    r_knee = apply_image_scale(pose_2d[13], scalex, scaley)
    l_hip = apply_image_scale(pose_2d[6], scalex, scaley)
    r_hip = apply_image_scale(pose_2d[12], scalex, scaley)

    pelvis = apply_image_scale(pose_2d[2], scalex, scaley)
    neck = apply_image_scale(pose_2d[0], scalex, scaley)

    head_estimages = []
    if non_zero(l_foot) and non_zero(l_knee):
        dist = dist_2d(l_foot, l_knee)
        head_estimages.append(dist / 2.0)
    if non_zero(r_foot) and non_zero(r_knee):
        dist = dist_2d(r_foot, r_knee)
        head_estimages.append(dist / 2.0)

    if non_zero(l_hip) and non_zero(l_knee):
        dist = dist_2d(l_hip, l_knee)
        head_estimages.append(dist / 2.0)
    if non_zero(r_hip) and non_zero(r_knee):
        dist = dist_2d(r_hip, r_knee)
        head_estimages.append(dist / 2.0)

    if non_zero(l_foot) and non_zero(l_hip):
        dist = dist_2d(l_foot, l_hip)
        head_estimages.append(dist / 4.0)
    if non_zero(r_foot) and non_zero(r_hip):
        dist = dist_2d(r_foot, r_hip)
        head_estimages.append(dist / 4.0)

    if non_zero(neck) and non_zero(pelvis):
        dist = dist_2d(neck, pelvis)
        head_estimages.append(dist / 3.0)
    if len(head_estimages) > 0:
        return mean(head_estimages)
    else:
        return -1


def non_zero(pt):
    return pt[0] > 0.0 and pt[1] > 0.0


def dist_2d(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def get_box(pt, width, height, box_size):
    minx = max(0, pt[0] - box_size)
    maxx = min(width - 1, pt[0] + box_size)
    miny = max(0, pt[1] - box_size)
    maxy = min(height - 1, pt[1] + box_size)
    return [minx, miny, maxx, maxy]


def get_box_dim(poses_2d, scalex, scaley):
    default = 30
    head_sizes = []
    for person in poses_2d:
        head_size = get_pixel_head_scale(person, scalex, scaley)
        head_sizes.append(head_size)
    if len(head_sizes) > 0:
        return round(max(head_sizes) * 1.3)  # 30% padding around larges head size in image space
    else:
        return default


def mean(list_of_numbers):
    sum = 0
    for el in list_of_numbers:
        sum = sum + el
    return sum / len(list_of_numbers)


# this function extracts key information from a given image.
# Note: this expects that SMAP has already been run with results
# written in the json at json_path
def get_pose_info(image_dir, json_path, box_size=30, compute_box_size=False):
    with open(json_path, 'r') as f:
        data = json.load(f)['3d_pairs']
    extracted_dict = {}
    common_scale = social_distance.compute_common_scale(data)
    for dat in data:
        im_dict = {}
        img_file = dat['image_path']
        full_path = os.path.join(image_dir, img_file)
        im_dict['full_im_path'] = full_path
        dist_mat,heights,head_pos_3d = social_distance.compute_social_dist(dat,common_scale)
        im_dict['distance_matrix'] = dist_mat.tolist()
        im_dict['num_people'] = dist_mat.shape[0]
        im = Image.open(full_path)
        scalex, scaley = scale_2d(im)
        poses_2d = dat['pred_2d']
        if compute_box_size:
            box_size = get_box_dim(poses_2d, scalex, scaley)
        im_dict['sub_image_size'] = box_size
        people = {}
        for i in range(len(poses_2d)):
            person = {}
            min_dist, closest_person = get_min_dist(i, dist_mat)
            person['min_dist'] = min_dist
            person['closest_person'] = closest_person
            pose = poses_2d[i]
            head_im, head_pt = head_sub_im(im, pose, box_size, scalex, scaley)
            person['height'] = heights[i]
            person['head_pos_3d'] = head_pos_3d[i]
            # person['sub_im'] = head_im
            person['head_pos'] = head_pt
            person['overlap'] = False  # will be overwritten
            person['overlap_list'] = []  # will be overwritten
            if non_zero(head_pt):
                person['valid_sub_im'] = True
            else:
                person['valid_sub_im'] = False
            people[i] = person
        sorted_keys = sorted([*people.keys()])
        for i in range(len(sorted_keys)):
            key1 = sorted_keys[i]
            p1 = people[key1]
            if p1['valid_sub_im']:
                pt1 = p1['head_pos']
                for j in range(i + 1, len(sorted_keys)):
                    key2 = sorted_keys[j]
                    if key1 != key2:
                        p2 = people[key2]
                        if p2['valid_sub_im']:
                            pt2 = p2['head_pos']
                            dist = dist_2d(pt1, pt2)
                            if dist < 2 * box_size:
                                p1['overlap_list'].append(key2)
                                p2['overlap_list'].append(key1)
                                p1['overlap'] = True
                                p2['overlap'] = True
        im_dict['people'] = people
        extracted_dict[img_file] = im_dict
    return extracted_dict

# main file used for debugging. nothing to see here
# if __name__ == "__main__":
#    json_path = "/home/saponaro/Documents/SMAP/model_logs/stage3_root2/result/stage3_root2_run_inference_test_.json"
#    image_dir = "/home/saponaro/Documents/data/aquarium_small/"
#    data_dict = get_pose_info(image_dir,json_path,compute_box_size=True)
#    for key in data_dict.keys():
#    	im_dict = data_dict[key]
#    	print(im_dict["full_im_path"])

# structure
# data_dict
#  *[image_filename] <dict>
#       *['full_im_path']
#           <string> full path to image
#       *['distance_matrix']
#           <numpy array> distance matrix (between every person in image)
#       *['num_people']
#           <int> number of people in the image
#       *['sub_image_size']
#           <int> dimension of (square) sub image (same across the whole image)
#       *['people']
#           *[person index<int>]
#               ['min_dist']
#                   <float> minimum distance to anyone else in the frame (meters)
#               ['closest_person']
#                   <int> index of person they are closest to
#               ['sub_im']
#                   <PIL.Image.Image> sub image of just the person's head
#               ['head_pos']
#                   <list of 2 ints> pixel position in full image
#               ['overlap']
#                   <boolean> whether or not this subimage overlaps another
#               ['overlap_list']
#                   <list of ints> indices that overlap this sub image
#               ['valid_sub_im']
#                   <boolean> whether or not the sub image is valid (not all people have subimages)
#               ['height']
#                   <float> height of person, based on avg person height distribution
