# this file contains code for computing social distance and visualizing it  
import math
import os

import cv2
import numpy as np
from natsort import natsorted


# main function that computes social distance (3d distance between head nodes)
def compute_social_dist(entry,scale):
    poses3d = entry['pred_3d']
    heights = []
    for person in poses3d:
        height = get_height(person)
        heights.append(height)
    scaled_heights = []
    head_pos_3d = []
    for h in heights:
        scaled_heights.append(h*scale)
    head_pos_list = []
    for person in poses3d:
        head_pos = person[1]
        scaled_head = [scale*head_pos[0],scale*head_pos[1],scale*head_pos[2]]
        head_pos_3d.append(scaled_head)
        head_pos_list.append(head_pos)
    dist_mat = create_distance_matrix(head_pos_list, scale)
    return dist_mat, scaled_heights, head_pos_3d

# find a common scale across multiple frames using the height of the people in the frames
def compute_common_scale(data):
    overall_heights = []
    for dat in data:
        peeps = dat['pred_3d']
        for person in peeps:
            height = get_height(person)
            overall_heights.append(height)
    common_scale = get_approximate_scale(overall_heights)
    return common_scale

# creates a person to person distance matrix
def create_distance_matrix(head_pos_list, scale):
    mat_size = len(head_pos_list)
    dist_mat = np.zeros((mat_size, mat_size))
    for i in range(mat_size):
        for j in range(i, mat_size):
            dist = distance(head_pos_list[i], head_pos_list[j])
            dist_mat[i][j] = dist * scale
    return dist_mat


# find scale to align sample heights to national average
def get_approximate_scale(heights):
    avg_american_height = 168.4  # cm source:wikipedia averaged between male and female
    sample_average = np.mean(np.array(heights))
    return avg_american_height / sample_average


# measures y distance to both foot nodes and returns the max
def get_height(person):
    head_pos = person[1]
    foot1_pose = person[8]
    foot2_pose = person[14]
    h1 = abs(head_pos[1] - foot1_pose[1])
    h2 = abs(head_pos[1] - foot2_pose[1])
    height = max(h1, h2)
    return height


# compute distance between two 3d points
def distance(pt1, pt2):
    t1 = (pt1[0] - pt2[0]) ** 2
    t2 = (pt1[1] - pt2[1]) ** 2
    t3 = (pt1[2] - pt2[2]) ** 2
    return math.sqrt(t1 + t2 + t3)


# writes out an image with an overlayed colored line showing
# social distances between people
def visualize_distance(dist_mat, image_dir, dat, out_dir):
    image_path = os.path.join(image_dir, dat['image_path'])
    img = cv2.imread(image_path)
    # img = cv2.imread(image_path)[:, :, ::-1]
    scalex, scaley = scale_2d(img)
    poses_2d = dat['pred_2d']
    cents = []
    for person in poses_2d:
        cent = get_2d_centroid(person, scalex, scaley)
        cents.append(cent)

    for i in range(len(poses_2d)):
        for j in range(i, len(poses_2d)):
            if cents[i] != (0, 0) and cents[j] != (0, 0) and dist_mat[i, j] > 0.0:
                color, thickness = get_line_desc(dist_mat[i, j])
                img = cv2.line(img, cents[i], cents[j], color, thickness)
    outpath = os.path.join(out_dir, dat['image_path'])
    cv2.imwrite(outpath, img)


# writes out an image with an overlayed colored line showing 
# social distances between people
def visualize_distance2(data_dict, out_dir):
    sortedKeys = natsorted(data_dict.keys())
    for key in sortedKeys:
        im_dict = data_dict[key]
        dist_mat = np.array(im_dict['distance_matrix'])
        image_path = im_dict["full_im_path"]
        img = cv2.imread(image_path)
        people = im_dict['people']
        cents = []
        for kk in range(len(people)):
            person = people[kk]
            if not person['valid_sub_im']:
                continue
            cent = tuple([int(x - 10) for x in person['head_pos']])
            cents.append(cent)
            mask_status = person['is_wearing_mask']
            col, thi = get_text_desc(mask_status)
            img = cv2.putText(img, mask_status, cent, cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)
        for i in range(len(cents)):
            for j in range(i, len(cents)):
                if cents[i] != (0, 0) and cents[j] != (0, 0) and dist_mat[i, j] > 0.0:
                    color, thickness = get_line_desc(dist_mat[i, j])
                    img = cv2.line(img, cents[i], cents[j], color, thickness)
        outpath = os.path.join(out_dir, key)
        cv2.imwrite(outpath, img)

# writes out an image with an overlayed colored line showing
# social distances between people, includes tracking
def visualize_distance3(data_dict, out_dir):
    sortedKeys = natsorted(data_dict.keys())
    for key in sortedKeys:
        im_dict = data_dict[key]
        dist_mat = np.array(im_dict['distance_matrix'])
        image_path = im_dict["full_im_path"]
        img = cv2.imread(image_path)
        people = im_dict['people']
        cents = []
        for kk in range(len(people)):
            person = people[kk]
            if not person['valid_sub_im']:
                continue
            cent = tuple([int(x - 10) for x in person['head_pos']])
            cents.append(cent)
            mask_status = person['is_wearing_mask']
            col, thi = get_text_desc(mask_status)
            img = cv2.putText(img, "ID: " + str(person['ID']) + ", " + mask_status, cent, cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)
        for i in range(len(cents)):
            for j in range(i, len(cents)):
                if cents[i] != (0, 0) and cents[j] != (0, 0) and dist_mat[i, j] > 0.0:
                    color, thickness = get_line_desc(dist_mat[i, j])
                    img = cv2.line(img, cents[i], cents[j], color, thickness)
        outpath = os.path.join(out_dir, key)
        cv2.imwrite(outpath, img)

# selects a line thickness and color based on social distancing
# ideally want to smoothly interpolate between colors and thickness
def get_line_desc(dist):
    # bgr colors 
    c0 = (0, 0, 128)  # dark red
    c1 = (0, 0, 255)  # bright red
    c2 = (0, 128, 255)  # orange red
    c3 = (0, 255, 255)  # yellow
    c4 = (0, 255, 0)  # green
    wide = 8
    med = 4
    thin = 2
    fine = 1
    if dist < 150.0:
        color = c0
        thick = wide
    elif dist < 200:
        color = c1
        thick = wide
    elif dist < 250:
        color = c2
        thick = med
    elif dist < 300:
        color = c3
        thick = thin
    else:
        color = c4
        thick = fine
    return color, thick


def get_text_desc(mask_status):
    # bgr colors 
    c0 = (0, 0, 128)  # dark red
    c1 = (0, 255, 255)  # yellow
    c2 = (0, 255, 0)  # green
    large = 4
    med = 2
    small = 1
    color = c1
    thick = med
    if mask_status == "mask":
        color = c2
        thick = large
    elif mask_status == "no_mask":
        color = c0
        thick = small
    elif mask_status == "unknown":
        color = c1
        thick = med
    return color, thick


# get 2D pixel for perosn and scale it appropriately
def get_2d_centroid(person2d, scalex, scaley):
    if person2d[0][0:2] != [0, 0]:  # neck
        return format_cent(person2d[0][0:2], scalex, scaley)
    if person2d[1][0:2] != [0, 0]:  # head
        return format_cent(person2d[1][0:2], scalex, scaley)
    if person2d[2][0:2] != [0, 0]:  # pelvis
        return format_cent(person2d[0][0:2], scalex, scaley)
    for pt in person2d:
        if pt[0:2] != [0, 0]:
            return format_cent(pt[0:2], scalex, scaley)
    return 0, 0


# format it for cv2 and do scaling
def format_cent(in_cent, scalex, scaley):
    return int(in_cent[0] * scalex), int(in_cent[1] * scaley)


# network outputs 832×512. need to scale to fit right
def scale_2d(img):
    xfrac = img.shape[1] / 832
    yfrac = img.shape[0] / 512
    return xfrac, yfrac

# REALLY ONLY EXISTS TO DEBUG
# # main script. assumes you have already run smap and have a json
# if __name__ == "__main__":
# #    image_dir = "/home/saponaro/Documents/data/park_video_DSCF0538/"
#     import json
#     import pdb
#     json_path = "/home/scott/Documents/sidehustle/covid/SMAP/model_logs/stage3_root2/result/stage3_root2_run_inference_test_.json"
#     out_dir = "/home/scott/Documents/sidehustle/covid/data/common_height/"
#     with open(json_path, 'r') as f:
#         data = json.load(f)['3d_pairs']
#     common_scale = compute_common_scale(data)
#     for dat in data:
#         dist_mat, scaled_heights, head_pos_3d = compute_social_dist(dat,common_scale)
#         pdb.set_trace()
