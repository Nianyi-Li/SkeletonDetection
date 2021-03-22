# this module contains code for extracting relevant pose
# information that can be used for mask detection
# Requires FFMPEG

import argparse
import codecs
import csv
import datetime
import json
import os
import shutil
import subprocess

import cv2
import do_facemask as df
import do_tracking as dt
import social_distance as sd

acceptable_extensions = ['.avi', '.mp4', '.mts']
meta_data_mapping = \
    {
        "pole": {
            "lat": 39.673397,
            "long": -75.743153,
            "project": "Hall",
            "sitename": "pole"
        },
        "academy": {
            "lat": 39.671183,
            "long": -75.750363,
            "project": "Hall",
            "sitename": "academy"
        },
        "pomeroy_tree": {
            "lat": 39.675618,
            "long": -75.742102,
            "project": "Hall",
            "sitename": "pomeroy_tree"
        },
        "pomeroy_woods": {
            "lat": 39.674503,
            "long": -75.739734,
            "project": "Hall",
            "sitename": "pomeroy_woods"
        },
        "train": {
            "lat": 39.670151,
            "long": -75.753649,
            "project": "Hall",
            "sitename": "train"
        },
        "Vintage": {
            "lat": 39.610066,
            "long": -75.726295,
            "project": "Vintage",
            "sitename": "Vintage"
        },
        "southtree": {
            "lat": 39.607587,
            "long": -75.727797,
            "project": "Glasgow",
            "sitename": "southtree"
        }
    }


def speedToActivityType(speed):
    if speed < 0.25:
        return "standing"
    if 0.25 <= speed < 2.25:
        return "walking"
    if 2.25 <= speed < 3.5:
        return "jogging"

    return "biking"


def heightToAge(height):
    if height < 1.25:
        return "child"

    return "adult"


def writeCSV(dict, meta_data, filepath):
    csv_columns = ['project', 'site_location', 'latitude', 'longitude', 'video_name', 'full_path', 'frame_rate', 'date',
                   'start_time', 'end_time', 'ID', 'enter_time', 'leave_time', 'closest_distance_avg (m)',
                   'closest_distance_min (m)', 'is_wearing_mask', 'height (m)', 'age', 'average speed (m/s)',
                   'activity type']
    # row = {'closest_distance_avg': closest_distance, 'closest_distance_min': closest_distance_min,
    #       'is_wearing_mask': is_wearing_mask, "first_frame": first_frame, "last_frame": last_frame}
    if os.path.exists(filepath):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    try:
        with open(filepath, append_write) as csvfile:
            writer = csv.writer(csvfile)
            if append_write == "w":
                writer.writerow(csv_columns)
            start_datetime_obj = meta_data['start_datetime_obj']
            fps = meta_data['frame_rate']
            for person_id in dict.keys():
                entry = dict[person_id]
                enterTime = start_datetime_obj + datetime.timedelta(0, entry['first_frame'] / fps)
                leaveTime = start_datetime_obj + datetime.timedelta(0, entry['last_frame'] / fps)
                totalTime = (leaveTime - enterTime)
                totalSeconds = totalTime.total_seconds()
                if (totalSeconds < 1):
                    continue
                row = []
                row.insert(0, speedToActivityType(entry['avg_speed']))
                row.insert(0, entry['avg_speed'])
                row.insert(0, heightToAge(entry['height']))
                row.insert(0, entry['height'])
                row.insert(0, entry['is_wearing_mask'])
                row.insert(0, entry['closest_distance_min'])
                row.insert(0, entry['closest_distance_avg'])
                row.insert(0, leaveTime.time())
                row.insert(0, enterTime.time())
                row.insert(0, person_id)
                row.insert(0, meta_data['video_end_time'])
                row.insert(0, meta_data['video_start_time'])
                row.insert(0, meta_data['video_date'])
                row.insert(0, meta_data['frame_rate'])
                row.insert(0, meta_data['video_full_path'])
                row.insert(0, meta_data['video_name'])
                row.insert(0, meta_data['longitude'])
                row.insert(0, meta_data['latitude'])
                row.insert(0, meta_data['site_location'])
                row.insert(0, meta_data['project'])
                writer.writerow(row)
    except IOError:
        print("I/O error")


def countFrames(v_path):
    video = cv2.VideoCapture(v_path)
    total = 0
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        total = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    else:
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    return total


def findVideoFrameRate(v_path):
    fps = 30.00
    video = cv2.VideoCapture(v_path)
    (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
    video.release()

    if(fps < 2):
        print("Warning, FPS IS FUCKY, SETTING TO 30")
        fps = 30.00

    return fps


def parseStartTime(filename):
    month = 1
    year = 1
    day = 1
    hour = 0
    minute = 0
    second = 0
    try:
        dot_split = filename.split(".")
        date = dot_split[0]
        mytimet = dot_split[2]
        date_split = date.split("_")
        year = int(date_split[0])
        month = int(date_split[1])
        day = int(date_split[2])
        time_split = mytimet.split("_")
        hour = int(time_split[0])
        minute = int(time_split[1])
        second = int(time_split[2])
    except:
        print("Failed to correctly parse filename for date/time")

    mydatetime = datetime.datetime(year, month, day, hour, minute, second)
    return mydatetime


def parseSiteLocation(filename):
    site_location = ""
    try:
        dot_split = filename.split(".")
        site_location = dot_split[1]
    except:
        print("Failed to correctly parse site location")

    return site_location


def processVideo(video_path, batch_size=2, conf_thresh=0.15, visualize=False, outputJSON=False, outputDir="",
                 first_id=1):

    SMAP_home_directory = os.path.abspath('../')
    weights_dir = os.path.join(os.path.abspath('../..'), 'weights')
    os.environ['PROJECT_HOME'] = SMAP_home_directory
    test_py_path = "\"" + SMAP_home_directory + "/exps/stage3_root2/test.py" "\""
    SMAP_model_path = "\"" + weights_dir + "/SMAP_model.pth\""
    RefineNet_path = "\"" + weights_dir + "/RefineNet.pth\""
    labelsPath = weights_dir + "/obj.data"
    weightsPath = weights_dir + "/yolov4_face_mask.weights"
    configPath = weights_dir + "/yolov4-obj.cfg"

    # parse filename/directory structure
    filename, file_extension = os.path.splitext(video_path)

    if (outputDir == "" or not os.path.isdir(outputDir)):
        result_dir = filename + "_results"
        result_temp_dir = filename + "_temp_results"
        frames_dir = filename
    else:
        result_dir = outputDir
        result_temp_dir = os.path.join(outputDir, "_temp_results")
        frames_dir = os.path.join(outputDir, "_frames")
    video_path_dir, video_path_file = os.path.split(video_path)

    outputCSVPath = result_dir + "/output.csv"

    # find video start and end times, and frame rate, and other meta data, if possible
    startDateTime = parseStartTime(video_path_file)
    fps = findVideoFrameRate(video_path)
    num_frames = countFrames(video_path)
    num_seconds = round(num_frames / fps)
    endDateTime = startDateTime + datetime.timedelta(0, num_seconds)
    site_location = parseSiteLocation(filename)
    latitude = 0
    longitude = 0
    project = ""
    try:
        meta_data_mapping_curr = meta_data_mapping[site_location]
        latitude = meta_data_mapping_curr['lat']
        longitude = meta_data_mapping_curr['long']
        project = meta_data_mapping_curr['project']
    except:
        print("Site location not found, no metadata available")

    meta_data = {"video_directory": video_path_dir,
                 "video_name": video_path_file,
                 "video_full_path": video_path,
                 "frame_rate": fps,
                 "video_date": startDateTime.date(),
                 "video_start_time": startDateTime.time(),
                 "video_end_time": endDateTime.time(),
                 "site_location": site_location,
                 "start_datetime_obj": startDateTime,
                 "project": project,
                 "latitude": latitude,
                 "longitude": longitude,
                 }

    # Create the result directories
    if not os.path.isdir(frames_dir):
        os.mkdir(frames_dir)
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    if not os.path.isdir(result_temp_dir):
        os.mkdir(result_temp_dir)

    # Write out video as frames
    # ffmpeg_command = "ffmpeg -i " + video_path + " -codec:v copy -bsf:v mjpeg2jpeg " + frames_dir + "/%05d.jpg"
    ffmpeg_command = "ffmpeg -i " + video_path + " -qscale:v 1 -qmin 1 -qmax 1 " + frames_dir + "/%05d.jpg"
    print("Deconstructing video into individual frames")
    subprocess.call(ffmpeg_command, shell=True)

    # Run skeleton fitting
    print("Performing skeleton fitting")
    smap_command = "python3 " + test_py_path + " -p " + SMAP_model_path + " -t run_inference -d test -rp " + \
                   RefineNet_path + " --batch_size " + str(
        batch_size) + " --do_flip 1 --dataset_path " + "\"" + frames_dir + "\""
    subprocess.call(smap_command, shell=True)

    # Run face mask detection
    skeleton_json_path = SMAP_home_directory + "/model_logs/stage3_root2/result/stage3_root2_run_inference_test_.json"
    net = df.readYolov4(labelsPath, weightsPath, configPath)
    print("Processing data for facemasks")
    data_dict = df.processJsonForFaces(skeleton_json_path, frames_dir, net, conf_thresh)

    # Tracking people between frames
    print("Tracking people between frames")
    data_dict, max_id = dt.doTracking(data_dict, first_id)
    outputDict = dt.parseTrackingDict(data_dict, first_id, max_id, fps)
    writeCSV(outputDict, meta_data, outputCSVPath)

    # Visualize the results
    if (visualize):
        sd.visualize_distance3(data_dict, result_temp_dir)
        ffmpeg_command2 = "ffmpeg -y -i " + result_temp_dir + "/%*.jpg -q 2 " + result_dir + "/output.avi"
        print("Reconstructing video from individual frames")
        subprocess.call(ffmpeg_command2, shell=True)
    if (outputJSON):
        json.dump(data_dict, codecs.open(result_dir + "/detections.json", 'w', encoding='utf-8'), separators=(',', ':'),
                  sort_keys=True, indent=4)

    shutil.rmtree(frames_dir)
    shutil.rmtree(result_temp_dir)

    return max_id


def processDirectory(video_path, batch_size, conf_thresh, visualize, outputJSON, outputDir, continuePath, continueID):
    skip_file = False
    if(not continuePath == ""):
        skip_file = True

    first_id = continueID
    for path, subdirs, files in os.walk(video_path):
        for name in files:
            full_path = os.path.join(path, name)
            file_without_extension, file_extension = os.path.splitext(full_path)

            if(skip_file == True and not continuePath == full_path):
                continue
            if(skip_file == True and  continuePath == full_path):
                skip_file = False
                continue

            print("Processing " + full_path)
            if file_extension.casefold() in (ext.casefold() for ext in acceptable_extensions):
                first_id = processVideo(full_path, batch_size, conf_thresh, visualize, outputJSON,
                                        outputDir, first_id + 1)


if __name__ == '__main__':
    # -v /media/saponaro/Glasgow2020/hill --outputDir /home/saponaro/MattProjectStorage --continuePath /media/saponaro/Glasgow2020/hill/100MEDIA/2020_11_19.hill.14_18_44.trail.avi --continueID 133
    # -v /home/saponaro/Documents/data/testDirectory --outputDir /home/saponaro/Documents/data/testDirectory
    # -v /media/saponaro/Glasgow2020/hill --outputDir /home/saponaro/MattProjectStorage
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", "-v", type=str, default="", required=True,
                        help='Path to video file to process')
    parser.add_argument("--batch_size", type=int, default=2,
                        help='Batch_size of test')
    parser.add_argument("--conf_thresh", "-c", type=float, default=0.15,
                        help='Face mask confidence threshold')
    parser.add_argument("--visualize", type=bool, default=False,
                        help='Output a video visualization')
    parser.add_argument("--outputJSON", type=bool, default=False,
                        help='Output a json containing results')
    parser.add_argument("--outputDir", type=str, default="",
                        help='Directory to outputResults')
    parser.add_argument("--continuePath", type=str, default="",
                        help='Last video file processed in csv to continue off of')
    parser.add_argument("--continueID", type=int, default=0,
                        help='Last ID recorded in last video file processed')


    args = parser.parse_args()
    video_path = args.video_path
    batch_size = args.batch_size
    conf_thresh = args.conf_thresh
    visualize = args.visualize
    outputJSON = args.outputJSON
    outputDir = args.outputDir
    continuePath = args.continuePath
    continueID = args.continueID

    if not os.path.isdir(video_path):
        processVideo(video_path, batch_size, conf_thresh, visualize, outputJSON, outputDir)
    else:
        processDirectory(video_path, batch_size, conf_thresh, visualize, outputJSON, outputDir, continuePath, continueID)
