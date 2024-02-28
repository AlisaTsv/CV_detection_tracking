import numpy as np
import pandas as pd
import cv2
import datetime
import re
import os
# library for tracker
# @inproceedings{Bewley2016_sort,
  # author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
  # booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
  # title={Simple online and realtime tracking},
  # year={2016},
  # pages={3464-3468},
  # keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
  # doi={10.1109/ICIP.2016.7533003}
# }
# https://github.com/abewley/sort
from sort import *
import onnxruntime as ort

# ***** MODEL
# https://github.com/WongKinYiu/yolov7

# ***** PATH
# ***** enter YOUR path ! *****
model_path = 'C:/Users/Alisa/PycharmProjects/pythonProject20/model/tl2.onnx'  # 640 352
video_path = 'C:/Users/Alisa/Downloads/test1.mp4'
# no / in the end
project_path = 'C:/Users/Alisa/PycharmProjects/pythonProject20'
# directories to save information and results
if not os.path.isdir(project_path + '/' + 'frames'):
    os.mkdir(project_path + '/' + 'frames')
if not os.path.isdir(project_path + '/' + 'info'):
    os.mkdir(project_path + '/' + 'info')
image_save_path = project_path + '/' + 'frames' + '/'
info_data_path = project_path + '/' + 'info' + '/'

# ***** PARAMETERS
# model size parameters
inp_w = 640
inp_h = 352
# class names
names = ['Truck', 'Loader']
# confidence threshold
SCORE_THRESH = 0.7
# area intersection threshold = intersection area / loader's bounding box area
IN_THRESHOLD = 0.25
limits1 = [0, 390, 190, 720]
limits2 = [1050, 720, 1050, 0]


# ***** TOOLS
def line_intersection(a0, a1, b0, b1):
    if a0 >= b0 and a1 <= b1:
        intersection = a1 - a0
    elif a0 < b0 and a1 > b1:
        intersection = b1 - b0
    elif a0 < b0 and a1 > b0:
        intersection = a1 - b0
    elif a1 > b1 and a0 < b1:
        intersection = b1 - a0
    else:
        intersection = 0
    return intersection


def square_intersection(polygon1, polygon2):
    # rectangle against which you are going to test the rest and its area
    # a = x, b = y
    a0, b0, a1, b1, = polygon1
    area_of_p1 = float((a1 - a0) * (b1 - b0))
    # rectangle to check
    # c = x, d = y
    c0, d0, c1, d1 = polygon2
    width_of_intersection = line_intersection(c0, c1, a0, a1)
    height_of_intersection = line_intersection(d0, d1, b0, b1)
    area_of_intersection = width_of_intersection * height_of_intersection
    portion = area_of_intersection / area_of_p1
    return portion


def timename():
    dtn = str(datetime.datetime.now())
    dtn = dtn.split()
    date = dtn[0]
    time = dtn[1].split('.')
    name = str(date) + '_' + str(time[0]) + str(time[1])
    name = re.sub(':', '_', name)
    return name


# ***** COUNTERS
img_counter = 0
# initial state for movement detector
initial_state = None
# registration of movement
motion_state = [0] * 4
# crossing the line events
in_count = []
# time of crossing the line
in_count_info = []
# intersection state registration
intersection_state = [0] * 3
bucket_count = []

# array to save predictions if needed
# predictions = np.empty((0, 7))
# array to save intersection info
intersections = np.empty((0, 4))

# ****** CYCLE
session = ort.InferenceSession(model_path)
# tracker for trucks
tracker = Sort(max_age=1200000, min_hits=5, iou_threshold=0.2)
# frame to start if needed
# first_frame = 150000
cap = cv2.VideoCapture(video_path)
# starting frame setting if needed
# cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)  # may start with gray frame for several seconds, wait
while True:
    success, img = cap.read()
    img_counter += 1

    # original shape saving
    image_shape = img.shape

    # image preprocessing
    image = cv2.resize(img, (inp_w, inp_h))
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)
    im = image.astype(np.float32)
    im /= 255

    # get the name of each output from the output of the session object and store it in a list
    inname = [i.name for i in session.get_inputs()]
    outname = [i.name for i in session.get_outputs()]
    inp = {inname[0]: im}
    # ONNX inference
    outputs = session.run(outname, inp)[0]

    # outputs saving if needed
    # pred_array = np.array(outputs)
    # predictions = np.vstack((predictions, pred_array))

    # np array for objects saving
    objects = np.empty((0, 8))

    # np array for trackers
    truck_detections = np.empty((0, 5))

    # saving info and draw boxes from outputs
    for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):  # for this model / in enumerate(outputs)
        if score < SCORE_THRESH:
            continue
        x0 = int(x0 / inp_w * image_shape[1])
        y0 = int(y0 / inp_h * image_shape[0])
        x1 = int(x1 / inp_w * image_shape[1])
        y1 = int(y1 / inp_h * image_shape[0])
        cls_id = int(cls_id)
        score = round(float(score), 3)

        # all info saving
        obj_array = np.array([img_counter, i, cls_id, x0, y0, x1, y1, score])
        objects = np.vstack((objects, obj_array))
        # saving info for truck tracker
        if cls_id == 0:
            current_array = np.array([x0, y0, x1, y1, score])
            truck_detections = np.vstack((truck_detections, current_array))

    # update tracker
    tracker_results = tracker.update(truck_detections)

    # borderlines painting
    cv2.line(img, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 255, 0), 5)
    cv2.line(img, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 255, 0), 5)

    # truck tracking
    for result in tracker_results:
        x0, y0, x1, y1, obj_id = result
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        w, h = x1 - x0, y1 - y0
        cx, cy = x0 + w // 2, y0 + h // 2

        cv2.rectangle(img, (x0, y0), (x1, y1), color=(255, 0, 255), thickness=2)
        cv2.putText(img, f'{int(obj_id)} {names[0]}', ((x0 + 3), (y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, thickness=3, color=(255, 0, 255))

        # center points painting
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # crossing line 1 detection
        if limits1[0] - 10 < cx < limits1[2] + 10 and limits1[1] - 10 < cy < limits1[3] + 10:
            if in_count.count(obj_id) == 0:
                in_count.append(obj_id)
                time_now = str(datetime.datetime.now())
                in_count_info.append([obj_id, time_now, 1])
                cv2.line(img, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (255, 0, 255), 5)

        # crossing line 2 detection
        if limits2[0] - 10 < cx < limits2[2] + 10 and limits2[1] - 10 < cy < limits2[3] + 10:
            if in_count.count(obj_id) == 0:
                in_count.append(obj_id)
                time_now = str(datetime.datetime.now())
                in_count_info.append([obj_id, time_now, 2])
                cv2.line(img, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (255, 0, 255), 5)

    # objects info collecting
    data = pd.DataFrame(objects)
    data = data.rename({0: 'img_counter', 1: 'i', 2: 'cls_id', 3: 'x0', 4: 'y0', 5: 'x1', 6: 'y1', 7: 'score'},
                       axis='columns')
    loader = data.loc[data['cls_id'] == 1, 'x0': 'y1']
    trucks = pd.DataFrame(tracker_results)

    # loader's bounding box painting
    if not loader.empty:
        loader_coord = loader.values.tolist()
        x0, y0, x1, y1 = loader_coord[0]
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.putText(img, f'{names[1]}', ((x0 + 3), (y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3,
                    color=(0, 255, 255))

    # square interception detecting
    truck_is_loaded = 0
    truck_loaded_label = 0
    if not (loader.empty or trucks.empty):
        intersection_state_micro = 0
        for _, col in trucks.iterrows():
            # getting values from pandas dataframes
            truck_square = col[[0, 1, 2, 3]].values.tolist()
            truck_label = col[4].tolist()
            loader_square = loader.values.tolist()
            loader_square = loader_square[0]
            # area of intersection: area of intersection / area of loader box
            area_intersection = square_intersection(loader_square, truck_square)
            # area of intersection analyzing, y0 of truck and loader comparison: axis Y starts at upper left corner
            if area_intersection >= IN_THRESHOLD:  # and loader_square[1] <= truck_square[1]:
                # intersection registration
                intersection_state_micro = 1
                truck_is_loaded = truck_square
                truck_loaded_label = truck_label
                break

        if intersection_state_micro == 1:
            intersection_state.append(1)
            cv2.rectangle(img, (int(truck_is_loaded[0]), int(truck_is_loaded[1])),
                          (int(truck_is_loaded[2]), int(truck_is_loaded[3])), (0, 0, 255), 5)
            # loading starts: 0 1 1 pattern in intersection_state
            if intersection_state[-3] == 0 and intersection_state[-2] == 1:
                bucket_count.append(1)
                # digit 1 means start loading
                intersect_array = np.array([1, truck_loaded_label, img_counter, str(datetime.datetime.now())])
                intersections = np.vstack((intersections, intersect_array))
            # if loading continues: 1 1 pattern in intersection_state
            elif intersection_state[-2] == 1:
                # digit 2 means loading continues
                intersect_array = np.array([2, truck_loaded_label, img_counter, str(datetime.datetime.now())])
                intersections = np.vstack((intersections, intersect_array))
            else:
                pass
        else:
            intersection_state.append(0)
            # loading finished: 1 0 0  pattern in intersection state
            if intersection_state[-3] == 1 and intersection_state[-2] == 0:
                # digit 3 means loading finished
                intersect_array = np.array([3, truck_loaded_label, img_counter, str(datetime.datetime.now())])
                intersections = np.vstack((intersections, intersect_array))
            else:
                pass

    # counters
    cv2.putText(img, f'{(len(in_count))}' ' trucks entered', (115, 115), cv2.FONT_HERSHEY_PLAIN, 3, (50, 255, 50), 3)
    cv2.putText(img, f'{(len(bucket_count))}' ' buckets', (115, 155), cv2.FONT_HERSHEY_PLAIN, 3, (50, 255, 50), 3)

    # video translation
    cv2.imshow('Image', img)

    # process control: current counters and results saving
    if img_counter % 5000 == 0:
        counters_list = [img_counter, in_count, bucket_count]
        print(counters_list)
        tname = timename()
        with open(info_data_path + tname + "_counters.txt", "w") as f:
            for counter in str(counters_list):
                f.write(counter)
        with open(info_data_path + tname + "_info_counters.txt", "w") as f:
            for line in str(in_count_info):
                f.write(line)
        intersection_data = pd.DataFrame(intersections,
                                         columns=['Action flag', 'Truck Loaded id', 'Image counter', 'Date and Time'])
        intersection_data.to_csv(info_data_path + tname + '_intersections.csv', sep=';')

    # save frames if needed
    # tname = timename()
    # img_name = '_' + str(img_counter)
    # if not cv2.imwrite(image_save_path + tname + img_name + '.jpg', img):
    # # # raise Exception("Could not write image")

    cv2.waitKey(1)

# ***** RESULTS SAVING
tname = timename()
print(img_counter)
print(in_count)
print(bucket_count)
counters_list = [img_counter, in_count, bucket_count]
with open(info_data_path + tname + "_counters.txt", "w") as f:
    for counter in str(counters_list):
        f.write(counter)
with open(info_data_path + tname + "_info_counters.txt", "w") as f:
    for line in str(in_count_info):
        f.write(line)
intersection_data = pd.DataFrame(intersections,
                                 columns=['Action flag', 'Truck Loaded id', 'Image counter', 'Date and Time'])
intersection_data.to_csv(info_data_path + tname + '_intersections.csv', sep=';')
# predict_data = pd.DataFrame(predictions)
# predict_data.to_csv(info_data_path + tname + '_predictions.csv', sep=';')
