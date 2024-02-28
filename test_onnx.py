import numpy as np
import cv2
import time
import datetime
import re
import os
import onnxruntime as ort

# model
model_path = 'C:/Users/Alisa/PycharmProjects/pythonProject20/optimized_models_384/best.onnx'
model_name = model_path.split('/')[-1]
model_size = '384х384'

# path
video_path = 'C:/Users/Alisa/Downloads/test1.mp4'
project_path = 'C:/Users/Alisa/PycharmProjects/pythonProject20'

# directories to save information and results
#if not os.path.isdir(project_path + '/' + 'frames'):
    #os.mkdir(project_path + '/' + 'frames')
if not os.path.isdir(project_path + '/' + 'info'):
    os.mkdir(project_path + '/' + 'info')
#image_save_path = project_path + '/' + 'frames' + '/'
info_data_path = project_path + '/' + 'info' + '/'

# model size parameters
inp_w = 384
inp_h = 384


def timename():
    dtn = str(datetime.datetime.now())
    dtn = dtn.split()
    date = dtn[0]
    time = dtn[1].split('.')
    name = str(date) + '_' + str(time[0]) + str(time[1])
    name = re.sub(':', '_', name)
    return name


start_time = time.time()
img_counter = 0
session = ort.InferenceSession(model_path)
cap = cv2.VideoCapture(video_path)
while img_counter <= 9999:
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
    print(outputs)


timing = time.time() - start_time
print(timing)
tname = timename()
results = str(model_name + ' ' + model_size + ' ' + str(img_counter) + ' ' + str(timing))
with open(info_data_path + tname + model_name + "_model_test_results.txt", "w") as f:
    f.write(results)

