import os
import openvino as ov
from predict_functions_yolov7 import *

# model
model_path = 'C:/Users/Alisa/PycharmProjects/pythonProject20/optimized_models_384/best.xml'
# model_path = 'C:/Users/Alisa/PycharmProjects/pythonProject20/optimized_models_384/best_int8.xml'
model_name = model_path.split('/')[-1]
model_size = '384х384'

# path
video_path = 'C:/Users/Alisa/Downloads/test1.mp4'
project_path = 'C:/Users/Alisa/PycharmProjects/pythonProject20'

# directories to save information and results
# if not os.path.isdir(project_path + '/' + 'frames'):
    # os.mkdir(project_path + '/' + 'frames')
if not os.path.isdir(project_path + '/' + 'info'):
    os.mkdir(project_path + '/' + 'info')
# image_save_path = project_path + '/' + 'frames' + '/'
info_data_path = project_path + '/' + 'info' + '/'

core = ov.Core()
# read converted model
model = core.read_model(model_path)
compiled_model = core.compile_model(model)

# label names for visualization
NAMES = ['truck', 'loader']
# colors for visualization
COLORS = {name: [np.random.randint(0, 255) for _ in range(3)]
          for i, name in enumerate(NAMES)}

start_time = time.time()
img_counter = 0
cap = cv2.VideoCapture(video_path)
while img_counter <= 9999:
    success, img = cap.read()
    img_counter += 1
    # outputs = detect(compiled_model, img)
    boxes, image, input_shape = detect(compiled_model, img)
    image_with_boxes = draw_boxes(boxes[0], input_shape, image, NAMES, COLORS)
    # visualize results
    cv2.imshow('image', image_with_boxes)
    cv2.waitKey(1)

# results
timing = time.time() - start_time
print(timing)
tname = timename()
results = str(model_name + ' ' + model_size + ' ' + str(img_counter) + ' ' + str(timing))
with open(info_data_path + tname + model_name + "_model_test_results.txt", "w") as f:
    f.write(results)

