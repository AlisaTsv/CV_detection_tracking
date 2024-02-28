## Count buckets / Считаем ковши ##  

![gifgit (1)](https://github.com/AlisaTsv/CV_detection_tracking/assets/84089860/fd28fb01-cb60-4112-9a86-2277e68b961d)

eng 

**Task**: to count dump trucks and buckets of sand loaded by the loader into the dump trucks.
The **model** is yolov7 (https://github.com/WongKinYiu/yolov7), pretrained on the COCO dataset and additionally trained on a special dataset for this case.
To track objects – tracker sort, developed by Alex Bewley https://github.com/abewley/sort, is used. It must be downloaded for the script to work.  

As an **indicator of loading**, intersection of bounding boxes of trucks and the loader is used. The value of intersection is defined by IN_TRESHOLD parameter, placed in the beginning of the script. 
In the tail part of the script, there is a block regularizing the process of data saving into the special folder (created in the working directory). Frames with bounding boxes can be saved in the special directory, if it is defined in the script. 

You can use an onnx model or an openvino model (for the second one some yolov7 functions are required, they should be imported from the special module predict_functions_yolov7).
3.0_Bucket_Counter_ON – script for an **onnx** model;
4.0_Bucket_Counter_OV – script for an **openvino** model. 

There are also scripts for velocity test to compare onnx and openvino formats (with making images from predictions or without it): test_onnx, test_onnx_pictures, test_ov, test_ov_pitcures.
You should change paths and image sizes for the ones you use. The scripts are made to be processed with CPU. 

рус

**Задача**: подсчет количества самосвалов и количества ковшей песка, загруженных погрузчиком в самосвалы.
**Модель** – предобученная на COCO yolov7 (https://github.com/WongKinYiu/yolov7), которая дообучалась на специальном датасете для данного кейса. 
Для слежения за объектами использован трекер sort, разработанный Алексом Бьюли. https://github.com/abewley/sort. Его необходимо загрузить для работы скрипта. 

**Индикатором** погрузки является пересечение площадей bounding box самосвалов и погрузчика, относительная величина пересечения IN_TRESHOLD, регулируется в начале скрипта. 
В конце скрипта помещена часть, регулирующая запись данных о пересечениях в специальную папку (создается в рабочей директории). Можно настроить сохранение размеченных кадров в специальной директории. 

Модель может быть подключена в формате onnx или в формате библиотеки openvino (в этом случае необходимые функции yolov7 импортируются из специального модуля predict_functions_yolov7). 
3.0_Bucket_Counter_ON – модель в формате **onnx**;
4.0_Bucket_Counter_OV – модель в формате **openvino** (модель представляет собой 2 файла: .xml и .bin)

Также есть файлы для тестирования скорости работы моделей onnx и openvino (с отрисовкой изображений и без отрисовки): test_onnx, test_onnx_pictures, test_ov, test_ov_pitcures. 
В скрипте необходимо изменять пути и размеры изображения, которое принимает модель. Все файлы запускались на CPU.  

