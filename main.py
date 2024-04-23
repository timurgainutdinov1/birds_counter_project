import cv2
import numpy as np

# Загрузка предварительно обученной модели YOLO
net = cv2.dnn.readNet("/home/acederys/Documents/urfu_semestr_2/PI_project/streamlit_project/yolo/keras-YOLOv3-model-set/yolov3.weights", "yolo/keras-YOLOv3-model-set/cfg/yolov3.cfg")
classes = []
with open("yolo/keras-YOLOv3-model-set/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
unconnected_layers = net.getUnconnectedOutLayers()
# print(unconnected_layers)
output_layers = [layer_names[layer_idx - 1] for layer_idx in unconnected_layers]


# Загрузка изображения для анализа
img = cv2.imread("dataset/BirdVsDrone/Birds/singleBirdinsky52.jpeg")  # Укажите путь к вашему изображению
height, width, channels = img.shape

# Обработка изображения и обнаружение объектов
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Подсчет количества птиц на изображении
bird_count = 0
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 and class_id == 14:  # 14 - ID класса для птиц в наборе данных COCO
            bird_count += 1

print("Количество птиц на изображении:", bird_count)
