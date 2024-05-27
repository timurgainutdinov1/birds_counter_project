import cv2
import numpy as np
from processing_class import MyQueueManager
from ultralytics import YOLO


model = YOLO("birds_counter_model.pt")
cap = cv2.VideoCapture("test.mp4")

assert cap.isOpened(), "Ошибка при чтении видео-файла"
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH,
              cv2.CAP_PROP_FRAME_HEIGHT,
              cv2.CAP_PROP_FPS)
)

video_writer = cv2.VideoWriter(
    "test_result.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)


ZONE_POLYGON = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

# Определяем область для подсчета объектов на все кадре
zone_polygon = (ZONE_POLYGON * np.array([w, h])).astype(int)

queue = MyQueueManager(
    classes_names=model.names,
    reg_pts=zone_polygon,
    line_thickness=3,
    fontsize=1.0,
    region_color=(255, 144, 31),
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Видео пустое или обработка видео успешно завершена.")
        break

    tracks = model.track(im0, show=False, persist=True, verbose=False)
    out = queue.process_queue(im0, tracks)

    video_writer.write(im0)


cap.release()
cv2.destroyAllWindows()
