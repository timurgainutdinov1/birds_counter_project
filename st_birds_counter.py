import streamlit as st
import cv2
import numpy as np
import tempfile
from shapely.geometry import Point
from ultralytics import YOLO
from ultralytics.solutions import QueueManager
from ultralytics.utils.plotting import Annotator, colors


class MyQueueManager(QueueManager):
    """
    This class inherits functions from QueueManager and modifies the text output on the screen.
    After processing the tracks, it displays the number of birds on the screen.
    """

    def extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks in the video stream."""

        # Инициализируется аннотатор
        self.annotator = Annotator(self.im0, self.tf, self.names)

        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            # Извлекаются треки
            for box, track_id, cls in zip(boxes, track_ids, clss):
                # Изображаются bounding boxes
                self.annotator.box_label(
                    box,
                    label=f"{self.names[cls]}#{track_id}",
                    color=colors(int(track_id), True),
                )

                # Обновляется история треков
                track_line = self.track_history[track_id]
                track_line.append(
                    (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))
                )
                if len(track_line) > 30:
                    track_line.pop(0)

                # Отрисовка следа, если необходимо
                if self.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(
                        track_line,
                        color=self.track_color or colors(int(track_id), True),
                        track_thickness=self.track_thickness,
                    )

                prev_position = (
                    self.track_history[track_id][-2]
                    if len(self.track_history[track_id]) > 1
                    else None
                )

                # Проверка, находится ли объект внутри зоны подсчета
                if len(self.reg_pts) >= 3:
                    is_inside = self.counting_region.contains(Point(track_line[-1]))
                    if prev_position is not None and is_inside:
                        self.counts += 1

        # Выводит счётчик птиц на экран
        label = f"Number of birds in the frame: {str(self.counts)}"
        if label is not None:
            self.annotator.queue_counts_display(
                label,
                points=self.reg_pts,
                region_color=self.region_color,
                txt_color=self.count_txt_color,
            )

        # Сброс счетчика после вывода на экран
        self.counts = 0
        self.display_frames()

def main():
    st.title("Bird Detection and Counting")
    
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        # Сохранение файла видео как временного
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        # Загрузка видео
        cap = cv2.VideoCapture(tfile.name)
        assert cap.isOpened(), "Error reading video file"
        
        w, h, fps = (
            int(cap.get(x))
            for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
        )
        
        # Загрузка модели
        model = YOLO("birds_counter_model.pt")
        
        ZONE_POLYGON = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        zone_polygon = (ZONE_POLYGON * np.array([w, h])).astype(int)
        
        queue = MyQueueManager(
            classes_names=model.names,
            reg_pts=zone_polygon,
            line_thickness=3,
            fontsize=1.0,
            region_color=(255, 144, 31),
        )
        
        # Обработка видео
        stframe = st.empty()
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                st.write("Видео пустое или обработка видео успешно завершена.")
                break
            
            tracks = model.track(im0, show=False, persist=True, verbose=False)
            queue.process_queue(im0, tracks)
            
            im0_resized = cv2.resize(im0, (int(w // 4), int(h // 4)))

            im0 = cv2.cvtColor(im0_resized, cv2.COLOR_BGR2RGB)
            stframe.image(im0, channels="RGB")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
