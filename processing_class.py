from shapely.geometry import Point
from ultralytics.solutions import QueueManager
from ultralytics.utils.plotting import Annotator, colors


class MyQueueManager(QueueManager):
    """

    Этот класс наследует функции класса QueueManager и
    изменяет текст вывода на экране.
    После обработки треков он отображает количество птиц на экране.

    """

    def extract_and_process_tracks(self, tracks):
        """Извлекает и обрабатывает треки в видео-потоке."""

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
                    (float((box[0] + box[2]) / 2),
                     float((box[1] + box[3]) / 2))
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

                # Проверка, находится ли объект внутри области подсчета
                if len(self.reg_pts) >= 3:
                    is_inside = (self.counting_region
                                 .contains(Point(track_line[-1])))
                    if prev_position is not None and is_inside:
                        self.counts += 1

        # Вывод на экран счетчика птиц
        label = f"Number of birds in the frame : {str(self.counts)}"
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
