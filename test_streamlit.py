from streamlit.testing.v1 import AppTest
from st_birds_counter import process_video
import io
from ultralytics import YOLO
import os

at = AppTest.from_file("st_birds_counter.py", default_timeout=1000).run()


def test_process_video():
    """Проверка обработки видеофайла."""
    video_file_path = "short_test.mp4"
    with open(video_file_path, "rb") as f:
        video_file_data = f.read()

    uploaded_video = io.BytesIO(video_file_data)
    output_path = process_video(uploaded_video, YOLO("birds_counter_model.pt"))
    assert os.path.exists(
        output_path
    ), f"Выходной видеофайл не существует по адресу {output_path}"
