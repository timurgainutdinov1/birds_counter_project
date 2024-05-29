import streamlit as st
import cv2
import numpy as np
import tempfile
from processing_class import MyQueueManager
from ultralytics import YOLO


@st.cache_resource
def load_model():
    """Загрузка модели"""
    return YOLO("birds_counter_model.pt")


def get_video_file():
    """Загрузка видеофайла."""
    return st.file_uploader("Загрузить видеофайл", type=["mp4", "avi", "mov"])


def open_video_file(uploaded_video):
    """Открытие видеофайла."""
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    tfile.close()
    cap = cv2.VideoCapture(tfile.name)
    assert cap.isOpened(), "Error reading video file"
    return cap


def get_video_properties(cap):
    """Получение свойств видеофайла."""
    w, h, fps = (
        int(cap.get(x))
        for x in (
            cv2.CAP_PROP_FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT,
            cv2.CAP_PROP_FPS,
        )
    )
    return w, h, fps


def create_queue_manager(model, w, h):
    """Создание менеджера очереди
    (используется для подсчета объектов на кадре)."""
    ZONE_POLYGON = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    zone_polygon = (ZONE_POLYGON * np.array([w, h])).astype(int)
    queue = MyQueueManager(
        classes_names=model.names,
        reg_pts=zone_polygon,
        line_thickness=3,
        fontsize=1.0,
        region_color=(255, 144, 31),
    )
    return queue


def process_video_frames(output_video_path, cap,
                         model, queue, stframe, w, h, fps):
    """Обработка кадров видеофайла."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            st.markdown("Обработка видео успешно завершена.")
            break
        tracks = model.track(im0, show=False, persist=True, verbose=False)
        queue.process_queue(im0, tracks)
        out.write(im0)
        im0_resized = cv2.resize(im0, (int(w // 4), int(h // 4)))
        im0 = cv2.cvtColor(im0_resized, cv2.COLOR_BGR2RGB)
        stframe.image(im0, channels="RGB", use_column_width=True)
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def process_video(uploaded_video, model):
    """Обработка видеофайла."""
    cap = open_video_file(uploaded_video)
    w, h, fps = get_video_properties(cap)
    queue = create_queue_manager(model, w, h)
    output_video_path = tempfile.NamedTemporaryFile(delete=False,
                                                    suffix=".mp4").name
    stframe = st.empty()
    process_video_frames(output_video_path, cap,
                         model, queue, stframe, w, h, fps)
    return output_video_path


def download_results(output_video_path):
    """Скачивание видеофайла."""

    with open(output_video_path, "rb") as f:
        video_bytes = f.read()
        st.download_button(
            label="Скачать обработанное видео",
            data=video_bytes,
            file_name="processed_video.mp4",
            mime="video/mp4",
        )


def main():
    st.title("Модель для обнаружения и подсчета птиц")
    st.caption(
        "Приложение предназначено для подсчета по видео. "
        "Вы можете загрузить ваш видеофайл ниже и запустить обработку."
    )
    uploaded_video = get_video_file()
    processing_button = st.button("Начать обработку")
    if uploaded_video is not None and processing_button:
        with st.spinner("Идет обработка... Пожалуйста, подождите..."):
            model = load_model()
            output_video_path = process_video(uploaded_video, model)
            download_results(output_video_path)


if __name__ == "__main__":
    main()
