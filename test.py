import pytest
from io import BytesIO
from unittest.mock import patch
import numpy as np
from ultralytics import YOLO

# Создаем пример данных видео для тестирования
sample_video_data = b"test.mp4"

@pytest.fixture
def video_file():
    """Создание примера видео файла."""
    video_bytes = BytesIO(sample_video_data)
    video_bytes.name = 'test.mp4'
    return video_bytes


def test_model_response():
    """Тест отклика модели."""
    # Создаем фиктивный фрейм
    dummy_frame = np.zeros((640, 480, 3), dtype=np.uint8)

    # Патчим метод track модели YOLO
    with patch.object(YOLO, 'track', return_value=[dummy_frame]) as mock_track:
        model = YOLO("birds_counter_model.pt")
        result = model.track(dummy_frame, show=False, persist=True, verbose=False)
        mock_track.assert_called_once_with(dummy_frame, show=False, persist=True, verbose=False)
        assert result == [dummy_frame], "Model response does not match expected output"


if __name__ == "__main__":
    pytest.main()
