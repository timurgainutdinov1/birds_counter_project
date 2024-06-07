import pytest
from ultralytics import YOLO


@pytest.fixture()
def load_model():
    """Загрузка модели"""
    model = YOLO("birds_counter_model.pt")
    return model


def test_model(load_model):
    """Проверка качества работы модели на валидационном наборе"""
    metrics = load_model.val(data="./valid_data/data.yaml")
    assert (
        metrics.box.mean_results()[0] > 0.8  # precision
        and metrics.box.mean_results()[1] > 0.8  # recall
        and metrics.box.mean_results()[2] > 0.8  # mAP50
        and metrics.box.mean_results()[3] > 0.4  # mAP50-95
    ), "Требования по метрикам выполняются"
