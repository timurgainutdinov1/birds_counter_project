# Модель для обнаружения и подсчета птиц

Этот репозиторий содержит модель, разработанную для обнаружения и подсчета птиц на видео. 
Модель основана на YOLO (You Only Look Once) архитектуре и обучена на наборе данных, 
содержащем изображения птиц различных видов. Помимо модели, здесь также представлено приложение на Streamlit, которое использует эту модель для обработки видео и отображения результатов.

## Особенности модели:

1. Обнаружение и подсчет птиц на видео.

2. Использование YOLO для точного обнаружения объектов.

3. Визуализация результатов с помощью аннотаций на видео.

## Установка и использование модели:

### Скачайте репозиторий:

`git clone https://github.com/timurgainutdinov1/birds_counter_project.git`

`cd birds_detection_model`

### Установите зависимости:

`pip install -r requirements.txt`

### Запустите обработку видео:

`streamlit run st_birds_counter.py`

Загрузите видео и нажмите кнопку "Начать обработку". Результаты будут отображены непосредственно в интерфейсе Streamlit.

После обработки видео будет сохранено в новый файл.

#### Дополнительная информация:

Для обучения модели использовался набор данных с изображениями птиц различных видов.

Модель может быть доработана и настроена для обнаружения других объектов.

Приложение на Streamlit легко настраивается и может быть расширено с дополнительными функциями визуализации и обработки видео.

## Использование развернутой модели

Модель доступна по ссылке [Birds Counter](https://birdscounter.streamlit.app/)

## Docker


### Запуск

#### Если не устанволен docker

Add Docker's official GPG key:

`sudo apt-get update`

`sudo apt-get install ca-certificates curl`

`sudo install -m 0755 -d /etc/apt/keyrings`

`sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc`

`sudo chmod a+r /etc/apt/keyrings/docker.asc`

Add the repository to Apt sources:

`echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null`

`sudo apt-get update`

#### Установите пакеты Docker

`sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin`

#### Сборка образа

`docker build -t app:latest -f Dockerfile .`

Найдем созданный образ:

`docker images | grep app`

#### Запуск образа

`docker run -p 8501:8501 -d app`

#### Остановка контейнера

Смотрим номера запущенных контейнеров

`docker ps`

Останавливаем контейнер

`docker stop {номер конейнера}`

#### Использование docker-compose

Для установки *docker-compose* выполняем команду:

`sudo apt-get update
sudo apt-get install docker-compose`

В корневой директории проекта создаем файл *docker-compose.yml* со следующим содержанием:

`
services:
  st_birds_counter:
    build: .
    ports:
      - "8501:8501"
`

Запускаем docker-compose при помощи команды:

`docker-compose up`


