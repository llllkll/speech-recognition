# speech-recognition

## Структура репозитория
```path

speech-recognition/
├── configs/
│   └── config.yaml
├── data/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── decoder.py
│   ├── utils/
│   │   ├── __init__.py
|   |   ├── distill.py
│   │   ├── metrics.py
|   |   ├── inference.py
│   │   └── logger.py
│   ├── train.py
│   └── test.py
├── requirements.txt
├── pyproject.toml
└── README.md
```
## Инструкция по запуску
Клонируем репозиторий:
```shell
!git clone https://github.com/llllkll/speech-recognition.git
```
Создаём необходимые директории:
```shell
os.chdir('speech-recognition')
if not os.path.exists('./data'):
    os.makedirs('./data')
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')
!chmod -R 777 ./data
```
Устанавливаем необходимые библиотеки:
```shell
!pip install -r requirements.txt
 ```
Запускаем обучение:
```shell
!python -m src.train
 ```
В файле configs/config.yaml загрузите необхомую конфигурацию

Для запуска тестирования необходимо загрузить файл /checkpoints/best_model.pth и выполнить команду
```shell
!python -m src.test
 ```
