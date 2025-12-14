# Управление четвероногим роботом базовой и residual RL-политикой (обновлено 2025-12-08)

## Участники
- ФИО студента 1  
- ФИО студента 2  
- ФИО студента 3  

## Описание исследования
Проект посвящён обучению политики ходьбы для четвероногого робота (Unitree Aliengo) в Isaac Gym. Базовая политика (AMP/BC) уверенно ходит до ~1.1 м/с. Residual-политика обучается добавлять корректирующее действие к базе, чтобы расширить диапазон скоростей вперёд/вбок и сохранить устойчивый трёхлапый паттерн. Репозиторий содержит:
- Конфиги задач (база и residual) и скрипты запуска/обучения.
- Реализацию ActorCriticResidual с замороженной базой и обучаемой residual-головой.
- PPO-вариант с KL к базе и L2 на residual, логированием метрик и параметров запуска.
- Скрипты инференса с ручной подачей команд и визуализацией трекинга скоростей/контактов.

Дополнительно: в `datasets/` лежат видеотреки для AMP, в `logs/` — чекпоинты обученных моделей (см. ниже).

## Демонстрация работы (Видео)
- Видео-демо: … (добавить ссылку на YouTube/Vimeo/облако)  
- GIF/встроенное превью: … (добавить при наличии)

## Установка и развертывание
- Проверено на Linux, CUDA GPU (Isaac Gym), Python 3.8.20.
- Требуемые системные компоненты: драйвер NVIDIA с поддержкой CUDA 12.x, библиотеки X11 (для рендера), установленный Isaac Gym (поставляется в репо в `isaacgym/`).
- Создание окружения (Conda):
  ```bash
  conda env create -f environment.yml
  conda activate mlr-project
  pip install -e .
  ```
- Если нужно, доустановите системные зависимости X11: `sudo apt-get install libx11-dev libxau-dev libxdmcp-dev`.

## Запуск и использование
- Обучение базовой residual-политики:
  ```bash
  python legged_gym/scripts/train.py \
    --task aliengo_residual \
    --output_name <run_name> \
    --rl_device cuda:0 \
    --headless
  # для продолжения: добавьте --resume --load_run <run_name> --checkpoint <N>
  ```
- Инференс (ручные команды, один робот):
  ```bash
  python legged_gym/scripts/play_manual.py \
    --task aliengo_residual \
    --output_name <run_name> \
    --load_run <run_name> \
    --checkpoint <N> \
    --headless
  ```
  Скрипт логирует трекинг скоростей в `logs/vel_tracking.png` и графики по суставам/контактам (через Logger).
- Тренировка базовой AMP-политики (пример):
  ```bash
  python legged_gym/scripts/train.py \
    --task aliengo_amp \
    --output_name <run_name> \
    --rl_device cuda:0 \
    --headless
  ```
- Инференс базовой политики:
  ```bash
  python legged_gym/scripts/play_manual.py \
    --task aliengo_amp \
    --output_name <run_name> \
    --load_run <run_name> \
    --checkpoint <N>
  ```

## Описание полученных результатов
- Цель: расширить диапазон скоростей вперёд/вбок при сохранении устойчивого паттерна. База стабильна до ~1.1 м/с; residual тянет выше за счёт корректирующих действий.
- Артефакты:
  - Чекпоинты (пример): `logs/aliengo_amp/video_limp/model_25000.pt` (база), `logs/aliengo_residual/<run_name>/model_*.pt` (residual).
  - Графики трекинга скоростей: `logs/vel_tracking.png` (генерируется `play_manual.py`).
  - Параметры обучения: `logs/aliengo_residual/<run_name>/params.json`, метрики в `metrics.csv`.
- Сырые данные: видео-траектории для AMP в `datasets/video_motion_limp_aliengo/…`. Добавьте ссылку/описание облачной папки: … (путь/URL, что есть что).

## Управление зависимостями
- Основной файл окружения: `environment.yml` (Conda).
- Установка пакета: `pip install -e .` из корня репозитория.
- Isaac Gym поставляется внутри `isaacgym/` (без отдельной загрузки).

## Структура проекта
```
Imitation_from_video/
  README.md                <-- этот файл
  environment.yml          <-- описание окружения (Conda)
  legged_gym/              <-- задачи, конфиги, скрипты запуска (train/play)
    envs/                  <-- среды (Aliengo AMP/residual и др.)
    scripts/               <-- play.py, play_manual.py, train.py, shell-скрипты
  rl/, rsl_rl/             <-- алгоритмы PPO/ResidualPPO, модули actor-critic
  datasets/                <-- видеотреки/AMP-траектории (локальные примеры)
  resources/, third_party/ <-- модели, вспомогательные ресурсы, внешние зависимости
  logs/                    <-- чекпоинты обученных политик, метрики
  unitree_legged_sdk/      <-- SDK для работы с роботом
```
Папки `models/`, `data/` не используются явно; веса и датасеты лежат в `logs/` и `datasets/`. При необходимости создайте отдельные каталоги с инструкциями.

## Лицензия
BSD-3-Clause (см. LICENSE).

The source code of this package is released under [GPLv2](https://www.gnu.org/licenses/) license. We only allow it free for academic usage with several patents. 
For commercial use or cooperation, please contact Dr. Peng Lu lupeng@hku.hk.

For any technical issues, please contact me via email zhaol@connect.hku.hk.

### Citation

Zhao, L., Luo, Z., Han, Y. et al. Learning aggressive animal locomotion skills for quadrupedal robots solely from monocular videos. npj Robot 3, 32 (2025). https://doi.org/10.1038/s44182-025-00048-x
