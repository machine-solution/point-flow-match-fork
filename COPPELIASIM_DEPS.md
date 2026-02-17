# Установка зависимостей для CoppeliaSim

Для корректной работы CoppeliaSim (без вылетов и тормозов) необходимо установить дополнительные библиотеки.

## Быстрая установка

```bash
bash bash/install_coppeliasim_deps.sh
```

Или вручную:

```bash
sudo apt-get update
sudo apt-get install -y \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavutil-dev \
    libswresample-dev \
    libavfilter-dev \
    libavdevice-dev \
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libfdk-aac-dev \
    libmp3lame-dev \
    libopus-dev \
    libvorbis-dev \
    libtheora-dev \
    libxvidcore-dev
```

## Что было исправлено

1. **Обработка ошибок CoppeliaSim**: Добавлена обработка `RuntimeError` в `_step_safe()` и `rlbench_runner_playback.py`
2. **Задержки инициализации**: Добавлены задержки после `env.launch()` и `env.reset()` для полной инициализации симулятора
3. **Улучшенная обработка ошибок**: Воспроизведение теперь корректно обрабатывает ошибки симулятора и не падает полностью

## После установки

Перезапустите терминал или выполните:
```bash
source ~/.bashrc
```

Затем можно использовать:
- `bash bash/record_open_fridge.sh` - запись действий
- `bash bash/playback_open_fridge.sh <file.json>` - воспроизведение

## Предупреждение о video compression library

Если вы видите предупреждение:
```
Could not find or correctly load the video compression library.
```

Но библиотеки установлены (проверьте: `dpkg -l | grep libavcodec`), это предупреждение можно игнорировать. Оно не влияет на работу симуляции, только на возможность записи видео из CoppeliaSim.

Если хотите убрать предупреждение:
1. Убедитесь, что установлены runtime библиотеки: `sudo apt-get install libavcodec60 libavformat60 libswscale7`
2. Обновите кэш библиотек: `sudo ldconfig`
3. Перезапустите CoppeliaSim
