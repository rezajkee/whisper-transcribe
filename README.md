# whisper-transcribe

Локальная транскрипция аудио с разделением спикеров на Mac (WhisperX + pyannote).

## Что делает

1. Конвертирует аудио в WAV (ffmpeg)
2. Распознаёт речь (Whisper)
3. Разделяет голоса (diarization)
4. Сохраняет `.txt`, `.srt`, `.json` и др.
5. Опционально собирает читаемый диалог в Markdown

## Требования

- macOS (Apple Silicon или Intel)
- [Homebrew](https://brew.sh)
- Python 3.12 venv с WhisperX (`~/whisper-env` по умолчанию)
- Аккаунт [Hugging Face](https://huggingface.co) с принятыми условиями:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

## Установка (один раз)

```bash
brew install ffmpeg python@3.12

python3.12 -m venv ~/whisper-env
source ~/whisper-env/bin/activate
pip install --upgrade pip
pip install whisperx

# Авторизация Hugging Face (один из вариантов):
hf auth login
# или: export HF_TOKEN="hf_..."
```

### Патч pyannote 4.x (если diarization падает с 403 на community-1)

WhisperX 3.8+ использует pyannote 4.x, который ошибочно тянет веса `community-1` даже для модели `3.1`. Если при diarization видите 403 на `speaker-diarization-community-1`, примените патч:

В файле `~/whisper-env/lib/python3.12/site-packages/pyannote/audio/pipelines/speaker_diarization.py` замените загрузку PLDA:

```python
        self.plda = plda
        self.klustering = clustering
        if clustering == "VBxClustering":
            self._plda = get_plda(plda, token=token, cache_dir=cache_dir)
        else:
            self._plda = None
```

(вместо безусловного `self._plda = get_plda(...)`)

## Структура проекта

```
whisper-transcribe/
  transcribe.sh       # транскрипция + diarization
  speakers.sh         # примеры реплик по спикерам
  build_dialog.sh     # сборка Markdown-диалога
  config.env.example  # настройки по умолчанию
  mappings/           # файлы с именами спикеров (любые имена)
  work/               # промежуточные WAV (временные)
  result/             # готовые транскрипты
```

## Быстрый старт

```bash
cd ~/whisper-transcribe

# 1. Транскрипция (3 спикера)
./transcribe.sh ~/Downloads/meeting.m4a -n 3 -x 3

# 2. Посмотреть, кто что говорил
./speakers.sh result/meeting.json

# 3. Создать mapping (скопировать шаблон и подставить имена)
cp mappings/example.json mappings/meeting.json
# отредактировать mappings/meeting.json

# 4. Собрать диалог
./build_dialog.sh result/meeting.json -m mappings/meeting.json
```

Результат: `result/meeting-диалог.md`

## transcribe.sh

```bash
./transcribe.sh <аудио> [опции]
```

| Опция | Описание |
|-------|----------|
| `-n, --min-speakers N` | Минимум спикеров (по умолчанию: 2) |
| `-x, --max-speakers N` | Максимум спикеров (по умолчанию = min) |
| `-o, --output-dir DIR` | Папка результатов |
| `-m, --model MODEL` | `tiny`, `base`, `small`, `medium`, `large-v3` |
| `-l, --language LANG` | Язык (`ru`, `en`, …) |
| `--no-diarize` | Без разделения голосов |

**Примеры:**

```bash
./transcribe.sh recording.m4a
./transcribe.sh recording.m4a -n 3 -x 3
./transcribe.sh recording.m4a -m large-v3 -n 2 -x 4
./transcribe.sh recording.m4a --no-diarize
```

### Настройки через config.env

```bash
cp config.env.example config.env
# отредактировать config.env
```

Переменные: `WHISPER_MODEL`, `WHISPER_LANGUAGE`, `WHISPER_MIN_SPEAKERS`, `WHISPER_VENV`, `WHISPER_OUTPUT_DIR` и др.

### Оценка времени (Mac M2, модель medium)

| Длина записи | Примерное время |
|--------------|-----------------|
| 15 мин | ~10–15 мин |
| 1 час | ~40–60 мин |

## speakers.sh

Показывает примеры реплик каждого `SPEAKER_XX` и выводит готовый шаблон JSON для mapping:

```bash
./speakers.sh result/meeting.json
./speakers.sh result/meeting.json 6   # 6 примеров на спикера
```

## build_dialog.sh

Собирает Markdown с группировкой реплик одного спикера подряд.

```bash
./build_dialog.sh result/meeting.json
./build_dialog.sh result/meeting.json -m mappings/meeting.json
./build_dialog.sh result/meeting.json -o result/notes.md -t "Совещание"
```

Без `-m` спикеры называются **Спикер 1**, **Спикер 2**, …

### Формат mapping (JSON)

```json
{
  "SPEAKER_00": "Алексей",
  "SPEAKER_01": "Игорь",
  "SPEAKER_02": "Мария"
}
```

Один и тот же человек может быть в нескольких метках (если diarization ошибся):

```json
{
  "SPEAKER_01": "Игорь",
  "SPEAKER_02": "Игорь"
}
```

> **Важно:** метки `SPEAKER_XX` не совпадают между разными записями. Для каждой записи создавайте свой mapping.

## Выходные файлы

После `transcribe.sh` в `result/`:

| Файл | Описание |
|------|----------|
| `*.txt` | Текст с метками спикеров |
| `*.srt` | Субтитры |
| `*.json` | Полные данные (слова, время, спикеры) |
| `*.vtt`, `*.tsv` | Другие форматы |

После `build_dialog.sh`:

| Файл | Описание |
|------|----------|
| `*-диалог.md` | Читаемый диалог в Markdown |

## Типичный workflow

```bash
# Запись с диктофона iPhone → AirDrop на Mac
./transcribe.sh ~/Downloads/voice.m4a -n 3 -x 3

./speakers.sh result/voice.json
# → создать mappings/voice.json с реальными именами

./build_dialog.sh result/voice.json -m mappings/voice.json
open result/voice-диалог.md
```

## Устранение проблем

| Симптом | Решение |
|---------|---------|
| 403 на `community-1` при diarization | Патч pyannote (см. выше) или `WHISPER_DIARIZE_MODEL=pyannote/speaker-diarization-3.1` |
| Предупреждение torchcodec | Можно игнорировать при работе с WAV |
| `--output_format txt,srt` invalid | Используйте актуальный `transcribe.sh` (`--output_format all`) |
| Мало RAM | `-m medium`, в `config.env`: `WHISPER_COMPUTE_TYPE=int8`, `WHISPER_BATCH_SIZE=4` |
| Нет HF token | `hf auth login` или `export HF_TOKEN=...` |

## Лицензии моделей

- **Whisper** — MIT
- **pyannote** — требует принятия условий на Hugging Face (бесплатно для research/personal use)
