#!/bin/bash
# WhisperX: транскрипция + разделение спикеров
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Настройки (можно переопределить через env или config.env)
: "${WHISPER_VENV:=$HOME/whisper-env}"
: "${WHISPER_OUTPUT_DIR:=$SCRIPT_DIR/result}"
: "${WHISPER_WORK_DIR:=$SCRIPT_DIR/work}"
: "${WHISPER_LANGUAGE:=ru}"
: "${WHISPER_MODEL:=medium}"
: "${WHISPER_DEVICE:=cpu}"
: "${WHISPER_COMPUTE_TYPE:=int8}"
: "${WHISPER_BATCH_SIZE:=8}"
: "${WHISPER_DIARIZE_MODEL:=pyannote/speaker-diarization-3.1}"
: "${WHISPER_MIN_SPEAKERS:=2}"
: "${WHISPER_MAX_SPEAKERS:=}"

[[ -f "$SCRIPT_DIR/config.env" ]] && source "$SCRIPT_DIR/config.env"

usage() {
  cat <<EOF
Использование: $0 <аудио-файл> [опции]

Аргументы:
  <аудио-файл>          .m4a, .wav, .mp3 и др. (поддерживается ffmpeg)

Опции:
  -n, --min-speakers N   Минимум спикеров (по умолчанию: $WHISPER_MIN_SPEAKERS)
  -x, --max-speakers N   Максимум спикеров (по умолчанию = min)
  -o, --output-dir DIR   Папка для результатов (по умолчанию: $WHISPER_OUTPUT_DIR)
  -m, --model MODEL      Модель Whisper: tiny, base, small, medium, large-v3
  -l, --language LANG    Язык (по умолчанию: $WHISPER_LANGUAGE)
  --no-diarize           Только транскрипция, без разделения голосов
  -h, --help             Справка

Переменные окружения:
  HF_TOKEN               Hugging Face token (или: hf auth login)
  WHISPER_VENV           Путь к Python venv
  WHISPER_OUTPUT_DIR     Папка результатов
  WHISPER_WORK_DIR       Папка для промежуточных WAV

Примеры:
  $0 ~/Downloads/meeting.m4a
  $0 recording.wav -n 3 -x 3
  $0 interview.m4a -n 2 -x 5 -m large-v3
EOF
}

INPUT=""
DIARIZE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -n|--min-speakers) WHISPER_MIN_SPEAKERS="$2"; shift 2 ;;
    -x|--max-speakers) WHISPER_MAX_SPEAKERS="$2"; shift 2 ;;
    -o|--output-dir) WHISPER_OUTPUT_DIR="$2"; shift 2 ;;
    -m|--model) WHISPER_MODEL="$2"; shift 2 ;;
    -l|--language) WHISPER_LANGUAGE="$2"; shift 2 ;;
    --no-diarize) DIARIZE=0; shift ;;
    -*) echo "Неизвестная опция: $1" >&2; usage; exit 1 ;;
    *)
      if [[ -z "$INPUT" ]]; then
        INPUT="$1"
      else
        echo "Лишний аргумент: $1" >&2; usage; exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "$INPUT" ]]; then
  usage
  exit 1
fi

if [[ ! -f "$INPUT" ]]; then
  echo "Файл не найден: $INPUT" >&2
  exit 1
fi

if [[ ! -d "$WHISPER_VENV" ]]; then
  echo "venv не найден: $WHISPER_VENV" >&2
  echo "См. README.md — раздел «Установка»" >&2
  exit 1
fi

MAX_SPEAKERS="${WHISPER_MAX_SPEAKERS:-$WHISPER_MIN_SPEAKERS}"

source "$WHISPER_VENV/bin/activate"

HF_TOKEN_ARG=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN_ARG=(--hf_token "$HF_TOKEN")
elif ! hf auth whoami >/dev/null 2>&1; then
  echo "Нужен Hugging Face token или вход: hf auth login" >&2
  echo "Принять условия: pyannote/speaker-diarization-3.1 и pyannote/segmentation-3.0" >&2
  exit 1
fi

mkdir -p "$WHISPER_OUTPUT_DIR" "$WHISPER_WORK_DIR"

BASENAME="$(basename "$INPUT" | sed 's/\.[^.]*$//')"
WAV="$WHISPER_WORK_DIR/${BASENAME}.wav"

if [[ "$INPUT" != *.wav ]]; then
  echo "→ Конвертация в WAV..."
  ffmpeg -y -i "$INPUT" -ar 16000 -ac 1 "$WAV" -loglevel warning
else
  WAV="$INPUT"
fi

DIARIZE_ARGS=()
if [[ "$DIARIZE" -eq 1 ]]; then
  DIARIZE_ARGS=(
    --diarize
    --diarize_model "$WHISPER_DIARIZE_MODEL"
    --min_speakers "$WHISPER_MIN_SPEAKERS"
    --max_speakers "$MAX_SPEAKERS"
  )
fi

echo "→ Транскрипция (модель: $WHISPER_MODEL, язык: $WHISPER_LANGUAGE)..."
whisperx "$WAV" \
  --language "$WHISPER_LANGUAGE" \
  --model "$WHISPER_MODEL" \
  --device "$WHISPER_DEVICE" \
  --compute_type "$WHISPER_COMPUTE_TYPE" \
  --batch_size "$WHISPER_BATCH_SIZE" \
  "${DIARIZE_ARGS[@]}" \
  "${HF_TOKEN_ARG[@]}" \
  --output_dir "$WHISPER_OUTPUT_DIR" \
  --output_format all

echo ""
echo "Готово: $WHISPER_OUTPUT_DIR/${BASENAME}.{txt,srt,json,...}"
echo ""
echo "Дальше:"
echo "  $SCRIPT_DIR/speakers.sh $WHISPER_OUTPUT_DIR/${BASENAME}.json"
echo "  $SCRIPT_DIR/build_dialog.sh $WHISPER_OUTPUT_DIR/${BASENAME}.json -m mappings/example.json"
