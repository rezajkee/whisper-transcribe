#!/bin/bash
# WhisperX: транскрипция + разделение спикеров
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Настройки (можно переопределить через env или config.env)
: "${WHISPER_VENV:=$HOME/whisper-env}"
: "${WHISPER_MLX_VENV:=$HOME/whisper-mlx-env}"
: "${WHISPER_CPU_THREADS:=4}"
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

# Explicit engines use the resumable stdlib coordinator. No engine flag retains
# the established WhisperX CLI path and its -m semantics.
for argument in "$@"; do
  case "$argument" in
    --engine|--engine=*|--mapping|--mapping=*|--glossary|--glossary=*|--resume|--force)
      export WHISPER_VENV WHISPER_MLX_VENV WHISPER_CPU_THREADS WHISPER_OUTPUT_DIR WHISPER_LANGUAGE WHISPER_MODEL
      export WHISPER_MIN_SPEAKERS WHISPER_MAX_SPEAKERS WHISPER_BATCH_SIZE WHISPER_DIARIZE_MODEL
      export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
      exec python3 -m whisper_transcribe "$@"
      ;;
  esac
done

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
  --engine ENGINE        whisperx|parakeet|nemotron|dual (явный новый pipeline)
  --mapping FILE         Необязательные имена спикеров (новый pipeline)
  --glossary FILE        Словарь для dual (новый pipeline)
  --resume / --force     Проверить кэш / пересчитать этапы (новый pipeline)
                        Новый pipeline: локальные кэшированные модели, без сети
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
fi

mkdir -p "$WHISPER_OUTPUT_DIR" "$WHISPER_WORK_DIR"

BASENAME="$(basename "$INPUT" | sed 's/\.[^.]*$//')"
NORMALIZE_DIR="$(mktemp -d "$WHISPER_WORK_DIR/normalize.XXXXXX")"
WAV="$NORMALIZE_DIR/${BASENAME}.wav"
echo "→ Конвертация в WAV (16 кГц mono PCM, включая входные WAV)..."
ffmpeg -nostdin -y -i "$INPUT" -vn -ar 16000 -ac 1 -c:a pcm_s16le "$WAV" -loglevel warning
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

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
  --model_cache_only True \
  --batch_size "$WHISPER_BATCH_SIZE" \
  ${DIARIZE_ARGS[@]+"${DIARIZE_ARGS[@]}"} \
  ${HF_TOKEN_ARG[@]+"${HF_TOKEN_ARG[@]}"} \
  --output_dir "$WHISPER_OUTPUT_DIR" \
  --output_format all

echo ""
echo "Готово: $WHISPER_OUTPUT_DIR/${BASENAME}.{txt,srt,json,...}"
echo ""
echo "Дальше:"
echo "  $SCRIPT_DIR/speakers.sh $WHISPER_OUTPUT_DIR/${BASENAME}.json"
echo "  $SCRIPT_DIR/build_dialog.sh $WHISPER_OUTPUT_DIR/${BASENAME}.json -m mappings/example.json"
