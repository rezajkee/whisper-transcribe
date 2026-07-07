#!/bin/bash
# Собрать читаемый диалог в Markdown из JSON WhisperX
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
  cat <<EOF
Использование: $0 <файл.json> [опции]

Аргументы:
  <файл.json>           JSON из WhisperX (из папки result/)

Опции:
  -m, --mapping FILE    JSON с именами спикеров (см. mappings/example.json)
  -o, --output FILE     Выходной .md (по умолчанию: <имя>-диалог.md рядом с json)
  -t, --title TITLE     Заголовок документа
  -h, --help            Справка

Без --mapping спикеры будут названы: Спикер 1, Спикер 2, ...

Примеры:
  $0 result/meeting.json
  $0 result/meeting.json -m mappings/my-speakers.json
  $0 result/meeting.json -o result/meeting-notes.md -t "Совещание 3 июля"
EOF
}

JSON=""
MAPPING=""
OUTPUT=""
TITLE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -m|--mapping) MAPPING="$2"; shift 2 ;;
    -o|--output) OUTPUT="$2"; shift 2 ;;
    -t|--title) TITLE="$2"; shift 2 ;;
    -*) echo "Неизвестная опция: $1" >&2; usage; exit 1 ;;
    *)
      if [[ -z "$JSON" ]]; then
        JSON="$1"
      else
        echo "Лишний аргумент: $1" >&2; usage; exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "$JSON" ]]; then
  usage
  exit 1
fi

if [[ ! -f "$JSON" ]]; then
  echo "Не найден: $JSON" >&2
  exit 1
fi

if [[ -z "$OUTPUT" ]]; then
  base="${JSON%.json}"
  OUTPUT="${base}-диалог.md"
fi

if [[ -z "$TITLE" ]]; then
  TITLE="$(basename "${JSON%.json}")"
fi

python3 - "$JSON" "$OUTPUT" "$TITLE" "$MAPPING" <<'PY'
import json
import sys
from pathlib import Path

json_path, out_path, title, mapping_path = sys.argv[1:5]

with open(json_path, encoding="utf-8") as f:
    data = json.load(f)

speakers = sorted({
    seg.get("speaker")
    for seg in data.get("segments", [])
    if seg.get("speaker")
})

if mapping_path and Path(mapping_path).is_file():
    with open(mapping_path, encoding="utf-8") as f:
        names = json.load(f)
else:
    names = {
        sp: f"Спикер {i}"
        for i, sp in enumerate(speakers, 1)
    }

# дополнить неизвестные метки
for i, sp in enumerate(speakers, 1):
    names.setdefault(sp, f"Спикер {i}")

speakers_line = ", ".join(dict.fromkeys(names[sp] for sp in speakers if sp in names))

def fmt_time(s):
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"

blocks = []
current = None

for seg in data.get("segments", []):
    sp = seg.get("speaker")
    text = (seg.get("text") or "").strip()
    if not sp or not text:
        continue
    name = names.get(sp, sp)
    start = seg.get("start", 0)
    end = seg.get("end", start)

    if current and current["name"] == name:
        current["texts"].append(text)
        current["end"] = end
    else:
        if current:
            blocks.append(current)
        current = {"name": name, "start": start, "end": end, "texts": [text]}

if current:
    blocks.append(current)

lines = [
    f"# {title} — транскрипт",
    "",
    f"Спикеры: {speakers_line}" if speakers_line else "",
    "",
    "---",
    "",
]

for b in blocks:
    t0, t1 = fmt_time(b["start"]), fmt_time(b["end"])
    body = " ".join(" ".join(b["texts"]).split())
    lines.append(f"**{b['name']}** `[{t0} – {t1}]`")
    lines.append("")
    lines.append(body)
    lines.append("")

with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Готово: {out_path} ({len(blocks)} блоков)")
PY
