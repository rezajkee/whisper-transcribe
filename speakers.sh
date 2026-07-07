#!/bin/bash
# Показать примеры реплик по каждому спикеру (для составления mapping)
set -euo pipefail

usage() {
  echo "Использование: $0 <файл.json> [кол-во примеров на спикера]"
  echo "Пример: $0 result/meeting.json 5"
}

JSON="${1:-}"
SAMPLES="${2:-4}"

if [[ -z "$JSON" || "$JSON" == "-h" || "$JSON" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "$JSON" ]]; then
  echo "Не найден: $JSON" >&2
  exit 1
fi

python3 - "$JSON" "$SAMPLES" <<'PY'
import json
import sys
from collections import defaultdict

json_path, n = sys.argv[1], int(sys.argv[2])

with open(json_path, encoding="utf-8") as f:
    data = json.load(f)

by_sp = defaultdict(list)
for seg in data.get("segments", []):
    sp = seg.get("speaker")
    text = (seg.get("text") or "").strip()
    if not sp or not text or len(text) < 20:
        continue
    if text.count("?") > 8 and len(text) < 80:
        continue
    by_sp[sp].append((seg.get("start", 0), text))

def fmt(s):
    s = int(s)
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"

if not by_sp:
    print("Спикеры не найдены. Запускали с --diarize?")
    sys.exit(0)

print("Скопируйте в mappings/your-file.json и замените имена:\n")
print("{")
for i, sp in enumerate(sorted(by_sp.keys())):
    comma = "," if i < len(by_sp) - 1 else ""
    print(f'  "{sp}": "Имя{i+1}"{comma}')
print("}\n")

for sp in sorted(by_sp.keys()):
    segs = by_sp[sp]
    print("=" * 60)
    print(f"{sp}  ({len(segs)} содержательных сегментов)")
    print("=" * 60)
    picks = []
    if segs:
        picks.append(segs[0])
    if len(segs) > 1:
        picks.append(segs[len(segs) // 3])
    if len(segs) > 2:
        picks.append(segs[2 * len(segs) // 3])
    if len(segs) > 1:
        picks.append(segs[-1])
    longest = max(segs, key=lambda x: len(x[1]))
    if longest not in picks:
        picks.append(longest)
    seen = set()
    for j, (t, text) in enumerate(picks[:n], 1):
        if text in seen:
            continue
        seen.add(text)
        if len(text) > 200:
            text = text[:197] + "..."
        print(f"\n{j}. [{fmt(t)}] {text}")
PY
