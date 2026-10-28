"""Canonical transcript validation, speaker attachment, and loss-preserving exports."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import re

UNKNOWN = 'UNKNOWN'


def interval(value, duration):
    start, end = float(value['start']), float(value['end'])
    if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start <= end <= duration + 1e-6:
        raise ValueError('Invalid transcript interval')
    return start, min(end, duration)


def canonical(data, duration=None):
    result = copy.deepcopy(data)
    duration = float(duration if duration is not None else data['duration'])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Audio duration must be positive and finite')
    result.update(schema_version=1, duration=duration)
    if 'segments' not in result:
        raise ValueError('Missing transcript segments')
    segments = result['segments']
    if not isinstance(segments, list):
        raise ValueError('segments must be a list')
    previous = 0
    ids = set()
    for index, segment in enumerate(segments):
        segment['start'], segment['end'] = interval(segment, duration)
        if segment['start'] < previous:
            raise ValueError('Segment starts must be monotonic')
        previous = segment['start']
        if not isinstance(segment.get('text'), str):
            raise ValueError('Segment text must be a string')
        segment.setdefault('id', f'segment-{index:06d}')
        if segment['id'] in ids:
            raise ValueError('Duplicate segment id')
        ids.add(segment['id'])
        segment['speaker'] = segment.get('speaker') or UNKNOWN
        segment.setdefault('words', [])
        if not isinstance(segment['words'], list):
            raise ValueError('words must be a list')
        prior_word = segment['start']
        for word in segment['words']:
            if not isinstance(word, dict) or not isinstance(word.get('word'), str):
                raise ValueError('Words must be objects with string word text')
            for field in ('start', 'end'):
                if field in word and (not math.isfinite(float(word[field])) or not 0 <= float(word[field]) <= duration):
                    raise ValueError('Invalid partial word timestamp')
            # Untimed aligner words stay present and are explicitly marked.
            if 'start' not in word or 'end' not in word:
                word['timing_status'] = 'unaligned'
                continue
            word['start'], word['end'] = interval(word, duration)
            if word['start'] < segment['start'] - 1e-6 or word['end'] > segment['end'] + 1e-6 or word['start'] < prior_word:
                raise ValueError('Word outside segment or nonmonotonic')
            prior_word = word['start']
    return result


def read_transcript(path, duration=None):
    return canonical(json.loads(Path(path).read_text(encoding='utf-8')), duration)


def speaker_for(start, end, turns):
    turns = [turn for turn in turns if turn['start'] < end and turn['end'] > start]
    scores = {}
    for turn in turns:
        overlap = max(0, min(end, turn['end']) - max(start, turn['start']))
        if overlap:
            speaker = turn.get('speaker') or UNKNOWN
            scores[speaker] = scores.get(speaker, 0) + overlap
    ordered = sorted(scores, key=lambda label: (-scores[label], label))
    candidates = [{'speaker': label, 'overlap_seconds': scores[label]} for label in ordered]
    # Multiple simultaneous speakers remain explicitly ambiguous.
    concurrent = any(a.get('speaker') != b.get('speaker') and max(start, a['start'], b['start']) < min(end, a['end'], b['end'])
                     for i, a in enumerate(turns) for b in turns[i + 1:]
                     if a['start'] < end and a['end'] > start and b['start'] < end and b['end'] > start)
    tied = len(ordered) > 1 and math.isclose(scores[ordered[0]], scores[ordered[1]], abs_tol=1e-8)
    return (ordered[0] if ordered and not tied and not concurrent else UNKNOWN), candidates, concurrent


def attach_speakers(data, diarization):
    result = canonical(data)
    turns = diarization.get('turns', [])
    for turn in turns:
        interval(turn, result['duration'])
    assigned = []
    for segment_index, segment in enumerate(result['segments']):
        for value in [*segment['words'], segment]:
            if 'start' not in value or 'end' not in value:
                value['speaker'] = UNKNOWN
                continue
            speaker, candidates, overlap = speaker_for(value['start'], value['end'], turns)
            value.update(speaker=speaker, speaker_candidates=candidates)
            if overlap:
                value['overlapping_speech'] = True
        timed = [w for w in segment['words'] if 'start' in w and 'end' in w]
        labels = {w.get('speaker', UNKNOWN) for w in timed}
        if timed and len(labels) == 1 and segment.get('alignment_coverage_complete') and len(timed) == len(segment['words']) and not segment.get('overlapping_speech'):
            segment['speaker'] = next(iter(labels))
        elif len(segment.get('speaker_candidates', [])) > 1:
            segment.update(speaker=UNKNOWN, mixed_speakers=True)
        # Split only when every source character can be retained exactly and all
        # words have aligned timing. Otherwise mixed speech remains Unknown.
        split = None
        if len(labels) > 1 and segment.get('alignment_coverage_complete') and len(timed) == len(segment['words']):
            offsets, cursor = [], 0
            for word in timed:
                position = segment['text'].find(word.get('word', ''), cursor)
                if not word.get('word') or position < 0:
                    break
                offsets.append(position)
                cursor = position + len(word['word'])
            if len(offsets) == len(timed):
                boundaries = [0] + [i for i in range(1, len(timed)) if timed[i]['speaker'] != timed[i-1]['speaker']] + [len(timed)]
                split = []
                for index, (first, last) in enumerate(zip(boundaries, boundaries[1:])):
                    part = copy.deepcopy(segment)
                    part.update(id=f'{segment["id"]}-speaker-{index:03d}',
                                source_segment_id=segment['id'], source_segment_text=segment['text'],
                                text=segment['text'][0 if first == 0 else offsets[first]:offsets[last] if last < len(timed) else len(segment['text'])],
                                words=timed[first:last], speaker=timed[first]['speaker'],
                                start=timed[first]['start'], end=max(w['end'] for w in timed[first:last]))
                    part.pop('mixed_speakers', None)
                    part.pop('overlapping_speech', None)
                    if any(w.get('overlapping_speech') for w in part['words']):
                        part['overlapping_speech'] = True
                    _, part['speaker_candidates'], _ = speaker_for(part['start'], part['end'], turns)
                    split.append(part)
                # A later parent may start inside this alignment window. Splitting
                # must not put a late fragment before that earlier parent start.
                if (segment_index + 1 < len(result['segments']) and
                        split[-1]['start'] > result['segments'][segment_index + 1]['start']):
                    split = None
                    segment.update(speaker=UNKNOWN, mixed_speakers=True,
                                   speaker_split_status='overlapping_alignment_windows')
        assigned.extend(split if split is not None else [segment])
    result['segments'] = assigned
    return canonical(result)


def stamp(seconds, separator='.'):
    milliseconds = round(seconds * 1000)
    hours, rest = divmod(milliseconds, 3600000)
    minutes, rest = divmod(rest, 60000)
    secs, ms = divmod(rest, 1000)
    return f'{hours:02d}:{minutes:02d}:{secs:02d}{separator}{ms:03d}'


def uncertainty_text(segment):
    notes = []
    if segment.get('boundary_review_required'):
        notes.append('стык независимых окон: требуется прослушивание')
    if segment.get('timing_status') == 'unaligned' or segment.get('unaligned_words'):
        notes.append('таймкоды не подтверждены выравниванием')
    if segment.get('mixed_speakers'):
        notes.append('смена спикеров: назначение неоднозначно')
    if segment.get('overlapping_speech'):
        notes.append('наложение голосов')
    for uncertainty in segment.get('uncertainties', []):
        if isinstance(uncertainty, dict):
            if uncertainty.get('kind') == 'punctuation_review':
                at = stamp(uncertainty['start']) + ': ' if isinstance(uncertainty.get('start'), (int, float)) else ''
                notes.append(at + 'пунктуация: требуется ручная проверка')
                continue
            alternatives = uncertainty.get('alternatives', [])
            if isinstance(alternatives, dict):
                variants = ' / '.join(f'{key}: «{value if value else "нет текста"}»' for key, value in alternatives.items())
            elif isinstance(alternatives, list):
                variants = ' / '.join(f'«{value}»' for value in alternatives if isinstance(value, str))
            else:
                variants = ''
            critical = uncertainty.get('critical', [])
            labels = {'numbers': 'числа', 'negation': 'отрицание', 'glossary_term': 'термин из словаря',
                      'invalid_response': 'ответ модели не прошёл проверку'}
            detail = ', '.join(labels.get(flag, 'расхождение текста') for flag in critical)
            at = stamp(uncertainty['start']) + ': ' if isinstance(uncertainty.get('start'), (int, float)) else ''
            note = at + 'варианты ASR, выбор не подтверждён аудио'
            if detail:
                note += ' (' + detail + ')'
            if variants:
                note += ': ' + variants
            notes.append(note)
        else:
            notes.append(str(uncertainty))
    return ' [Проверить ' + stamp(segment['start']) + ': ' + '; '.join(notes) + ']' if notes else ''


def render(data, mapping=None, title='Транскрипт'):
    data = canonical(data)
    mapping = mapping or {}
    if not isinstance(mapping, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items()):
        raise ValueError('Speaker mapping must be a JSON object of names')
    labels = sorted({s['speaker'] for s in data['segments'] if s['speaker'] != UNKNOWN})
    names = {label: mapping.get(label, f'Спикер {index}') for index, label in enumerate(labels, 1)}
    names[UNKNOWN] = mapping.get(UNKNOWN, 'Неизвестный')
    txt, srt, vtt, tsv, md = [], [], ['WEBVTT', ''], ['start\tend\tspeaker\ttext'], [f'# {title}', '']
    if data.get('provisional'):
        md.extend(['**Предварительный результат — требуется проверка.**', ''])
    if data.get('boundaries'):
        md.extend(['**Стыки независимых окон (совпадение текста не подтверждает аудио):** ' + ', '.join(stamp(b['time']) for b in data['boundaries'] if 'time' in b), ''])
    number = 0
    seen_uncertainties = set()
    for segment in data['segments']:
        if not segment['text'].strip():
            continue
        number += 1
        name, text = names[segment['speaker']], segment['text']
        # Speaker splitting copies provenance/uncertainties deliberately. Show
        # each alternative once; the full copies remain in canonical JSON.
        display = dict(segment)
        display['uncertainties'] = []
        for uncertainty in segment.get('uncertainties', []):
            key = (segment.get('source_segment_id', segment['id']), json.dumps(uncertainty, sort_keys=True, ensure_ascii=False))
            if key not in seen_uncertainties:
                display['uncertainties'].append(uncertainty)
                seen_uncertainties.add(key)
        note = uncertainty_text(display)
        body = text + note
        txt.append(f'[{stamp(segment["start"])}] {name}: {body}')
        srt.extend([str(number), f'{stamp(segment["start"], ",")} --> {stamp(segment["end"], ",")}', body, ''])
        vtt.extend([f'{stamp(segment["start"])} --> {stamp(segment["end"])}', body, ''])
        tsv.append(f'{segment["start"]:.3f}\t{segment["end"]:.3f}\t{name.replace(chr(9), " ")}\t{body.replace(chr(9), " ").replace(chr(10), " ")}')
        md.extend([f'**{name}** `[{stamp(segment["start"])} – {stamp(segment["end"])}]`', '', body, ''])
    return {extension: '\n'.join(lines) + '\n' for extension, lines in [('txt', txt), ('srt', srt), ('vtt', vtt), ('tsv', tsv), ('md', md)]}
