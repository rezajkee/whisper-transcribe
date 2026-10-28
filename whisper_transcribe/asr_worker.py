"""Isolated MLX ASR worker. Normal runs never contact Hugging Face."""
from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import tempfile
import time
import wave

MODELS = {
    'parakeet': ('mlx-community/parakeet-tdt-0.6b-v3', 'ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15'),
    'nemotron': ('mlx-community/nemotron-3.5-asr-streaming-0.6b', 'e550040c0478027ed679b2b6b0d055502c103663'),
}


def windows(duration, chunk, overlap):
    """Windows and disjoint ownership intervals; include a short final tail."""
    if not math.isfinite(chunk) or not math.isfinite(overlap) or chunk <= 0 or not 0 <= overlap < chunk:
        raise ValueError('Require finite chunk > overlap >= 0')
    start = 0.0
    while start < duration:
        end = min(start + chunk, duration)
        yield start, end, start + (overlap / 2 if start else 0), end - (overlap / 2 if end < duration else 0)
        if end == duration:
            break
        start += chunk - overlap


def result_segments(raw, offset, duration, engine):
    """Keep real sentence/token timing. Tokens are subwords, never called words."""
    segments = []
    for sentence in raw.get('sentences', []):
        start, end = sentence.get('start'), sentence.get('end')
        if start is None or end is None:
            raise ValueError('Runtime returned a sentence without timestamps')
        start, end = float(start) + offset, float(end) + offset
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ValueError('Runtime returned invalid timestamps')
        segments.append({'start': min(start, duration), 'end': min(end, duration),
                         'text': sentence['text'].strip(),
                         'timing_source': 'decoder_emission' if engine == 'nemotron' else 'tdt_decoder'})
        if end > duration or start > duration:
            segments[-1]['timestamp_clipped'] = {'raw_start': start, 'raw_end': end,
                                                  'reason': 'decoder estimate beyond audio duration'}
    if not segments and raw.get('text', '').strip():
        segments.append({'start': offset, 'end': duration, 'text': raw['text'].strip(),
                         'timing_source': 'audio_window_coarse'})
    return segments


def timed_words(raw, offset=0):
    """Aggregate space-prefixed subword tokens without inventing timing."""
    words = []
    for sentence in raw.get('sentences', []):
        first_token = True
        for token in sentence.get('tokens', []):
            text = token['text']
            if not text:
                continue
            start = float(token['start']) + offset
            end = float(token['end'] if 'end' in token else token['start'] + token['duration']) + offset
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
                raise ValueError('Runtime returned invalid token times')
            if first_token or text[0].isspace():
                words.append({'word': text.strip(), 'start': start, 'end': end})
            else:
                words[-1]['word'] += text
                words[-1]['end'] = max(words[-1]['end'], end)
            first_token = False
    return [w for w in words if w['word']]


def clip_words(words, duration):
    """Copy canonical words into audio bounds, preserving original estimates."""
    clipped = []
    for source in words:
        word = dict(source)
        start, end = word['start'], word['end']
        word['start'] = max(0.0, min(start, duration))
        word['end'] = max(word['start'], min(end, duration))
        if (word['start'], word['end']) != (start, end):
            word['timestamp_clipped'] = {'raw_start': start, 'raw_end': end}
        clipped.append(word)
    return clipped


def word_segments(raw, duration, engine, max_seconds=30):
    """Bound alignment units using genuine decoder word times, retaining raw elsewhere."""
    words = timed_words(raw)
    groups = []
    current = []
    for word in words:
        if current and word['end'] - current[0]['start'] > max_seconds:
            groups.append(current)
            current = []
        current.append(word)
    if current:
        groups.append(current)
    segments = []
    for group in groups:
        segment = {'start': min(group[0]['start'], duration),
                   'end': min(group[-1]['end'], duration),
                   'text': ' '.join(w['word'] for w in group),
                   'timing_source': 'decoder_emission' if engine == 'nemotron' else 'tdt_decoder'}
        segment['words'] = clip_words(group, duration)
        clipped = [w['timestamp_clipped'] for w in segment['words'] if 'timestamp_clipped' in w]
        if clipped:
            segment['timestamp_clipped'] = clipped
        segments.append(segment)
    return segments


def owned_words(raw, offset, left, right, final=False):
    return [w for w in timed_words(raw, offset)
            if left <= (w['start'] + w['end']) / 2 and
            ((w['start'] + w['end']) / 2 < right or (final and (w['start'] + w['end']) / 2 == right))]


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            json.dump(data, out, ensure_ascii=False, indent=2, allow_nan=False)
            out.write('\n')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def run(args):
    if not args.allow_download:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from huggingface_hub import snapshot_download
    model_id, revision = MODELS[args.engine]
    if args.model and args.model != model_id:
        raise ValueError('--model must match the pinned candidate for this engine')
    snapshot = Path(snapshot_download(model_id, revision=revision, local_files_only=not args.allow_download))
    # Validate input before allocating model weights. Upstream normalization is a separate stage.
    with wave.open(str(args.audio), 'rb') as audio:
        if audio.getnchannels() != 1 or audio.getframerate() != 16000 or audio.getsampwidth() != 2:
            raise ValueError('ASR worker requires PCM16 16000 Hz mono WAV; normalize input first')
        duration = audio.getnframes() / audio.getframerate()
    if duration <= 0:
        raise ValueError('Audio is empty')
    planned = list(windows(duration, args.chunk_seconds, args.overlap_seconds))
    from mlx_audio.stt.utils import load_model
    import mlx.core as mx
    started = time.perf_counter()
    model = load_model(snapshot)
    loaded = time.perf_counter()
    raw_results, segments, boundaries = [], [], []
    if args.engine == 'nemotron':
        # Native stateful chunking maintains RNNT/encoder state and flushes the tail.
        # Whole-file PCM is loaded by upstream (about 230 MB/hour at float32).
        raw = dataclasses.asdict(model.generate(str(args.audio), language=args.language,
                                               chunk_duration=args.chunk_seconds, verbose=False))
        raw_results.append({'offset': 0, 'result': raw})
        segments = word_segments(raw, duration, args.engine, args.chunk_seconds) or result_segments(raw, 0, duration, args.engine)
    else:
        # Read at most one PCM window. Do not use upstream's lossy merge fallback.
        import numpy as np
        with wave.open(str(args.audio), 'rb') as audio:
            for index, (start, end, left, right) in enumerate(planned):
                audio.setpos(round(start * 16000))
                pcm = np.frombuffer(audio.readframes(round((end-start)*16000)), dtype='<i2').astype(np.float32) / 32768
                raw = dataclasses.asdict(model.generate(mx.array(pcm), verbose=False))
                raw_results.append({'offset': start, 'ownership': [left, right], 'result': raw})
                if index:
                    boundaries.append({'time': left, 'review_required': True,
                                       'raw_windows': [index - 1, index],
                                       'reason': 'Independent windows may disagree; raw hypotheses retained'})
                if len(planned) == 1:
                    segments.extend(result_segments(raw, start, duration, args.engine))
                else:
                    words = owned_words(raw, start, left, right, index == len(planned)-1)
                    if words:
                        words = clip_words(words, duration)
                        segments.append({'start': min(w['start'] for w in words),
                                         'end': max(w['end'] for w in words),
                                         'text': ' '.join(w['word'] for w in words),
                                         'words': words, 'timing_source': 'tdt_decoder',
                                         'boundary_review_required': True})
                        clipped = [w['timestamp_clipped'] for w in words if 'timestamp_clipped' in w]
                        if clipped:
                            segments[-1]['timestamp_clipped'] = clipped
                    elif raw.get('text', '').strip() and not timed_words(raw):
                        raise ValueError('Cannot join windows without genuine token timestamps')
                mx.clear_cache()
    for index, segment in enumerate(segments):
        segment['id'] = f'{args.engine}-{index:06d}'
    versions = {name: importlib.metadata.version(name) for name in ('mlx-audio', 'mlx', 'huggingface_hub', 'numpy')}
    result = {'schema_version': 1, 'engine': args.engine, 'model': model_id,
              'model_revision': revision, 'runtime_versions': versions,
              'duration': duration, 'language': args.language, 'segments': segments,
              'raw_results': raw_results, 'boundaries': boundaries,
              'parameters': {'chunk_seconds': args.chunk_seconds, 'overlap_seconds': args.overlap_seconds,
                             'overlap_applied': args.engine == 'parakeet',
                             'strategy': 'window_word_midpoint' if args.engine == 'parakeet' else 'native_stateful_streaming'},
              'measurements': {'load_seconds': loaded-started, 'inference_seconds': time.perf_counter()-loaded,
                               'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                               'mlx_peak_bytes': mx.get_peak_memory()}}
    atomic_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', required=True, choices=MODELS)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model')
    parser.add_argument('--language', default='ru')
    parser.add_argument('--chunk-seconds', type=float, default=30)
    parser.add_argument('--overlap-seconds', type=float, default=4)
    parser.add_argument('--allow-download', action='store_true')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        parser.exit(1, f'ASR worker failed ({type(error).__name__}): {error}\n')


if __name__ == '__main__':
    main()
