"""One Torch stage per process. No heavy imports before offline guards."""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import platform
import resource
import time
import wave
from .formats import canonical
from .state import atomic_json

ALIGN_MODELS = {'ru': 'jonatasgrosman/wav2vec2-large-xlsr-53-russian',
                'en': 'facebook/wav2vec2-base-960h'}



def whisperx_result(raw, duration, model, language):
    """Adapt decoder estimates to audio bounds without altering the native record.

    VAD/Whisper time grids can round the last interval past the final PCM frame.
    Clipping is a bounds diagnostic, never forced alignment or proof of precision.
    Validate native ordering before clipping so out-of-bounds values cannot hide
    reversed intervals or a reordered transcript.
    """
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Audio duration must be positive and finite')
    result = copy.deepcopy(raw)
    result.update(engine='whisperx', model=model, duration=duration, language=language,
                  raw_result=copy.deepcopy(raw), precise_timing=False)

    def clip_interval(value):
        start, end = float(value['start']), float(value['end'])
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError('WhisperX returned invalid native timestamps')
        value['start'] = min(duration, max(0.0, start))
        value['end'] = min(duration, max(0.0, end))
        if (value['start'], value['end']) != (start, end):
            value['timestamp_clipped'] = {'raw_start': start, 'raw_end': end,
                'reason': 'native decoder estimate outside audio duration'}
        return start

    previous = -math.inf
    for index, segment in enumerate(result['segments']):
        start = clip_interval(segment)
        if start < previous:
            raise ValueError('WhisperX returned nonmonotonic native segment starts')
        previous = start
        segment.update(id=f'whisperx-{index:06d}', timing_source='whisper_decoder')
        prior_word = -math.inf
        for word in segment.get('words', []):
            if 'start' in word and 'end' in word:
                start = clip_interval(word)
                if start < prior_word:
                    raise ValueError('WhisperX returned nonmonotonic native word starts')
                prior_word = start
    return canonical(result)


def diarization_result(raw, duration, model):
    """Bound native diarization estimates while retaining the exact native record.

    Pyannote's itertracks (used by WhisperX) yields chronological starts, but
    voices may overlap and their ends need not be monotonic. Validate native
    intervals and start order before clipping, which could otherwise hide errors.
    Keep even wholly outside turns as zero-length intervals at the audio edge;
    clipping is a bounds diagnostic, not a timing precision claim.
    """
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Audio duration must be positive and finite')
    if not isinstance(raw.get('turns'), list):
        raise ValueError('Missing diarization turns')
    result = copy.deepcopy(raw)
    result.update(schema_version=1, duration=duration, model=model,
                  raw_result=copy.deepcopy(raw))
    previous = -math.inf
    for turn in result['turns']:
        start, end = float(turn['start']), float(turn['end'])
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError('Diarization returned invalid native timestamps')
        if start < previous:
            raise ValueError('Diarization returned nonmonotonic native turn starts')
        previous = start
        turn['start'] = min(duration, max(0.0, start))
        turn['end'] = min(duration, max(0.0, end))
        if (turn['start'], turn['end']) != (start, end):
            turn['timestamp_clipped'] = {'raw_start': start, 'raw_end': end,
                'reason': 'native diarization estimate outside audio duration'}
    return result


def align_segments(data, audio, model, metadata, align, duration, padding=5):
    """Align bounded padded units; keep exact input text/metadata, never discard tails.

    Decoder times only select an acoustic search window. They are not corrected
    by a presumed constant latency. A segment exceeding 30 s fails closed unless
    genuine timed words permit bounded grouping upstream.
    """
    result = copy.deepcopy(data)
    output = []
    for segment in data['segments']:
        original = copy.deepcopy(segment)
        if not segment['text'].strip():
            output.append(original)
            continue
        if segment['end'] - segment['start'] > 30.001:
            raise ValueError('Alignment units must be bounded to 30 seconds before alignment')
        left, right = max(0, segment['start'] - padding), min(duration, segment['end'] + padding)
        crop = audio[round(left * 16000):round(right * 16000)]
        aligned = align([{'start': 0.0, 'end': len(crop) / 16000, 'text': segment['text']}],
                        model, metadata, crop, 'cpu', print_progress=False)
        words = [copy.deepcopy(word) for unit in aligned.get('segments', []) for word in unit.get('words', [])]
        # Some aligner fallbacks expose only word_segments.
        if not words:
            words = copy.deepcopy(aligned.get('word_segments', []))
        for word in words:
            if 'start' in word and 'end' in word:
                word['start'] += left
                word['end'] += left
        original['prealignment'] = {'start': segment['start'], 'end': segment['end'],
                                    'timing_source': segment.get('timing_source'), 'words': segment.get('words', [])}
        original['alignment_window'] = {'start': left, 'end': right}
        original['words'] = words
        timed = [w for w in words if 'start' in w and 'end' in w]
        untimed = [w for w in words if 'start' not in w or 'end' not in w]
        # Alignment may omit tokens (unsupported alphabet, digits, empty paths).
        # Exact source text stays authoritative; a lexical mismatch is explicit.
        def lexemes(text):
            import re
            return re.findall(r'\w+', text.casefold())
        coverage = lexemes(' '.join(w.get('word', '') for w in words)) == lexemes(segment['text'])
        original['alignment_coverage_complete'] = coverage and not untimed and bool(words)
        if original['alignment_coverage_complete']:
            original.update(start=min(w['start'] for w in timed), end=max(w['end'] for w in timed), timing_source='forced_alignment', timing_status='aligned')
        else:
            original.update(timing_source=segment.get('timing_source', 'source_interval'), timing_status='unaligned',
                            unaligned_words=untimed, alignment_coverage_mismatch=not coverage)
            if timed:
                original['start'] = min(original['start'], min(w['start'] for w in timed))
                original['end'] = max(original['end'], max(w['end'] for w in timed))
        output.append(original)
    # Padded repeated speech can align to the wrong occurrence. Never reorder
    # semantic source text to make the timing look plausible.
    if any(b['start'] < a['start'] for a, b in zip(output, output[1:])):
        for index, segment in enumerate(output):
            segment['alignment_attempt'] = {key: segment.get(key) for key in ('start', 'end', 'words', 'timing_source')}
            original = data['segments'][index]
            segment.update(start=original['start'], end=original['end'], words=copy.deepcopy(original.get('words', [])),
                           timing_source=original.get('timing_source', 'source_interval'), timing_status='unaligned',
                           alignment_coverage_complete=False, alignment_crossing=True)
    result['segments'] = output
    result['precise_timing'] = all(s.get('timing_status') == 'aligned' for s in output if s['text'].strip())
    return canonical(result, duration)


def run(args):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    if args.cpu_threads < 1:
        raise ValueError('cpu-threads must be positive')
    os.environ['OMP_NUM_THREADS'] = str(args.cpu_threads)
    os.environ['MKL_NUM_THREADS'] = str(args.cpu_threads)
    started = time.monotonic()
    with wave.open(str(args.audio), 'rb') as stream:
        if (stream.getnchannels(), stream.getframerate(), stream.getsampwidth()) != (1, 16000, 2):
            raise ValueError('Expected normalized PCM16 mono 16000 Hz WAV')
        duration = stream.getnframes() / 16000
        import numpy as np
        audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').astype(np.float32) / 32768
    if duration <= 0:
        raise ValueError('Empty audio')
    import torch
    torch.set_num_threads(args.cpu_threads)
    import whisperx
    if args.stage == 'whisperx':
        model = whisperx.load_model(args.model, 'cpu', compute_type='int8', language=args.language,
                                    local_files_only=True, vad_method='pyannote')
        raw = model.transcribe(audio, batch_size=args.batch_size, language=args.language)
        result = whisperx_result(raw, duration, args.model, args.language)
    elif args.stage == 'diarize':
        from whisperx.diarize import DiarizationPipeline
        pipeline = DiarizationPipeline(model_name=args.diarize_model, device='cpu')
        frame = pipeline(audio, min_speakers=args.min_speakers, max_speakers=args.max_speakers)
        raw = {'turns': [{'start': float(row.start), 'end': float(row.end),
                          'speaker': str(row.speaker)} for row in frame.itertuples()]}
        result = diarization_result(raw, duration, args.diarize_model)
    else:
        if not args.input:
            raise ValueError('--input is required for align')
        data = canonical(json.loads(args.input.read_text(encoding='utf-8')), duration)
        name = args.align_model or ALIGN_MODELS.get(args.language)
        if not name:
            raise ValueError('Specify a cached Hugging Face --align-model for this language')
        # Explicit HF model prevents implicit torchaudio bundle downloads in English.
        model, metadata = whisperx.load_align_model(args.language, 'cpu', model_name=name, model_cache_only=True)
        result = align_segments(data, audio, model, metadata, whisperx.align, duration, args.alignment_padding)
        result['alignment_model'] = name
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['worker_measurements'] = {'elapsed_seconds': time.monotonic() - started,
                                   'peak_rss_bytes': rss if platform.system() == 'Darwin' else rss * 1024,
                                   'peak_rss_scope': 'this_worker_process'}
    atomic_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['whisperx', 'diarize', 'align'])
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--language', default='ru')
    parser.add_argument('--model', default='medium')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--cpu-threads', type=int, default=int(os.environ.get('WHISPER_CPU_THREADS', '4')))
    parser.add_argument('--diarize-model', default='pyannote/speaker-diarization-3.1')
    parser.add_argument('--min-speakers', type=int, default=2)
    parser.add_argument('--max-speakers', type=int, default=2)
    parser.add_argument('--align-model')
    parser.add_argument('--alignment-padding', type=float, default=5)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        parser.exit(1, f'Torch worker failed ({type(error).__name__}): {error}\n')


if __name__ == '__main__':
    main()
