"""Explicit experimental engine CLI; legacy shell defaults remain WhisperX."""
import argparse
import json
import math
import os
from pathlib import Path
from .pipeline import ROOT, run


def parser():
    p = argparse.ArgumentParser(description='Local sequential ASR. Default legacy shell mode stays WhisperX; dual requires reconciliation.')
    p.add_argument('input', help='Audio file, including WAV (always normalized to a work copy)')
    p.add_argument('--engine', choices=['whisperx', 'parakeet', 'nemotron', 'dual'], default='whisperx')
    p.add_argument('-n', '--min-speakers', type=int, default=int(os.environ.get('WHISPER_MIN_SPEAKERS', '2')))
    p.add_argument('-x', '--max-speakers', type=int, default=int(os.environ['WHISPER_MAX_SPEAKERS']) if os.environ.get('WHISPER_MAX_SPEAKERS') else None)
    p.add_argument('-m', '--model', default=os.environ.get('WHISPER_MODEL', 'medium'), help='Whisper model only; MLX ASR candidates are pinned')
    p.add_argument('-l', '--language', default=os.environ.get('WHISPER_LANGUAGE', 'ru'))
    p.add_argument('-o', '--output-dir', default=os.environ.get('WHISPER_OUTPUT_DIR', str(ROOT / 'result')))
    p.add_argument('--no-diarize', action='store_true')
    p.add_argument('--mapping', help='Optional JSON speaker names; changes only export')
    p.add_argument('--glossary', help='Optional local glossary for dual reconciliation')
    p.add_argument('--resume', action='store_true', help='Resume verified completed stages (default)')
    p.add_argument('--force', action='store_true', help='Recompute stages; retain previous artifacts')
    p.add_argument('--chunk-seconds', type=float, default=30)
    p.add_argument('--overlap-seconds', type=float, default=4)
    p.add_argument('--align-model', help='Cached Hugging Face model; no automatic downloads')
    p.add_argument('--alignment-padding', type=float, default=5)
    p.add_argument('--cpu-threads', type=int, default=int(os.environ.get('WHISPER_CPU_THREADS', '4')))
    p.add_argument('--batch-size', type=int, default=int(os.environ.get('WHISPER_BATCH_SIZE', '8')))
    p.add_argument('--diarize-model', default=os.environ.get('WHISPER_DIARIZE_MODEL', 'pyannote/speaker-diarization-3.1'))
    p.add_argument('--mlx-venv', default=os.environ.get('WHISPER_MLX_VENV', str(Path.home() / 'whisper-mlx-env')))
    p.add_argument('--whisper-venv', default=os.environ.get('WHISPER_VENV', str(Path.home() / 'whisper-env')))
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.glossary and args.engine != 'dual':
        p.error('--glossary applies only to --engine dual reconciliation; it does not condition single-engine ASR')
    if args.max_speakers is None:
        args.max_speakers = args.min_speakers
    if args.min_speakers < 1 or args.max_speakers < args.min_speakers or args.batch_size < 1 or args.cpu_threads < 1:
        p.error('Require 1 <= min-speakers <= max-speakers and positive batch-size')
    if not all(math.isfinite(v) for v in (args.chunk_seconds, args.overlap_seconds, args.alignment_padding)) or not 0 <= args.overlap_seconds < args.chunk_seconds <= 30 or not 0 <= args.alignment_padding <= 15:
        p.error('Require 0 <= overlap < chunk <= 30 and alignment-padding in [0,15]')
    try:
        result = run(args)
    except KeyboardInterrupt:
        p.exit(130, 'Interrupted; completed stages retained. Run again to resume.\n')
    except Exception as error:
        p.exit(1, f'Pipeline incomplete ({type(error).__name__}): {error}\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
