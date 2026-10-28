"""One offline MLX text model process, after all ASR processes have exited."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import time
import wave
from .formats import read_transcript
from .reconcile import (MODEL, REVISION, MAX_OUTPUT_TOKENS, MAX_PROMPT_TOKENS, load_glossary, reconcile)
from .state import atomic_json


class LocalDecider:
    def __init__(self):
        self.model = self.tokenizer = None
        self.calls = 0
        self.load_seconds = 0
        self.inference_seconds = 0

    def __call__(self, messages):
        if self.model is None:
            from huggingface_hub import snapshot_download
            from mlx_lm import load
            started = time.monotonic()
            snapshot = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
            self.model, self.tokenizer = load(snapshot)
            self.load_seconds = time.monotonic() - started
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if len(prompt) > MAX_PROMPT_TOKENS:
            # Bounded units should fit; extreme tokenization remains an explicit
            # unresolved local fallback rather than a truncated prompt.
            return '{}'
        started = time.monotonic()
        raw = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=MAX_OUTPUT_TOKENS,
                       sampler=make_sampler(temp=0.0), verbose=False)
        self.calls += 1
        self.inference_seconds += time.monotonic() - started
        return raw


def run(args):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    with wave.open(str(args.audio), 'rb') as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (1, 2, 16000):
            raise ValueError('Expected normalized mono PCM16 16000 Hz WAV')
        duration = stream.getnframes() / 16000
    a, b = read_transcript(args.parakeet, duration), read_transcript(args.nemotron, duration)
    glossary, digest = load_glossary(args.glossary)
    decider = LocalDecider()
    result = reconcile(a, b, decider, glossary, digest)
    result['language'] = args.language
    result['reconciliation']['runtime_versions'] = {name: importlib.metadata.version(name) for name in ('mlx-lm', 'mlx', 'huggingface_hub')}
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['measurements'] = {'load_seconds': decider.load_seconds, 'inference_seconds': decider.inference_seconds,
                              'model_calls': decider.calls, 'peak_rss_bytes': peak if platform.system() == 'Darwin' else peak * 1024}
    if decider.model is not None:
        import mlx.core as mx
        result['measurements']['mlx_peak_bytes'] = mx.get_peak_memory()
    atomic_json(Path(args.output), result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('audio', 'parakeet', 'nemotron', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--language', default='ru')
    p.add_argument('--glossary', type=Path)
    args = p.parse_args()
    try:
        run(args)
    except Exception as error:
        p.exit(1, f'Reconciliation incomplete ({type(error).__name__}): {error}\n')


if __name__ == '__main__':
    main()
