"""Sequential stdlib coordinator. Heavy runtimes live only in child processes."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import wave
from .formats import attach_speakers, read_transcript, render
from .state import Journal, atomic_json, file_hash, fingerprint, run_lock

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    'parakeet': ('mlx-community/parakeet-tdt-0.6b-v3', 'ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15'),
    'nemotron': ('mlx-community/nemotron-3.5-asr-streaming-0.6b', 'e550040c0478027ed679b2b6b0d055502c103663'),
}

# Executed by a target interpreter without importing MLX, Torch or Transformers.
RUNTIME_PROBE = r'''
import hashlib, importlib.metadata as md, json, pathlib, platform, sys
h = hashlib.sha256()
# VAD and other models can ship inside a distribution rather than the HF cache.
model_assets = {'.bin', '.safetensors', '.onnx', '.pt', '.pth', '.ckpt', '.npz', '.npy', '.tflite', '.gguf', '.pb', '.model'}
versions = {}
for distribution in sorted(md.distributions(), key=lambda d: d.metadata.get('Name', '')):
    name = distribution.metadata.get('Name', '')
    versions[name] = distribution.version
    for entry in sorted(distribution.files or [], key=str):
        path = pathlib.Path(distribution.locate_file(entry))
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in model_assets or suffix in ('.py', '.json', '.yaml', '.yml') or path.name in ('METADATA', 'RECORD'):
            h.update(str(entry).encode())
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
        elif suffix in ('.so', '.dylib'):
            st = path.stat()
            h.update(str((str(entry), st.st_size, st.st_mtime_ns)).encode())
print(json.dumps({'python': sys.version, 'executable': str(pathlib.Path(sys.executable).resolve()),
                  'platform': platform.platform(), 'packages': versions, 'installed_code_sha256': h.hexdigest()}))
'''


def runtime_identity(python):
    process = subprocess.run([str(python), '-c', RUNTIME_PROBE], capture_output=True, text=True, check=True)
    result = json.loads(process.stdout)
    result['executable_sha256'] = file_hash(result['executable'])
    result['venv_python'] = str(python)
    return result


def model_identity(model):
    """Hash local cache files, including weight contents; no Hub imports or network."""
    direct = Path(model).expanduser()
    hub = Path(os.environ.get('HF_HUB_CACHE', str(Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface'))) / 'hub')))
    directory = direct if direct.exists() else hub / ('models--' + model.replace('/', '--'))
    entries = []
    if directory.is_file():
        entries.append((directory.name, file_hash(directory)))
    elif directory.is_dir():
        # snapshots contain symlinks to blobs. Hash each referenced file once.
        roots = [directory / 'snapshots', directory / 'refs'] if (directory / 'snapshots').exists() else [directory]
        seen = {}
        for root in roots:
            for path in sorted(root.rglob('*')):
                if path.is_file() and '.locks' not in path.parts:
                    resolved = str(path.resolve())
                    if resolved not in seen:
                        seen[resolved] = file_hash(path)
                    entries.append((str(path.relative_to(directory)), seen[resolved]))
    return {'model': model, 'cache_path': str(directory), 'files_sha256': fingerprint(entries), 'file_count': len(entries),
            'cached_revisions': sorted(path.name for path in (directory / 'snapshots').iterdir()) if (directory / 'snapshots').is_dir() else []}


def source_identity(*names):
    return {name: file_hash(ROOT / 'whisper_transcribe' / name) if (ROOT / 'whisper_transcribe' / name).exists() else 'missing' for name in names}


def execute(command, output):
    """Capture this child's wait4 usage; never subtract cumulative child peaks."""
    output = Path(output)
    before = time.monotonic()
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    env['PYTHONPATH'] = str(ROOT) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    usage, returncode = None, None
    with (output.parent / 'stdout.log').open('wb') as stdout, (output.parent / 'stderr.log').open('wb') as stderr:
        process = subprocess.Popen([str(part) for part in command], stdout=stdout, stderr=stderr, cwd=ROOT, env=env, start_new_session=True)
        try:
            _, status, usage = os.wait4(process.pid, 0)
            returncode = os.waitstatus_to_exitcode(status)
            process.returncode = returncode
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            rss = usage.ru_maxrss if usage else None
            atomic_json(output.parent / 'child_measurements.json', {
                'elapsed_seconds': time.monotonic() - before, 'returncode': returncode,
                'peak_rss_bytes': (rss if platform.system() == 'Darwin' else rss * 1024) if rss is not None else None,
                'peak_rss_scope': 'single_child_wait4', 'stderr_path': 'stderr.log', 'stdout_path': 'stdout.log'})
    if returncode:
        raise RuntimeError(f'Worker exited {returncode}; inspect {output.parent / "stderr.log"}')


def wav_duration(path):
    with wave.open(str(path), 'rb') as stream:
        if stream.getnchannels() != 1 or stream.getframerate() != 16000 or stream.getsampwidth() != 2 or stream.getcomptype() != 'NONE':
            raise ValueError('Normalization did not produce PCM16 mono 16000Hz WAV')
        duration = stream.getnframes() / 16000
    if duration <= 0:
        raise ValueError('Input contains no audio frames')
    return duration


def normalize(source, output):
    # This runs for WAV too. Output is always a different attempt-local file.
    execute(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-i', source,
             '-vn', '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', '-f', 'wav', output], output)


def run(args, *, worker_command=None, identities=None):
    """worker_command/identities are injection seams for stdlib subprocess tests."""
    if args.glossary and args.engine != 'dual':
        raise ValueError('--glossary applies only to dual reconciliation')
    source = Path(args.input).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError('Input must be a file')
    digest = file_hash(source)
    safe = re.sub(r'[^\w.-]+', '_', source.stem, flags=re.UNICODE).strip('._')[:80] or 'audio'
    run_id = safe + '-' + digest[:20]
    output = Path(args.output_dir).expanduser().resolve()
    directory = output / '.runs' / run_id
    output.mkdir(parents=True, exist_ok=True)
    mlx_python = Path(args.mlx_venv).expanduser() / 'bin/python'
    torch_python = Path(args.whisper_venv).expanduser() / 'bin/python'
    config = {key: value for key, value in vars(args).items() if key not in ('resume', 'force')}
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    config['repository_revision'] = revision.stdout.strip() if revision.returncode == 0 else 'unversioned'
    config.update(source=str(source), source_sha256=digest, run_id=run_id, pipeline_source=source_identity('pipeline.py', 'state.py'))
    with run_lock(directory):
        journal = Journal(directory, config, force=args.force)
        # Published copies only represent a completed CURRENT attempt. Historic raw
        # artifacts and old completed exports remain immutable under stages/.
        prefix = output / (run_id + '-' + args.engine)
        for extension in ('json', 'txt', 'srt', 'vtt', 'tsv', 'md'):
            Path(str(prefix) + '.' + extension).unlink(missing_ok=True)
        try:
            fft = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
            normalized = journal.stage('normalize', {'source': digest, 'ffmpeg': fft, 'pcm': 's16le-16k-mono-v1', 'source_code': source_identity('pipeline.py')},
                                       lambda path: normalize(source, path), wav_duration, filename='audio.wav')
            if file_hash(source) != digest:
                raise RuntimeError('Source audio changed while normalizing; run again with the stable file')
            duration = wav_duration(normalized)
            audio_hash = file_hash(normalized)
            identity_cache = identities if identities is not None else {}
            def identity(kind):
                if kind not in identity_cache:
                    identity_cache[kind] = runtime_identity(mlx_python if kind == 'mlx' else torch_python)
                return identity_cache[kind]
            def worker(stage, path, *flags):
                if worker_command:
                    command = [*worker_command, '--stage', stage, '--audio', normalized, '--output', path, *flags]
                else:
                    command = ([mlx_python, '-m', 'whisper_transcribe.asr_worker', '--engine', stage]
                               if stage in MODELS else [torch_python, '-m', 'whisper_transcribe.torch_worker', '--stage', stage])
                    command += ['--audio', normalized, '--output', path, *flags]
                if stage not in MODELS:
                    command += ['--cpu-threads', str(args.cpu_threads)]
                execute(command, path)
            def transcript(path):
                return read_transcript(path, duration)
            asrs = {}
            for engine in (('parakeet', 'nemotron') if args.engine == 'dual' else (args.engine,)):
                model = MODELS[engine][0] if engine in MODELS else args.model
                model_cache = model_identity(model if engine in MODELS or '/' in model or Path(model).exists() else 'Systran/faster-whisper-' + model)
                key = {'audio': audio_hash, 'engine': engine, 'model': model_cache, 'revision': MODELS.get(engine, ('', None))[1],
                       'language': args.language, 'runtime': identity('mlx' if engine in MODELS else 'torch'),
                       'source': source_identity('asr_worker.py' if engine in MODELS else 'torch_worker.py'),
                       'chunk': args.chunk_seconds if engine in MODELS else None,
                       'overlap': args.overlap_seconds if engine == 'parakeet' else None,
                       'batch': args.batch_size if engine == 'whisperx' else None,
                       'cpu_threads': args.cpu_threads if engine == 'whisperx' else None}
                flags = ['--language', args.language]
                flags += ['--chunk-seconds', str(args.chunk_seconds), '--overlap-seconds', str(args.overlap_seconds)] if engine in MODELS else ['--model', args.model, '--batch-size', str(args.batch_size)]
                asrs[engine] = journal.stage(engine, key, lambda path, e=engine, f=flags: worker(e, path, *f), transcript)
            diarization = None
            if not args.no_diarize:
                def validate_turns(path):
                    from .formats import interval
                    value = json.loads(path.read_text())
                    if not isinstance(value.get('turns'), list):
                        raise ValueError('Missing diarization turns')
                    for turn in value['turns']:
                        interval(turn, duration)
                diarization = journal.stage('diarize', {'audio': audio_hash, 'runtime': identity('torch'),
                    'model': model_identity(args.diarize_model), 'dependencies': [model_identity(m) for m in ('pyannote/segmentation-3.0', 'pyannote/wespeaker-voxceleb-resnet34-LM')],
                    'min': args.min_speakers, 'max': args.max_speakers, 'cpu_threads': args.cpu_threads, 'source': source_identity('torch_worker.py')},
                    lambda path: worker('diarize', path, '--min-speakers', str(args.min_speakers), '--max-speakers', str(args.max_speakers), '--diarize-model', args.diarize_model), validate_turns)
            selected = next(iter(asrs.values()))
            if args.engine == 'dual':
                glossary_bytes = Path(args.glossary).read_bytes() if args.glossary else None
                glossary = hashlib.sha256(glossary_bytes).hexdigest() if glossary_bytes is not None else None
                key = {'audio': audio_hash, 'asrs': {name: file_hash(path) for name, path in asrs.items()}, 'glossary': glossary,
                       'runtime': identity('mlx'), 'source': source_identity('merge_worker.py', 'reconcile.py'),
                       'model': model_identity('mlx-community/Qwen3-4B-Instruct-2507-4bit'),
                       'revision': '50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b', 'protocol': 1}
                def merge(path):
                    if not worker_command and not (ROOT / 'whisper_transcribe/merge_worker.py').exists():
                        raise RuntimeError('Dual reconciliation worker is not installed; both hypotheses are preserved, final result incomplete')
                    command = [*worker_command, '--stage', 'merge'] if worker_command else [mlx_python, '-m', 'whisper_transcribe.merge_worker']
                    command += ['--audio', normalized, '--parakeet', asrs['parakeet'], '--nemotron', asrs['nemotron'], '--output', path, '--language', args.language]
                    if args.glossary:
                        snapshot = path.parent / ('glossary' + Path(args.glossary).suffix)
                        snapshot.write_bytes(glossary_bytes)
                        command += ['--glossary', snapshot]
                    execute(command, path)
                    if json.loads(path.read_text()).get('provisional'):
                        raise RuntimeError('Reconciliation returned provisional text; dual is incomplete')
                selected = journal.stage('merge', key, merge, transcript)
            from .torch_worker import ALIGN_MODELS
            align_model = args.align_model or ALIGN_MODELS.get(args.language)
            aligned = journal.stage('align', {'input': file_hash(selected), 'audio': audio_hash,
                'runtime': identity('torch'), 'model': model_identity(align_model) if align_model else None,
                'language': args.language, 'padding': args.alignment_padding, 'cpu_threads': args.cpu_threads, 'source': source_identity('torch_worker.py', 'formats.py')},
                lambda path: worker('align', path, '--input', selected, '--language', args.language,
                                   '--alignment-padding', str(args.alignment_padding), *(['--align-model', align_model] if align_model else [])), transcript)
            provenance = {'source_audio_sha256': digest, 'run_id': run_id, 'engine': args.engine,
                          'raw_hypotheses': {engine: str(path.relative_to(directory)) for engine, path in asrs.items()}}
            def assign(path):
                data = transcript(aligned)
                if diarization:
                    data = attach_speakers(data, json.loads(diarization.read_text()))
                data.update(provenance)
                atomic_json(path, data)
            assigned = journal.stage('speakers', {'aligned': file_hash(aligned), 'diarization': file_hash(diarization) if diarization else None,
                'provenance': provenance, 'source': source_identity('formats.py')}, assign, transcript)
            mapping_bytes = Path(args.mapping).read_bytes() if args.mapping else None
            mapping = json.loads(mapping_bytes.decode('utf-8')) if mapping_bytes is not None else {}
            mapping_hash = hashlib.sha256(mapping_bytes).hexdigest() if mapping_bytes is not None else None
            def export(path):
                data = transcript(assigned)
                for extension, text in render(data, mapping, source.stem).items():
                    (path.parent / ('transcript.' + extension)).write_text(text, encoding='utf-8')
                atomic_json(path, data)
                atomic_json(path.parent / 'exports.json', {p.name: file_hash(p) for p in path.parent.glob('transcript.*')})
            def validate_exports(path):
                transcript(path)
                hashes = json.loads((path.parent / 'exports.json').read_text())
                if set(hashes) != {'transcript.' + ext for ext in ('json', 'txt', 'srt', 'vtt', 'tsv', 'md')}:
                    raise ValueError('Incomplete export inventory')
                if any(file_hash(path.parent / name) != digest for name, digest in hashes.items()):
                    raise ValueError('Export content changed')
            exported = journal.stage('export', {'input': file_hash(assigned), 'mapping': mapping_hash,
                'source': source_identity('formats.py')}, export, validate_exports, filename='transcript.json')
            published = []
            for extension in ('json', 'txt', 'srt', 'vtt', 'tsv', 'md'):
                target = Path(str(prefix) + '.' + extension)
                temporary = target.with_name('.' + target.name + '.tmp')
                shutil.copyfile(exported.parent / ('transcript.' + extension), temporary)
                os.replace(temporary, target)
                published.append(str(target))
            journal.data['outputs'] = published
            journal.finish('complete')
            return {'manifest': str(journal.path), 'outputs': published, 'status': 'complete'}
        except BaseException as error:
            journal.finish('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error)
            raise
