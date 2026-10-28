"""Crash-safe stage journal and content-validated cache (stdlib only)."""
from __future__ import annotations
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
import uuid


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another process owns this run; retry after it exits') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def memory_snapshot():
    """System snapshots, not a claim about stage or machine peak memory."""
    commands = [['sysctl', 'vm.swapusage'], ['memory_pressure', '-Q']] if platform.system() == 'Darwin' else []
    result = {}
    for command in commands:
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=5)
            result[command[0]] = process.stdout.strip() if process.returncode == 0 else 'unavailable'
        except (OSError, subprocess.TimeoutExpired):
            result[command[0]] = 'unavailable'
    return result


class Journal:
    def __init__(self, directory, config, force=False):
        self.directory = Path(directory)
        self.path = self.directory / 'manifest.json'
        self.force = force
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {'schema_version': 1, 'stages': {}, 'history': []}
        self.data.update(status='running', config=config, started_at=time.time(), pipeline_version=1)
        self.data.pop('error', None)
        self.data['outputs'] = []
        for record in self.data['stages'].values():
            record['current_run_validated'] = False
        self.save()

    def save(self):
        atomic_json(self.path, self.data)

    def stage(self, name, identity, action, validate=None, filename='artifact.json'):
        key = fingerprint(identity)
        old = self.data['stages'].get(name, {})
        if not self.force and old.get('status') == 'complete' and old.get('key') == key:
            try:
                path = self.directory / old['path']
                if file_hash(path) == old['sha256']:
                    if validate:
                        validate(path)
                    old['current_run_validated'] = True
                    self.save()
                    return path
            except (OSError, ValueError, KeyError, TypeError):
                pass
        if old:
            self.data['history'].append(dict(old, stage=name))
        attempt = self.directory / 'stages' / (name + '-' + key[:12] + '-' + uuid.uuid4().hex[:8])
        attempt.mkdir(parents=True)
        path = attempt / filename
        record = {'key': key, 'identity': identity, 'status': 'running', 'path': str(path.relative_to(self.directory)), 'started_at': time.time()}
        self.data['stages'][name] = record
        self.save()
        started = time.monotonic()
        before = memory_snapshot()
        try:
            action(path)
            if validate:
                validate(path)
            record.update(status='complete', sha256=file_hash(path), current_run_validated=True)
        except BaseException as error:
            record.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error_type=type(error).__name__)
            raise
        finally:
            record['measurements'] = {'elapsed_seconds': time.monotonic() - started, 'system_before': before, 'system_after': memory_snapshot()}
            self.save()
        return path

    def finish(self, status, error=None):
        self.data.update(status=status, finished_at=time.time())
        if error:
            # Do not copy subprocess stderr or credentials into the manifest.
            self.data['error'] = {'type': type(error).__name__, 'message': 'Run incomplete; inspect the failed stage log locally'}
        self.save()
