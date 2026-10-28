import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave
from whisper_transcribe.cli import parser
from whisper_transcribe.pipeline import run, runtime_identity
from whisper_transcribe.state import Journal, file_hash, run_lock

FAKE = r'''
import argparse, json, os, pathlib, sys, wave
p=argparse.ArgumentParser()
p.add_argument('--stage'); p.add_argument('--audio'); p.add_argument('--output'); p.add_argument('--input'); p.add_argument('--parakeet'); p.add_argument('--nemotron')
a, rest=p.parse_known_args()
with open(os.environ['FAKE_EVENTS'], 'a') as stream: stream.write(a.stage+'\n')
if os.environ.get('FAIL_STAGE') == a.stage:
    print('fixture failure', file=sys.stderr); sys.exit(17)
with wave.open(a.audio) as stream: duration=stream.getnframes()/stream.getframerate()
if a.stage == 'diarize': result={'duration': duration, 'turns':[{'start': 0, 'end': duration, 'speaker': 'SPEAKER_00'}]}
elif a.stage == 'align':
    result=json.loads(pathlib.Path(a.input).read_text())
    for s in result['segments']:
        s.update(words=[{'word': s['text'], 'start': s['start'], 'end': s['end']}], timing_status='aligned', alignment_coverage_complete=True)
elif a.stage == 'merge':
    result=json.loads(pathlib.Path(a.parakeet).read_text())
    result['merge_fixture']=True
    result['segments'][0]['uncertainties']=[{'alternatives':['один','два']}]
else: result={'duration':duration,'language':'ru','segments':[{'id':a.stage+'-0','start':0,'end':duration,'text':'Привет мир!','timing_source':'decoder'}], 'raw_results':[{'text':'Привет мир!'}]}
pathlib.Path(a.output).write_text(json.dumps(result, ensure_ascii=False))
'''


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / 'Запись v.1 с пробелом.wav'
        with wave.open(str(self.source), 'wb') as stream:
            stream.setnchannels(2); stream.setsampwidth(2); stream.setframerate(8000)
            stream.writeframes(b'\x00\x00' * 2 * 8000)
        self.fixture = self.root / 'fake_worker.py'
        self.fixture.write_text(FAKE)
        self.events = self.root / 'events'
        self.environment = patch.dict(os.environ, {'FAKE_EVENTS': str(self.events)})
        self.environment.start()
        self.models = patch('whisper_transcribe.pipeline.model_identity', side_effect=lambda name: {'model': name, 'fixture': True})
        self.models.start()
        self.snapshots = patch('whisper_transcribe.state.memory_snapshot', return_value={'fixture': True})
        self.snapshots.start()
        self.identities = {'mlx': {'version': 1}, 'torch': {'version': 1}}
        self.args = parser().parse_args([str(self.source), '--engine', 'parakeet', '-o', str(self.root / 'out')])
        self.args.max_speakers = self.args.min_speakers

    def tearDown(self):
        self.snapshots.stop(); self.models.stop(); self.environment.stop(); self.temporary.cleanup()

    def run_pipeline(self):
        return run(self.args, worker_command=[sys.executable, str(self.fixture)], identities=self.identities)

    def stages(self):
        return self.events.read_text().splitlines() if self.events.exists() else []

    def manifest(self):
        return json.loads(next((self.root / 'out/.runs').glob('*/manifest.json')).read_text())

    def test_normalizes_wav_unicode_safe_names_and_resumes(self):
        before = file_hash(self.source)
        result = self.run_pipeline()
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(file_hash(self.source), before)
        normalized = next((self.root / 'out/.runs').glob('*/stages/normalize-*/audio.wav'))
        with wave.open(str(normalized)) as stream:
            self.assertEqual((stream.getnchannels(), stream.getframerate(), stream.getsampwidth()), (1, 16000, 2))
        first = self.stages()
        self.run_pipeline()
        self.assertEqual(self.stages(), first)
        self.assertEqual(len(result['outputs']), 6)
        self.assertTrue(all(Path(path).exists() for path in result['outputs']))
        self.assertTrue(all('v.1' in Path(path).name for path in result['outputs']))
        self.assertTrue(all(record['current_run_validated'] for record in self.manifest()['stages'].values()))

    def test_same_stem_different_content_isolation(self):
        first = self.run_pipeline()
        other = self.root / 'other' / self.source.name
        other.parent.mkdir()
        other.write_bytes(self.source.read_bytes() + b'changed')
        self.args.input = str(other)
        second = self.run_pipeline()
        self.assertNotEqual(first['manifest'], second['manifest'])
        self.assertTrue(set(first['outputs']).isdisjoint(second['outputs']))

    def test_failure_keeps_completed_stage_and_resume_skips_it(self):
        with patch.dict(os.environ, {'FAIL_STAGE': 'align'}):
            with self.assertRaises(RuntimeError): self.run_pipeline()
        manifest = self.manifest()
        self.assertEqual(manifest['status'], 'failed')
        self.assertEqual(manifest['stages']['parakeet']['status'], 'complete')
        self.assertEqual(manifest['stages']['align']['status'], 'failed')
        self.assertFalse(manifest['outputs'])
        self.run_pipeline()
        self.assertEqual(self.stages().count('parakeet'), 1)
        self.assertEqual(self.stages().count('align'), 2)
        self.assertEqual(self.manifest()['status'], 'complete')

    def test_interrupt_keeps_completed_stages(self):
        from whisper_transcribe.pipeline import execute
        def interrupt(command, output):
            if 'align' in command:
                raise KeyboardInterrupt()
            execute(command, output)
        with patch('whisper_transcribe.pipeline.execute', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt): self.run_pipeline()
        self.assertEqual(self.manifest()['status'], 'interrupted')
        self.run_pipeline()
        self.assertEqual(self.stages().count('parakeet'), 1)

    def test_granular_mapping_speaker_runtime_and_artifact_invalidation(self):
        self.run_pipeline()
        initial = self.stages()
        mapping = self.root / 'mapping.json'
        mapping.write_text('{"SPEAKER_00":"Иван"}')
        self.args.mapping = str(mapping)
        result = self.run_pipeline()
        self.assertEqual(self.stages(), initial)
        self.assertIn('Иван', Path(next(p for p in result['outputs'] if p.endswith('.md'))).read_text())
        self.args.min_speakers = 1
        self.run_pipeline()
        self.assertEqual(self.stages()[len(initial):], ['diarize'])
        self.identities['mlx'] = {'version': 2}
        prior = len(self.stages())
        self.run_pipeline()
        self.assertEqual(self.stages()[prior:], ['parakeet'])
        # A tampered artifact is rebuilt even with unchanged configuration.
        manifest = self.manifest()
        directory = Path(result['manifest']).parent
        artifact = directory / manifest['stages']['parakeet']['path']
        artifact.write_text('{}')
        prior = len(self.stages())
        self.run_pipeline()
        self.assertEqual(self.stages()[prior:], ['parakeet'])
        # Export validation covers every companion file, not just canonical JSON.
        export = directory / self.manifest()['stages']['export']['path']
        (export.parent / 'transcript.md').write_text('tampered')
        self.run_pipeline()
        self.assertIn('Привет мир!', Path(next(p for p in result['outputs'] if p.endswith('.md'))).read_text())

    def test_model_identity_invalidates_asr(self):
        self.run_pipeline()
        with patch('whisper_transcribe.pipeline.model_identity', side_effect=lambda name: {'model': name, 'changed': name.startswith('mlx-')}):
            self.run_pipeline()
        self.assertEqual(self.stages().count('parakeet'), 2)

    def test_dual_glossary_invalidates_merge_not_asr(self):
        self.args.engine = 'dual'
        glossary = self.root / 'glossary.txt'
        glossary.write_text('термин')
        self.args.glossary = str(glossary)
        result = self.run_pipeline()
        self.assertEqual(self.stages(), ['parakeet', 'nemotron', 'diarize', 'merge', 'align'])
        final = json.loads(Path(next(path for path in result['outputs'] if path.endswith('.json'))).read_text())
        self.assertEqual(final['segments'][0]['uncertainties'][0]['alternatives'], ['один', 'два'])
        self.assertEqual(set(final['raw_hypotheses']), {'parakeet', 'nemotron'})
        glossary.write_text('другой термин')
        before = len(self.stages())
        self.run_pipeline()
        self.assertEqual(self.stages()[before:], ['merge'])
        self.assertEqual(self.stages().count('parakeet'), 1)
        self.assertEqual(self.stages().count('nemotron'), 1)
        self.assertEqual(self.manifest()['stages']['merge']['status'], 'complete')

    def test_reconcile_source_change_refreshes_merge_and_changed_downstream(self):
        # Use actual source_identity hashes and the normal Journal. All workers
        # are fixtures; a changed merge output must invalidate its dependants.
        import shutil
        from whisper_transcribe import pipeline
        package = self.root / 'package'
        shutil.copytree(pipeline.ROOT / 'whisper_transcribe', package / 'whisper_transcribe',
                        ignore=shutil.ignore_patterns('__pycache__'))
        reconcile_file = package / 'whisper_transcribe/reconcile.py'
        fixture = self.fixture.read_text().replace("    result['merge_fixture']=True",
            "    import hashlib\n    result['merge_fixture']=hashlib.sha256(pathlib.Path(os.environ['MERGE_SOURCE_FILE']).read_bytes()).hexdigest()")
        self.fixture.write_text(fixture)
        self.args.engine = 'dual'
        with patch.object(pipeline, 'ROOT', package), patch.dict(os.environ, {'MERGE_SOURCE_FILE': str(reconcile_file)}):
            self.run_pipeline()
            first = copy.deepcopy(self.manifest())
            before = len(self.stages())
            reconcile_file.write_text(reconcile_file.read_text() + '\n# synthetic source identity change\n')
            self.run_pipeline()
            second = copy.deepcopy(self.manifest())
            self.assertEqual(self.stages()[before:], ['merge', 'align'])
            for name in ('normalize', 'parakeet', 'nemotron', 'diarize'):
                self.assertEqual(first['stages'][name]['path'], second['stages'][name]['path'])
                self.assertEqual(first['stages'][name]['sha256'], second['stages'][name]['sha256'])
            for name in ('merge', 'align', 'speakers', 'export'):
                self.assertNotEqual(first['stages'][name]['key'], second['stages'][name]['key'])
                self.assertNotEqual(first['stages'][name]['sha256'], second['stages'][name]['sha256'])
            self.assertNotEqual(first['stages']['merge']['identity']['source']['reconcile.py'],
                                second['stages']['merge']['identity']['source']['reconcile.py'])
            before = self.stages()
            self.run_pipeline()
            self.assertEqual(self.stages(), before)
            self.assertTrue(all(stage['current_run_validated'] for stage in self.manifest()['stages'].values()))

    def test_switching_to_identical_dual_output_refreshes_provenance(self):
        # A valid merger may choose A without adding text or metadata. Its bytes
        # can match the single ASR exactly, so alignment can correctly be reused.
        fixture = self.fixture.read_text().replace("    result['merge_fixture']=True\n", '').replace("    result['segments'][0]['uncertainties']=[{'alternatives':['один','два']}]\n", '')
        self.fixture.write_text(fixture)
        first = self.run_pipeline()
        first_data = json.loads(Path(next(path for path in first['outputs'] if path.endswith('.json'))).read_text())
        first_speakers = self.manifest()['stages']['speakers']['path']
        self.assertEqual(first_data['engine'], 'parakeet')
        self.args.engine = 'dual'
        result = self.run_pipeline()
        final = json.loads(Path(next(path for path in result['outputs'] if path.endswith('.json'))).read_text())
        self.assertEqual(self.stages(), ['parakeet', 'diarize', 'align', 'nemotron', 'merge'])
        self.assertEqual(final['engine'], 'dual')
        self.assertEqual(set(final['raw_hypotheses']), {'parakeet', 'nemotron'})
        self.assertNotEqual(self.manifest()['stages']['speakers']['path'], first_speakers)
        for path in final['raw_hypotheses'].values():
            self.assertTrue((Path(result['manifest']).parent / path).is_file())
        self.args.engine = 'parakeet'
        back = self.run_pipeline()
        restored = json.loads(Path(next(path for path in back['outputs'] if path.endswith('.json'))).read_text())
        self.assertEqual(restored['engine'], 'parakeet')
        self.assertEqual(set(restored['raw_hypotheses']), {'parakeet'})

    def test_exclusive_lock(self):
        with run_lock(self.root / 'lock'):
            process = subprocess.run([sys.executable, '-c', 'from whisper_transcribe.state import run_lock; import sys\nwith run_lock(sys.argv[1]): pass', str(self.root / 'lock')], capture_output=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn(b'Another process owns', process.stderr)

    def test_failed_changed_run_unpublishes_old_success(self):
        result = self.run_pipeline()
        self.args.min_speakers = 1
        with patch.dict(os.environ, {'FAIL_STAGE': 'diarize'}):
            with self.assertRaises(RuntimeError): self.run_pipeline()
        self.assertTrue(all(not Path(path).exists() for path in result['outputs']))
        manifest = self.manifest()
        self.assertEqual(manifest['status'], 'failed')
        self.assertFalse(manifest['stages']['align']['current_run_validated'])
        self.assertTrue(manifest['stages']['parakeet']['current_run_validated'])
        self.run_pipeline()
        self.assertTrue(all(Path(path).exists() for path in result['outputs']))

    def test_legacy_shell_keeps_basename_and_offline_empty_arrays(self):
        venv = self.root / 'legacy-env'
        binary = venv / 'bin'
        binary.mkdir(parents=True)
        (binary / 'activate').write_text('export PATH="' + str(binary) + ':$PATH"\n')
        command = binary / 'whisperx'
        command.write_text('#!' + sys.executable + '\nimport json, os, pathlib, sys\npathlib.Path(os.environ["LEGACY_ARGS"]).write_text(json.dumps({"args":sys.argv[1:],"offline":os.environ.get("HF_HUB_OFFLINE")}))\n')
        command.chmod(0o755)
        report = self.root / 'legacy-args.json'
        environment = dict(os.environ, WHISPER_VENV=str(venv), WHISPER_WORK_DIR=str(self.root / 'legacy-work'), LEGACY_ARGS=str(report))
        environment.pop('HF_TOKEN', None)
        subprocess.run(['bash', str(Path(__file__).resolve().parents[1] / 'transcribe.sh'), str(self.source), '--no-diarize', '-o', str(self.root / 'legacy-out')], env=environment, capture_output=True, check=True)
        arguments = json.loads(report.read_text())
        self.assertEqual(Path(arguments['args'][0]).name, self.source.name)
        self.assertNotEqual(Path(arguments['args'][0]), self.source)
        self.assertNotIn('--diarize', arguments['args'])
        self.assertNotIn('--hf_token', arguments['args'])
        self.assertIn('--model_cache_only', arguments['args'])
        self.assertEqual(arguments['offline'], '1')

    def test_packaged_model_asset_mutation_changes_runtime_and_asr_key(self):
        distribution = self.root / 'fixture_vad-1.0.dist-info'
        distribution.mkdir()
        (distribution / 'METADATA').write_text('Metadata-Version: 2.1\nName: fixture-vad\nVersion: 1.0\n')
        assets = [self.root / ('fixture_model' + suffix) for suffix in ('.bin', '.safetensors', '.onnx', '.pt')]
        for asset in assets:
            asset.write_bytes(b'original weights')
        (distribution / 'RECORD').write_text(''.join(asset.name + ',,\n' for asset in assets))
        with patch.dict(os.environ, {'PYTHONPATH': str(self.root) + os.pathsep + os.environ.get('PYTHONPATH', '')}):
            previous = runtime_identity(sys.executable)
            self.assertEqual(previous['packages']['fixture-vad'], '1.0')
            self.args.engine = 'whisperx'
            self.identities['torch'] = previous
            self.run_pipeline()
            for asset in assets:
                with self.subTest(suffix=asset.suffix):
                    stat = asset.stat()
                    asset.write_bytes(b'modified weights')
                    os.utime(asset, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                    current = runtime_identity(sys.executable)
                    self.assertEqual(current['packages'], previous['packages'])
                    self.assertNotEqual(current['installed_code_sha256'], previous['installed_code_sha256'])
                    previous = current
            self.identities['torch'] = previous
            self.run_pipeline()
            self.assertEqual(self.stages().count('whisperx'), 2)

    def test_runtime_probe_imports_no_heavy_models(self):
        identity = runtime_identity(sys.executable)
        self.assertIn('python', identity)
        self.assertEqual(len(identity['installed_code_sha256']), 64)


if __name__ == '__main__':
    unittest.main()
