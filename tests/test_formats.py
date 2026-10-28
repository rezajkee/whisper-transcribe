import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch
import wave
from whisper_transcribe.formats import UNKNOWN, attach_speakers, canonical, interval, render
from whisper_transcribe.torch_worker import align_segments, diarization_result, run as run_torch_worker, whisperx_result


class FormatsTests(unittest.TestCase):
    def transcript(self):
        return {'duration': 4, 'segments': [{'id': 's', 'start': 0, 'end': 4, 'text': 'Первый. Второй!',
                'words': [{'word': 'Первый.', 'start': 0, 'end': 1}, {'word': 'Второй!', 'start': 2, 'end': 4}],
                'alignment_coverage_complete': True, 'provenance': {'sources': ['a', 'b']},
                'uncertainties': [{'alternatives': ['два', 'три']}]}]}

    def test_unknown_retained_all_formats_and_legacy(self):
        data = self.transcript()
        for extension, text in render(data).items():
            self.assertIn('Первый. Второй!', text, extension)
            self.assertIn('два', text, extension)
        self.assertIn('Неизвестный', render(data)['md'])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'input.json'
            path.write_text(json.dumps(data))
            subprocess.run(['bash', str(Path(__file__).resolve().parents[1] / 'build_dialog.sh'), str(path)], check=True, capture_output=True)
            self.assertIn('Первый. Второй!', path.with_name('input-диалог.md').read_text())
            self.assertIn('Неизвестный', path.with_name('input-диалог.md').read_text())

    def test_word_speaker_split_preserves_exact_text_and_provenance(self):
        source = self.transcript()
        result = attach_speakers(source, {'turns': [{'start': 0, 'end': 1.5, 'speaker': 'A'}, {'start': 1.5, 'end': 4, 'speaker': 'B'}]})
        self.assertEqual([s['speaker'] for s in result['segments']], ['A', 'B'])
        self.assertEqual(''.join(s['text'] for s in result['segments']), source['segments'][0]['text'])
        self.assertTrue(all(s['provenance'] == source['segments'][0]['provenance'] for s in result['segments']))
        self.assertIn('Аня', render(result, {'A': 'Аня'})['md'])
        self.assertEqual(source, self.transcript())

    def test_overlap_and_ties_are_unknown_with_candidates(self):
        data = self.transcript()
        result = attach_speakers(data, {'turns': [{'start': 0, 'end': 4, 'speaker': 'A'}, {'start': 0, 'end': 4, 'speaker': 'B'}]})
        self.assertEqual(result['segments'][0]['speaker'], UNKNOWN)
        self.assertEqual(len(result['segments'][0]['speaker_candidates']), 2)
        self.assertTrue(result['segments'][0]['overlapping_speech'])

    def test_overlapping_parent_windows_preserve_order_text_words_and_uncertainty(self):
        data = {'duration': 12, 'segments': [
            {'id': 'a', 'start': 1, 'end': 8, 'text': 'One two',
             'words': [{'word': 'One', 'start': 1, 'end': 2},
                       {'word': 'two', 'start': 7, 'end': 8}],
             'alignment_coverage_complete': True, 'provenance': {'source': 'a'},
             'uncertainties': [{'alternatives': ['two', 'too']}]},
            {'id': 'b', 'start': 6, 'end': 10, 'text': 'Three four',
             'words': [{'word': 'Three', 'start': 6, 'end': 6.5},
                       {'word': 'four', 'start': 9, 'end': 10}],
             'alignment_coverage_complete': True, 'provenance': {'source': 'b'}},
            {'id': 'c', 'start': 8, 'end': 11, 'text': 'Five',
             'words': [{'word': 'Five', 'start': 8, 'end': 11}],
             'alignment_coverage_complete': True}]}
        before = copy.deepcopy(data)
        turns = {'turns': [{'start': 0, 'end': 3, 'speaker': 'A'},
                           {'start': 3, 'end': 12, 'speaker': 'B'}]}
        result = attach_speakers(data, turns)
        self.assertEqual(data, before)
        self.assertEqual([s['id'] for s in result['segments']], ['a', 'b', 'c'])
        self.assertEqual([s['start'] for s in result['segments']], [1, 6, 8])
        self.assertEqual([s['text'] for s in result['segments']], [s['text'] for s in before['segments']])
        self.assertEqual([[w['word'] for w in s['words']] for s in result['segments']],
                         [[w['word'] for w in s['words']] for s in before['segments']])
        a = result['segments'][0]
        self.assertEqual((a['start'], a['end']), (1, 8))
        self.assertEqual(a['speaker'], UNKNOWN)
        self.assertTrue(a['mixed_speakers'])
        self.assertEqual(a['speaker_split_status'], 'overlapping_alignment_windows')
        self.assertEqual([w['speaker'] for w in a['words']], ['A', 'B'])
        self.assertEqual([(w['start'], w['end']) for w in a['words']], [(1, 2), (7, 8)])
        self.assertEqual(a['provenance'], before['segments'][0]['provenance'])
        self.assertEqual(a['uncertainties'], before['segments'][0]['uncertainties'])
        self.assertIn('смена спикеров', render(result)['txt'])

    def test_split_allowed_when_last_fragment_start_equals_next_parent_start(self):
        data = self.transcript()
        data['duration'] = 5
        data['segments'].append({'id': 'next', 'start': 2, 'end': 5, 'text': 'Next', 'words': []})
        result = attach_speakers(data, {'turns': [{'start': 0, 'end': 1.5, 'speaker': 'A'},
                                                  {'start': 1.5, 'end': 5, 'speaker': 'B'}]})
        self.assertEqual([s['id'] for s in result['segments']], ['s-speaker-000', 's-speaker-001', 'next'])
        self.assertEqual([s['start'] for s in result['segments']], [0, 2, 2])

    def test_mixed_incomplete_alignment_does_not_attribute_majority(self):
        data = self.transcript()
        data['segments'][0]['words'] = [{'word': 'Первый.', 'start': 0, 'end': 1}]
        data['segments'][0]['alignment_coverage_complete'] = False
        result = attach_speakers(data, {'turns': [{'start': 0, 'end': 3, 'speaker': 'A'}, {'start': 3, 'end': 4, 'speaker': 'B'}]})
        self.assertEqual(result['segments'][0]['speaker'], UNKNOWN)
        self.assertTrue(result['segments'][0]['mixed_speakers'])

    def test_invalid_times_empty_and_silence(self):
        for start, end in [(float('nan'), 1), (-1, 2), (3, 2), (0, 5), (0, float('inf'))]:
            data = self.transcript()
            data['segments'][0].update(start=start, end=end)
            with self.assertRaises(ValueError):
                canonical(data)
        with self.assertRaises(ValueError):
            canonical({'duration': 0, 'segments': []})
        with self.assertRaises(ValueError):
            canonical({'duration': 4})
        self.assertEqual(canonical({'duration': 1, 'segments': []})['segments'], [])
        self.assertIn('WEBVTT', render({'duration': 1, 'segments': []})['vtt'])

    def test_whisperx_native_tail_rounding_clipped_without_text_or_raw_loss(self):
        # Observed failing clip: native end162.031 for162.008s PCM.
        raw = {'language': 'ru', 'segments': [
            {'start': 0.0, 'end': 1.0, 'text': 'Первая фраза.'},
            {'start': 152.581, 'end': 162.031, 'text': 'Хвост!',
             'words': [{'word': 'Хвост!', 'start': 152.581, 'end': 162.031}]},
            {'start': 162.02, 'end': 162.031, 'text': 'Ещё.'}]}
        before = copy.deepcopy(raw)
        result = whisperx_result(raw, 162.008, 'medium', 'ru')
        self.assertEqual(raw, before)
        self.assertEqual(result['raw_result'], before)
        self.assertEqual([s['text'] for s in result['segments']], [s['text'] for s in raw['segments']])
        tail = result['segments'][1]
        self.assertEqual(tail['end'], 162.008)
        self.assertEqual(tail['words'][0]['end'], 162.008)
        self.assertEqual(tail['timestamp_clipped']['raw_end'], 162.031)
        self.assertEqual(tail['words'][0]['timestamp_clipped']['raw_end'], 162.031)
        self.assertEqual((result['segments'][2]['start'], result['segments'][2]['end']), (162.008, 162.008))
        self.assertFalse(result['precise_timing'])
        self.assertTrue(all(s['timing_source'] == 'whisper_decoder' for s in result['segments']))
        self.assertTrue(all(0 <= s['start'] <= s['end'] <= result['duration'] for s in result['segments']))

    def test_whisperx_worker_serializes_clipped_bounds_and_unchanged_raw_result(self):
        raw = {'language': 'ru', 'segments': [{'start': .5, 'end': 1.031, 'text': 'Хвост!'}]}
        class PCM:
            def astype(self, dtype):
                return self
            def __truediv__(self, denominator):
                return self
        numpy = types.SimpleNamespace(frombuffer=lambda *args, **kwargs: PCM(), float32=object())
        model = types.SimpleNamespace(transcribe=lambda *args, **kwargs: copy.deepcopy(raw))
        whisperx = types.SimpleNamespace(load_model=lambda *args, **kwargs: model)
        torch = types.SimpleNamespace(set_num_threads=lambda count: None)
        with tempfile.TemporaryDirectory() as temporary:
            audio, output = Path(temporary) / 'audio.wav', Path(temporary) / 'result.json'
            with wave.open(str(audio), 'wb') as stream:
                stream.setnchannels(1)
                stream.setframerate(16000)
                stream.setsampwidth(2)
                stream.writeframes(bytes(16128 * 2))
            args = argparse.Namespace(audio=audio, output=output, stage='whisperx', cpu_threads=4,
                                      model='medium', language='ru', batch_size=8)
            with patch.dict('sys.modules', {'numpy': numpy, 'torch': torch, 'whisperx': whisperx}), patch.dict(os.environ):
                run_torch_worker(args)
            result = json.loads(output.read_text())
            self.assertEqual(result['duration'], 1.008)
            self.assertEqual(result['segments'][0]['end'], 1.008)
            self.assertEqual(result['segments'][0]['timestamp_clipped']['raw_end'], 1.031)
            self.assertEqual(result['segments'][0]['text'], raw['segments'][0]['text'])
            self.assertEqual(result['raw_result'], raw)
            self.assertFalse(result['precise_timing'])

    def test_whisperx_native_invalid_order_and_nonfinite_are_not_hidden(self):
        for start, end in [(float('nan'), 1), (0, float('inf')), (163, 162.02)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                whisperx_result({'segments': [{'start': start, 'end': end, 'text': 'Текст'}]}, 162.008, 'medium', 'ru')
        with self.assertRaises(ValueError):
            whisperx_result({'segments': [
                {'start': 164, 'end': 165, 'text': 'Первый'},
                {'start': 163, 'end': 164, 'text': 'Второй'}]}, 162.008, 'medium', 'ru')
        with self.assertRaises(ValueError):
            whisperx_result({'segments': [{'start': 0, 'end': 1, 'text': 'Слова', 'words': [
                {'word': 'Один', 'start': 164, 'end': 165},
                {'word': 'два', 'start': 163, 'end': 164}]}]}, 162.008, 'medium', 'ru')

    def test_diarization_observed_hour_tail_clipped_with_exact_raw_retained(self):
        # Actual pyannote3.1 frame-grid tail for 3789.332 seconds of PCM.
        raw = {'turns': [{'start': 3786.9328437500008, 'end': 3789.97034375,
                          'speaker': 'SPEAKER_01'}], 'native_metadata': {'retained': True}}
        before = copy.deepcopy(raw)
        result = diarization_result(raw, 3789.332, 'pyannote/speaker-diarization-3.1')
        self.assertEqual(raw, before)
        self.assertEqual(result['raw_result'], before)
        self.assertEqual(len(result['turns']), len(raw['turns']))
        turn = result['turns'][0]
        self.assertEqual(turn['start'], raw['turns'][0]['start'])
        self.assertEqual(turn['end'], 3789.332)
        self.assertEqual(turn['speaker'], raw['turns'][0]['speaker'])
        self.assertEqual(turn['timestamp_clipped'], {
            'raw_start': raw['turns'][0]['start'], 'raw_end': raw['turns'][0]['end'],
            'reason': 'native diarization estimate outside audio duration'})
        interval(turn, result['duration'])
        # Neither canonical nor preserved native objects alias the caller.
        result['raw_result']['native_metadata']['retained'] = False
        result['turns'][0]['speaker'] = 'changed'
        self.assertEqual(raw, before)

    def test_diarization_negative_outside_and_overlapping_turns_all_retained(self):
        raw = {'turns': [
            {'start': -3, 'end': -1, 'speaker': 'before'},
            {'start': -1, 'end': 2, 'speaker': 'A'},
            {'start': 0, 'end': 4, 'speaker': 'B', 'extra': {'value': 'keep'}},
            {'start': 0, 'end': 1, 'speaker': 'C'},
            {'start': 2, 'end': 5, 'speaker': 'A'},
            {'start': 5, 'end': 6, 'speaker': 'after'}]}
        before = copy.deepcopy(raw)
        result = diarization_result(raw, 4, 'model')
        self.assertEqual(raw, before)
        self.assertEqual(result['raw_result'], before)
        self.assertEqual([t['speaker'] for t in result['turns']], [t['speaker'] for t in raw['turns']])
        self.assertEqual([(t['start'], t['end']) for t in result['turns']],
                         [(0, 0), (0, 2), (0, 4), (0, 1), (2, 4), (4, 4)])
        self.assertEqual(result['turns'][2], raw['turns'][2])
        self.assertEqual([i for i, t in enumerate(result['turns']) if 'timestamp_clipped' in t], [0, 1, 4, 5])
        for turn in result['turns']:
            interval(turn, result['duration'])

    def test_diarization_native_invalid_times_and_order_not_hidden_by_clipping(self):
        for start, end in [(float('nan'), 1), (0, float('inf')), (-float('inf'), 0),
                           (6, 5), (-1, -2)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                diarization_result({'turns': [{'start': start, 'end': end, 'speaker': 'A'}]}, 4, 'model')
        # The backend promises chronological starts. Both of these would become
        # equal after clipping, so check native order first; overlaps stay valid.
        for starts in [(6, 5), (-1, -2)]:
            with self.subTest(starts=starts), self.assertRaises(ValueError):
                diarization_result({'turns': [
                    {'start': start, 'end': start + 1, 'speaker': 'A'} for start in starts]}, 4, 'model')

    def test_diarization_invalid_duration_and_empty_turns(self):
        for duration in [0, -1, float('nan'), float('inf')]:
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                diarization_result({'turns': []}, duration, 'model')
        for raw in [{}, {'turns': None}, {'turns': {}}]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                diarization_result(raw, 4, 'model')
        result = diarization_result({'turns': []}, 4, 'model')
        self.assertEqual(result['turns'], [])
        self.assertEqual(result['raw_result'], {'turns': []})

    def test_diarization_worker_serializes_clipped_bounds_and_exact_native_turns(self):
        turns = [{'start': -.01, 'end': .8, 'speaker': 'A'},
                 {'start': .5, 'end': 1.031, 'speaker': 'B'},
                 {'start': 1.02, 'end': 1.04, 'speaker': 'A'}]
        seen = []
        class PCM:
            def astype(self, dtype):
                return self
            def __truediv__(self, denominator):
                return self
        class Pipeline:
            def __init__(self, **kwargs):
                seen.append(kwargs)
            def __call__(self, audio, **kwargs):
                seen.append(kwargs)
                return types.SimpleNamespace(itertuples=lambda: [types.SimpleNamespace(**t) for t in turns])
        numpy = types.SimpleNamespace(frombuffer=lambda *args, **kwargs: PCM(), float32=object())
        torch = types.SimpleNamespace(set_num_threads=lambda count: None)
        whisperx = types.ModuleType('whisperx')
        diarize = types.ModuleType('whisperx.diarize')
        diarize.DiarizationPipeline = Pipeline
        with tempfile.TemporaryDirectory() as temporary:
            audio, output = Path(temporary) / 'audio.wav', Path(temporary) / 'result.json'
            with wave.open(str(audio), 'wb') as stream:
                stream.setnchannels(1)
                stream.setframerate(16000)
                stream.setsampwidth(2)
                stream.writeframes(bytes(16128 * 2))
            args = argparse.Namespace(audio=audio, output=output, stage='diarize', cpu_threads=4,
                                      diarize_model='pyannote/speaker-diarization-3.1',
                                      min_speakers=2, max_speakers=3)
            with patch.dict('sys.modules', {'numpy': numpy, 'torch': torch, 'whisperx': whisperx,
                                           'whisperx.diarize': diarize}), patch.dict(os.environ):
                run_torch_worker(args)
            result = json.loads(output.read_text())
        self.assertEqual(seen, [{'model_name': args.diarize_model, 'device': 'cpu'},
                                {'min_speakers': 2, 'max_speakers': 3}])
        self.assertEqual(result['duration'], 1.008)
        self.assertEqual(result['raw_result'], {'turns': turns})
        self.assertEqual([(t['start'], t['end']) for t in result['turns']],
                         [(0, .8), (.5, 1.008), (1.008, 1.008)])
        self.assertEqual([t['speaker'] for t in result['turns']], [t['speaker'] for t in turns])
        for turn in result['turns']:
            interval(turn, result['duration'])
        self.assertEqual(result['worker_measurements']['peak_rss_scope'], 'this_worker_process')

    def test_align_preserves_empty_result_tail_and_metadata(self):
        source = self.transcript()
        result = align_segments(source, [0] * 64000, None, None, lambda *a, **k: {'segments': []}, 4)
        segment = result['segments'][0]
        self.assertEqual(segment['text'], source['segments'][0]['text'])
        self.assertEqual(segment['provenance'], source['segments'][0]['provenance'])
        self.assertEqual(segment['uncertainties'], source['segments'][0]['uncertainties'])
        self.assertEqual(segment['timing_status'], 'unaligned')
        self.assertFalse(result['precise_timing'])

    def test_alignment_crops_bounded_and_retains_untimed_words(self):
        source = self.transcript()
        seen = []
        def align(units, model, metadata, audio, device, **kwargs):
            seen.append((units, len(audio)))
            return {'segments': [{'words': [{'word': 'Первый.', 'start': .5, 'end': 1}, {'word': 'Второй!'}]}]}
        result = align_segments(source, [0] * 64000, None, None, align, 4)
        self.assertEqual(seen[0][1], 64000)
        self.assertFalse(result['precise_timing'])
        self.assertEqual(result['segments'][0]['words'][1]['word'], 'Второй!')

    def test_alignment_crossings_never_reorder_text(self):
        data = {'duration': 10, 'segments': [
            {'id': 'a', 'start': 1, 'end': 2, 'text': 'Первый', 'timing_source': 'decoder'},
            {'id': 'b', 'start': 3, 'end': 4, 'text': 'Второй', 'timing_source': 'decoder'}]}
        positions = iter([(4, 5), (1, 2)])
        def align(units, *args, **kwargs):
            start, end = next(positions)
            return {'segments': [{'words': [{'word': units[0]['text'], 'start': start, 'end': end}]}]}
        result = align_segments(data, [0] * 160000, None, None, align, 10)
        self.assertEqual([s['text'] for s in result['segments']], ['Первый', 'Второй'])
        self.assertEqual([s['start'] for s in result['segments']], [1, 3])
        self.assertTrue(all(s['alignment_crossing'] for s in result['segments']))
        self.assertFalse(result['precise_timing'])


if __name__ == '__main__':
    unittest.main()
