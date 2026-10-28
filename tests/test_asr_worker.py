import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
import wave
from whisper_transcribe.asr_worker import atomic_json, owned_words, result_segments, timed_words, windows, word_segments, run


def raw(tokens):
    return {'text': ''.join(t['text'] for t in tokens), 'sentences': [
        {'text': ''.join(t['text'] for t in tokens), 'start': tokens[0]['start'],
         'end': tokens[-1]['start']+tokens[-1]['duration'], 'tokens': tokens}]}


def token(text, start, duration=.3):
    return {'text': text, 'start': start, 'duration': duration}


class ASRAdapterTests(unittest.TestCase):
    def test_full_worker_clips_final_parakeet_words_with_original_diagnostics(self):
        tail = raw([token(' tail', 4.5, .8)])
        model = MagicMock()
        model.generate.side_effect = [{'text': '', 'sentences': []}, tail]
        core = MagicMock()
        mlx = MagicMock(core=core)
        pcm = MagicMock()
        pcm.astype.return_value.__truediv__.return_value = pcm
        numpy = SimpleNamespace(frombuffer=lambda *a, **kw: pcm, float32='float32')
        modules = {'huggingface_hub': SimpleNamespace(snapshot_download=MagicMock(return_value='/cached')),
                   'mlx_audio': MagicMock(), 'mlx_audio.stt': MagicMock(),
                   'mlx_audio.stt.utils': SimpleNamespace(load_model=lambda path: model),
                   'mlx': mlx, 'mlx.core': core, 'numpy': numpy}
        with tempfile.TemporaryDirectory() as folder:
            audio, output = Path(folder)/'audio.wav', Path(folder)/'output.json'
            with wave.open(str(audio), 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b'\0\0' * (31 * 16000))
            args = SimpleNamespace(engine='parakeet', model=None, audio=audio, output=output,
                                   language='ru', chunk_seconds=30, overlap_seconds=4, allow_download=False)
            core.get_peak_memory.return_value = 0
            with patch.dict('sys.modules', modules), patch.dict('os.environ'), \
                 patch('whisper_transcribe.asr_worker.dataclasses.asdict', side_effect=lambda value: value), \
                 patch('whisper_transcribe.asr_worker.importlib.metadata.version', return_value='test'):
                run(args)
            result = json.loads(output.read_text())
            segment = result['segments'][0]
            word = segment['words'][0]
            self.assertEqual((word['start'], word['end']), (30.5, 31.0))
            self.assertTrue(0 <= segment['start'] <= word['start'] <= word['end'] <= segment['end'] <= result['duration'])
            self.assertAlmostEqual(word['timestamp_clipped']['raw_end'], 31.3)
            self.assertEqual(word['timestamp_clipped']['raw_start'], 30.5)
            self.assertEqual(segment['timestamp_clipped'], [word['timestamp_clipped']])
            self.assertEqual(result['raw_results'][1]['result'], tail)
            self.assertAlmostEqual(tail['sentences'][0]['tokens'][0]['start'] + tail['sentences'][0]['tokens'][0]['duration'], 5.3)

    def test_tail_and_partition(self):
        chunks = list(windows(61, 30, 4))
        self.assertEqual(chunks[-1][1], 61)
        self.assertEqual(chunks[0][2], 0)
        self.assertEqual(chunks[-1][3], 61)
        for previous, following in zip(chunks, chunks[1:]):
            self.assertEqual(previous[3], following[2])
        self.assertEqual(len(list(windows(30, 30, 4))), 1)

    def test_invalid_chunk_parameters(self):
        for chunk, overlap in [(0, 0), (30, -1), (30, 30), (float('nan'), 4)]:
            with self.assertRaises(ValueError):
                list(windows(61, chunk, overlap))

    def test_whole_word_boundary_and_no_overlap_duplicate(self):
        a = raw([token(' first', 26), token(' bound', 27.8), token('ary', 28.1)])
        b = raw([token(' boundary', 1.8, .6), token(' tail', 3)])
        chosen = owned_words(a, 0, 0, 28) + owned_words(b, 26, 28, 56)
        self.assertEqual([w['word'] for w in chosen], ['first', 'boundary', 'tail'])
        self.assertEqual(timed_words(a)[1]['word'], 'boundary')

    def test_long_native_sentence_becomes_bounded_alignment_units(self):
        value = raw([token(' one', 0, 1), token(' two', 20, 1), token(' three', 32, 1), token(' tail', 60, 2)])
        segments = word_segments(value, 61, 'nemotron')
        self.assertEqual(' '.join(s['text'] for s in segments), 'one two three tail')
        self.assertTrue(all(s['end']-s['start'] <= 30 for s in segments))
        self.assertEqual(segments[-1]['end'], 61)
        self.assertIn('timestamp_clipped', segments[-1])
        self.assertEqual(segments[0]['words'][1]['start'], 20)

    def test_end_only_tokens_and_sentence_word_boundary(self):
        value = {'sentences': [
            {'tokens': [{'text': 'one', 'start': 0, 'end': 1}]},
            {'tokens': [{'text': 'two', 'start': 2, 'end': 3}]}]}
        self.assertEqual([w['word'] for w in timed_words(value)], ['one', 'two'])

    def test_clipping_has_explicit_diagnostics(self):
        value = raw([token(' tail', 9, 2)])
        segment = result_segments(value, 0, 10, 'nemotron')[0]
        self.assertEqual(segment['end'], 10)
        self.assertEqual(segment['timestamp_clipped']['raw_end'], 11)

    def test_repeated_real_words_are_not_text_deduplicated(self):
        value = raw([token(' да', 1), token(' да', 2)])
        self.assertEqual(len(owned_words(value, 0, 0, 3)), 2)

    def test_real_times_and_coarse_fallback(self):
        value = raw([token(' пример', 1)])
        segments = result_segments(value, 10, 20, 'nemotron')
        self.assertEqual(segments[0]['start'], 11)
        self.assertEqual(segments[0]['timing_source'], 'decoder_emission')
        self.assertNotIn('words', segments[0])
        coarse = result_segments({'text': 'пример'}, 10, 20, 'parakeet')
        self.assertEqual(coarse[0]['timing_source'], 'audio_window_coarse')
        with self.assertRaises(ValueError):
            result_segments({'sentences': [{'text': 'x', 'start': float('nan'), 'end': 2}]}, 0, 20, 'nemotron')

    def test_atomic_unicode_and_failure_keeps_prior_file(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'результат.json'
            atomic_json(output, {'text': 'пример'})
            with self.assertRaises(ValueError):
                atomic_json(output, {'bad': float('nan')})
            self.assertEqual(json.loads(output.read_text()), {'text': 'пример'})
            self.assertEqual(len(list(Path(folder).iterdir())), 1)


if __name__ == '__main__':
    unittest.main()
