import json
from pathlib import Path
import tempfile
import unittest
from whisper_transcribe.evaluate import edit_counts, mark_verified, metrics, score_manifest


class EvaluationTests(unittest.TestCase):
    def test_known_wer_cer_case_punctuation_number_policy(self):
        result = metrics('The cat sat.', 'the dog sat!')
        self.assertEqual(result['wer']['substitutions'], 1)
        self.assertAlmostEqual(result['wer']['rate'], 1/3)
        self.assertEqual(metrics('Привет, Мир!', 'привет мир')['cer']['edits'], 0)
        self.assertEqual(metrics('2', 'two')['wer']['edits'], 1)
        self.assertEqual(edit_counts('kitten', 'sitting')['edits'], 3)
        for a, b, deleted, inserted in [(['a'], [], 1, 0), ([], ['a','b'], 0, 2), (['a','b'], ['a'], 1, 0)]:
            result = edit_counts(a,b)
            self.assertEqual((result['deletions'], result['insertions']), (deleted, inserted))
        self.assertIsNone(metrics('', 'noise')['wer']['rate'])
        self.assertEqual(metrics('', 'noise')['wer']['insertions'], 1)
        self.assertIsNone(metrics('', '')['cer']['rate'])

    def fixture(self, root):
        root = Path(root)
        (root/'a.txt').write_text('один два')
        (root/'b.txt').write_text('')
        (root/'holdout.txt').write_text('секрет')
        (root/'candidate.json').write_text(json.dumps({'segments':[{'text':'один три'}]}))
        (root/'candidate-empty.txt').write_text('шум')
        manifest = root/'manifest.json'
        manifest.write_text(json.dumps({'items':[
            {'id':'a','reference':'a.txt','split':'tuning','reference_verified':False},
            {'id':'b','reference':'b.txt','split':'tuning','reference_verified':False},
            {'id':'h','reference':'holdout.txt','split':'holdout','reference_verified':False}]}))
        return manifest, {'a':root/'candidate.json','b':root/'candidate-empty.txt'}

    def test_verification_acknowledgement_hash_guard_and_holdout_separation(self):
        with tempfile.TemporaryDirectory() as root:
            manifest, candidates = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'human verified'):
                score_manifest(manifest, candidates, 'tuning')
            with self.assertRaises(ValueError):
                mark_verified(manifest, 'a')
            for item in ('a','b'):
                mark_verified(manifest, item, acknowledged=True)
            result = score_manifest(manifest, candidates, 'tuning')
            self.assertEqual(result['aggregate']['wer']['edits'], 2)
            self.assertEqual(result['aggregate']['wer']['reference_length'], 2)
            self.assertEqual(result['aggregate']['wer']['rate'], 1)
            self.assertEqual(result['items'][0]['reference_text'], 'один два')
            self.assertIsNone(result['items'][1]['wer']['rate'])
            self.assertFalse(result['quality_promotion'])
            self.assertIsNone(result['semantic_metric'])
            with self.assertRaisesRegex(ValueError, 'human verified'):
                score_manifest(manifest, {'h':candidates['a']}, 'holdout')
            (Path(root)/'a.txt').write_text('исправлено после проверки')
            with self.assertRaisesRegex(ValueError, 'changed after verification'):
                score_manifest(manifest, candidates, 'tuning')

    def test_duplicate_unknown_and_missing_candidates_refused(self):
        with tempfile.TemporaryDirectory() as root:
            manifest, candidates = self.fixture(root)
            with self.assertRaises(ValueError):
                score_manifest(manifest, {'a':candidates['a']}, 'tuning')
            with self.assertRaises(ValueError):
                mark_verified(manifest, 'unknown', True)
            data = json.loads(manifest.read_text())
            data['items'].append(data['items'][0])
            manifest.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                score_manifest(manifest, candidates, 'tuning')
