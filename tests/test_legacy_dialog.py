"""Exercise the public Bash rebuild entry point without model dependencies."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LegacyDialogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='диалог с пробелами ')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        # Copy the entry point and its pure renderer to exercise installation
        # paths containing spaces/Cyrillic independently of the checkout name.
        self.install = self.directory / 'проект с пробелами'
        self.install.mkdir()
        shutil.copy2(ROOT / 'build_dialog.sh', self.install)
        package = self.install / 'whisper_transcribe'
        package.mkdir()
        (package / '__init__.py').write_text('')
        shutil.copy2(ROOT / 'whisper_transcribe' / 'formats.py', package)

    def rebuild(self, data, mapping=None, missing_mapping=False):
        source = self.directory / 'запись встречи.json'
        source.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        self.output = self.directory / 'готовый диалог.md'
        args = ['bash', str(self.install / 'build_dialog.sh'), str(source),
                '-o', str(self.output), '-t', 'Совещание с командой']
        if mapping is not None or missing_mapping:
            path = self.directory / 'имена спикеров.json'
            if not missing_mapping:
                path.write_text(json.dumps(mapping, ensure_ascii=False), encoding='utf-8')
            args.extend(['--mapping', str(path)])
        return subprocess.run(args, cwd='/', capture_output=True, text=True)

    def canonical(self):
        return {'schema_version': 1, 'engine': 'dual', 'duration': 3,
                'provisional': True, 'quality': {'status': 'review_required'},
                'boundaries': [{'time': 1.25}], 'raw_result': {'debug': 'SECRET_DEBUG'},
                'segments': [{'id': 's', 'start': .125, 'end': 2.875,
                              'speaker': 'UNKNOWN', 'text': 'Получили два заказа.',
                              'boundary_review_required': True, 'timing_status': 'unaligned',
                              'mixed_speakers': True, 'overlapping_speech': True,
                              'uncertainties': [{'start': .5, 'critical': ['numbers'],
                                                 'alternatives': {'A': 'два заказа', 'B': 'три заказа'}}]}]}

    def test_canonical_review_metadata_and_mapping_title_survive(self):
        result = self.rebuild(self.canonical(), {'UNKNOWN': 'Голос не определён'})
        self.assertEqual(result.returncode, 0, result.stderr)
        text = self.output.read_text(encoding='utf-8')
        for expected in ['# Совещание с командой', '**Голос не определён**',
                         'Предварительный результат', 'Стыки независимых окон',
                         '00:00:01.250', '00:00:00.125 – 00:00:02.875',
                         'Получили два заказа.', 'стык независимых окон: требуется прослушивание',
                         'таймкоды не подтверждены выравниванием', 'назначение неоднозначно',
                         'наложение голосов', 'выбор не подтверждён аудио (числа)',
                         'A: «два заказа» / B: «три заказа»']:
            self.assertIn(expected, text)
        self.assertNotIn('SECRET_DEBUG', text)
        self.assertNotIn('raw_result', text)

    def test_canonical_unknown_and_list_alternatives_survive(self):
        data = self.canonical()
        data['segments'][0]['uncertainties'][0]['alternatives'] = ['два', 'три']
        result = self.rebuild(data)
        self.assertEqual(result.returncode, 0, result.stderr)
        text = self.output.read_text(encoding='utf-8')
        self.assertIn('**Неизвестный**', text)
        self.assertIn('«два» / «три»', text)

    def test_inherited_uncertainty_is_displayed_once(self):
        data = self.canonical()
        original = data['segments'][0]
        original.update(id='s-speaker-000', source_segment_id='s', end=1, speaker='A')
        second = copy.deepcopy(original)
        second.update(id='s-speaker-001', start=1, end=3, speaker='B', text='Продолжение.')
        data['segments'].append(second)
        result = self.rebuild(data, {'A': 'Аня', 'B': 'Борис'})
        self.assertEqual(result.returncode, 0, result.stderr)
        text = self.output.read_text(encoding='utf-8')
        self.assertEqual(text.count('варианты ASR'), 1)
        self.assertEqual(text.count('A: «два заказа» / B: «три заказа»'), 1)
        self.assertIn('**Аня**', text)
        self.assertIn('**Борис**', text)

    def test_invalid_canonical_does_not_fall_back_or_overwrite(self):
        variants = []
        for key, value in [('duration', 0), ('schema_version', 2)]:
            data = self.canonical()
            data[key] = value
            variants.append(data)
        data = self.canonical()
        del data['duration']
        variants.append(data)
        data = self.canonical()
        data['segments'][0]['end'] = 4
        variants.append(data)
        for data in variants:
            with self.subTest(data=data):
                result = self.rebuild(data)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.output.exists())

    def test_canonical_invalid_or_missing_mapping_is_visible(self):
        for mapping in [[], '', {'A': 42}]:
            with self.subTest(mapping=mapping):
                result = self.rebuild(self.canonical(), mapping)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Speaker mapping', result.stderr)
                self.assertFalse(self.output.exists())
        result = self.rebuild(self.canonical(), missing_mapping=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_legacy_adjacent_grouping_unknown_and_optional_names(self):
        data = {'segments': [
            {'start': 0, 'end': 1, 'speaker': 'A', 'text': 'Первая   фраза.'},
            {'start': 1, 'end': 2, 'speaker': 'A', 'text': 'Вторая фраза.'},
            {'start': 2, 'end': 3, 'text': 'Без спикера.'},
            {'start': 3, 'end': 4, 'speaker': 'A', 'text': 'Снова первый.'}]}
        for mapping, name in [(None, 'Спикер 1'), ({'A': 'Аня'}, 'Аня')]:
            with self.subTest(mapping=mapping):
                result = self.rebuild(data, mapping)
                self.assertEqual(result.returncode, 0, result.stderr)
                text = self.output.read_text(encoding='utf-8')
                self.assertIn('# Совещание с командой — транскрипт', text)
                self.assertIn('Первая фраза. Вторая фраза.', text)
                self.assertIn(f'**{name}** `[0:00 – 0:02]`', text)
                self.assertIn('**Неизвестный** `[0:02 – 0:03]`', text)
                self.assertIn('Без спикера.', text)
                self.assertEqual(text.count(f'**{name}**'), 2)
                self.assertIn('(3 блоков)', result.stdout)


if __name__ == '__main__':
    unittest.main()
