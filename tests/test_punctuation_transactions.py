"""Atomic acoustic-unit punctuation decisions use immutable full context."""
import copy
import json
import unittest
from unittest.mock import patch

from whisper_transcribe import reconcile as r
from test_punctuation_safety import source as base_source, run, downstream, REVIEW
from whisper_transcribe.formats import canonical, render


def source(*texts):
    data = base_source(*texts)
    data['duration'] = max(60, len(texts) * 20)
    for i, segment in enumerate(data['segments']):
        segment.update(start=i * 20, end=i * 20 + 19)
    return data


class PunctuationTransactionTests(unittest.TestCase):
    def test_one_invalid_unit_does_not_cancel_safe_units(self):
        for bad_index in range(3):
            for old_end, new_end in [('.', ','), ('.', ''), (',', '.')]:
                with self.subTest(bad_index=bad_index, old=old_end, new=new_end):
                    # Four units: every tested offending cut is nonfinal.
                    texts = ['alpha beta', 'gamma delta', 'epsilon zeta', 'eta theta']
                    texts[bad_index] += old_end
                    original = source(*texts)
                    original['duration'] = 90
                    def edit(d):
                        if d['text'] == texts[bad_index]:
                            return 'Alpha, beta' + new_end if bad_index == 0 else d['text'][:-1] + new_end
                        return d['text'][0].upper() + d['text'][1:]
                    result, _ = run(original, edit)
                    expected = [t[0].upper() + t[1:] for t in texts]
                    expected[bad_index] = texts[bad_index]
                    self.assertEqual([s['text'] for s in result['segments']], expected)
                    self.assertTrue(r.punctuation_only(' '.join(texts), ' '.join(expected)))
                    bad = result['segments'][bad_index]
                    self.assertEqual(bad['provenance']['punctuation']['transaction']['status'],
                                     'unchanged' if new_end == '.' else 'rejected')
                    entries = result['reconciliation']['punctuation_journal']
                    for j in entries:
                        if j['segment_id'] == bad['id']:
                            self.assertEqual(j['validation'], 'review_required')
                            self.assertEqual(j['source'], j['result'])
                            if new_end == '.':
                                self.assertEqual(j['error'], 'New terminal punctuation at continuing chunk boundary')
                            else:
                                self.assertEqual(j['fallback_stage'], 'acoustic_transaction')
                                self.assertEqual(j['rollback_history'][-1]['validation'], 'accepted')
                        else:
                            self.assertEqual(j['validation'], 'accepted')

    def test_transaction_rejects_all_its_chunks_preserving_earlier_errors(self):
        text = 'alpha beta ' * 16 + 'omega.'
        data = source(text, 'next words')
        chunks = r.punctuation_chunks(text)
        def edit(d):
            if d['id'].endswith('punct-0001'):
                return 'lexical corruption'
            return d['text'].replace('alpha', 'Alpha').replace('omega.', 'omega,')
        result, _ = run(data, edit)
        self.assertEqual(result['segments'][0]['text'], text)
        entries = [j for j in result['reconciliation']['punctuation_journal']
                   if j['segment_id'] == result['segments'][0]['id']]
        self.assertGreater(len(entries), 2)
        self.assertTrue(all(j['validation'] == 'review_required' and j['result'] == j['source'] for j in entries))
        prior = entries[1]['rollback_history'][-1]
        self.assertEqual(prior['validation'], 'review_required')
        self.assertEqual(prior['error'], 'Punctuation response changed protected text elements')
        self.assertIn('lexical corruption', entries[1]['model_response'])
        self.assertTrue(any(u['reason'] == prior['error'] for u in result['segments'][0]['uncertainties']))

    def test_cumulative_candidate_not_independent_isolation(self):
        # Instrument the safety oracle to model any contextual interaction.
        # Neither proposal alone fails. Only their combined state is rejected.
        actual = r.punctuation_only
        seen = []
        def contextual(old, new):
            if old == 'alpha beta gamma delta epsilon zeta':
                seen.append(new)
                if new.startswith('Alpha beta Gamma delta'):
                    return False
            return actual(old, new)
        with patch.object(r, 'punctuation_only', side_effect=contextual):
            result, _ = run(source('alpha beta', 'gamma delta', 'epsilon zeta'),
                            lambda d: d['text'].capitalize())
        self.assertEqual([s['text'] for s in result['segments']], ['Alpha beta', 'gamma delta', 'Epsilon zeta'])
        self.assertIn('Alpha beta Gamma delta epsilon zeta', seen)
        self.assertIn('Alpha beta gamma delta Epsilon zeta', seen)

    def test_final_backstop_still_fails_closed(self):
        actual = r.punctuation_only
        calls = 0
        def fail_final(old, new):
            nonlocal calls
            if old == 'alpha beta gamma delta' and new == 'Alpha beta Gamma delta':
                calls += 1
                if calls == 2:
                    return False
            return actual(old, new)
        with patch.object(r, 'punctuation_only', side_effect=fail_final):
            result, _ = run(source('alpha beta', 'gamma delta'), lambda d:d['text'].capitalize())
        self.assertEqual([s['text'] for s in result['segments']], ['alpha beta', 'gamma delta'])
        self.assertTrue(all(j['fallback_stage'] == 'final_passage_backstop'
                            for j in result['reconciliation']['punctuation_journal']))

    def test_ambiguous_cross_unit_context_and_untouched_paths(self):
        for texts in [('alpha ( beta', 'gamma delta'), ('alpha < beta', 'gamma delta'),
                      ('alpha " beta', 'gamma delta')]:
            result, _ = run(source(*texts), lambda d:d['text'].capitalize())
            self.assertEqual([s['text'] for s in result['segments']], list(texts))
            self.assertTrue(any(j['validation'] == 'review_required' for j in result['reconciliation']['punctuation_journal']))
        for edit in [lambda d:d['text'], lambda d:'invented']:
            result, _ = run(source('alpha beta', 'gamma delta'), edit)
            self.assertEqual([s['text'] for s in result['segments']], ['alpha beta', 'gamma delta'])
            self.assertTrue(all(s['provenance']['punctuation']['transaction']['status'] == 'unchanged'
                                for s in result['segments']))
        result, _ = run(source('alpha beta', 'gamma delta'),
                        lambda d:d['text'] + ('.' if not d['continues_in_next_chunk'] else ''))
        self.assertEqual(result['segments'][-1]['text'], 'gamma delta.')
        empty = {'duration': 1, 'segments': []}
        result = r.reconcile(empty, empty, lambda d:self.fail('no model request expected'))
        self.assertEqual(result['segments'], [])

    def test_source_choice_provenance_and_all_exports(self):
        a = source('not 12 alpha.', 'gamma delta', 'epsilon zeta')
        b = source('13 beta.', 'gamma delta', 'epsilon zeta')
        before = copy.deepcopy((a, b))
        def model(messages):
            d = json.loads(messages[1]['content'])
            if 'segments' in d:
                return json.dumps({'segments':[{'id':d['segments'][0]['id'],'choice':'B'}]})
            return json.dumps({'id':d['id'],'text':d['text'].replace('beta.', 'beta,').capitalize()})
        result = r.reconcile(a, b, model)
        self.assertEqual((a,b), before)
        self.assertEqual([s['text'] for s in result['segments']], ['13 beta.', 'Gamma delta', 'Epsilon zeta'])
        self.assertEqual(result['source_transcripts'], {'A':a['segments'], 'B':b['segments']})
        ids = {j['id'] for j in result['reconciliation']['journal']}
        self.assertTrue(all(set(j['source_refs']) <= ids for j in result['reconciliation']['punctuation_journal']))
        assigned = downstream(result)
        self.assertEqual(assigned['reconciliation'], result['reconciliation'])
        self.assertEqual(assigned['source_transcripts'], result['source_transcripts'])
        self.assertEqual(assigned['boundaries'], result['boundaries'])
        self.assertFalse(assigned['quality_verified'])
        self.assertEqual(canonical(json.loads(json.dumps(assigned))), assigned)
        for kind, text in render(assigned, {'SPEAKER_00':'Alex'}).items():
            self.assertIn(REVIEW, text, kind)
            self.assertIn('Gamma', text, kind)
            if kind in ('txt', 'tsv', 'md'):
                self.assertIn('Alex', text, kind)
        flags = {f for s in assigned['segments'] for u in s.get('uncertainties', []) for f in u.get('critical', [])}
        self.assertTrue({'numbers','negation'} <= flags)
        for s in result['segments']:
            self.assertEqual(s['provenance']['punctuation']['result_text'], s['text'])
        for s in result['segments']:
            for label, spans in s['provenance']['source_spans'].items():
                for span in spans:
                    native = next(n for n in before[0 if label == 'A' else 1]['segments'] if n['id'] == span['segment_id'])
                    self.assertEqual(native['text'][span['char_start']:span['char_end']], span['text'])


if __name__ == '__main__':
    unittest.main()
