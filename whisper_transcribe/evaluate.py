"""Score explicitly verified human references; this tool never promotes an engine."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from .state import atomic_json

NORMALIZATION = {'version': 'lexemes-casefold-v1', 'wer': 'Unicode casefold; Unicode word runs; punctuation removed; whitespace collapsed',
                 'cer': 'Same normalized words joined by one ASCII space; spaces count as characters',
                 'numbers': 'No spoken/digit equivalence; 2 and two differ',
                 'txt': 'UTF-8 plain transcript; labels, timestamps and notes are not automatically stripped'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    return re.findall(r'\w+', text.casefold(), flags=re.UNICODE)


def edit_counts(reference, hypothesis):
    """Levenshtein counts in O(min(n,m)) storage; deterministic tie breaking.

    Tuple order is distance, substitutions, deletions, insertions. Swapping the
    axes also swaps deletion/insertion semantics back before returning.
    """
    swapped = len(hypothesis) > len(reference)
    if swapped:
        reference, hypothesis = hypothesis, reference
    row = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, source in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, target in enumerate(hypothesis, 1):
            if source == target:
                current.append(row[j-1])
            else:
                sub, delete, insert = row[j-1], row[j], current[j-1]
                options = [(sub[0]+1, sub[1]+1, sub[2], sub[3]),
                           (delete[0]+1, delete[1], delete[2]+1, delete[3]),
                           (insert[0]+1, insert[1], insert[2], insert[3]+1)]
                current.append(min(options, key=lambda item: item[0]))
        row = current
    edits, substitutions, deletions, insertions = row[-1]
    if swapped:
        deletions, insertions = insertions, deletions
    return {'edits': edits, 'substitutions': substitutions, 'deletions': deletions, 'insertions': insertions}


def metrics(reference, candidate):
    words, predicted = normalize(reference), normalize(candidate)
    chars, predicted_chars = ' '.join(words), ' '.join(predicted)
    result = {}
    for name, source, target in [('wer', words, predicted), ('cer', chars, predicted_chars)]:
        counts = edit_counts(source, target)
        result[name] = {**counts, 'reference_length': len(source), 'candidate_length': len(target),
                        'rate': counts['edits'] / len(source) if source else None}
    return result


def candidate_text(path):
    path = Path(path)
    raw = path.read_text(encoding='utf-8-sig')
    if path.suffix.lower() != '.json':
        return raw
    value = json.loads(raw)
    if not isinstance(value, dict) or not isinstance(value.get('segments'), list):
        raise ValueError('Candidate JSON requires segments')
    if value.get('provisional'):
        raise ValueError('Cannot score an incomplete provisional candidate')
    if any(not isinstance(segment, dict) or not isinstance(segment.get('text'), str) for segment in value['segments']):
        raise ValueError('Candidate segments require string text')
    return ' '.join(segment['text'] for segment in value['segments'])


def read_manifest(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not isinstance(data.get('items'), list):
        raise ValueError('Manifest requires items')
    ids = set()
    for item in data['items']:
        if not isinstance(item, dict) or not isinstance(item.get('id'), str) or item['id'] in ids or item.get('split') not in ('tuning', 'holdout') or not isinstance(item.get('reference'), str):
            raise ValueError('Invalid or duplicate evaluation item')
        ids.add(item['id'])
    return data


def mark_verified(path, item_id, acknowledged=False):
    if not acknowledged:
        raise ValueError('Explicit acknowledgement of human audio/reference review is required')
    path = Path(path)
    data = read_manifest(path)
    selected = [item for item in data['items'] if item['id'] == item_id]
    if not selected:
        raise ValueError('Unknown reference id')
    item = selected[0]
    reference = path.parent / item['reference']
    reference.read_text(encoding='utf-8-sig')
    item.update(reference_verified=True, reference_sha256=digest(reference),
                verification={'method': 'explicit_human_audio_review_acknowledgement',
                              'verified_at': datetime.now(timezone.utc).isoformat()})
    data['normalization'] = NORMALIZATION
    data['status'] = 'references_verified' if all(i.get('reference_verified') is True for i in data['items']) else 'awaiting_human_reference'
    atomic_json(path, data)


def score_manifest(path, candidates, split):
    path = Path(path)
    data = read_manifest(path)
    selected = [item for item in data['items'] if item['split'] == split]
    if not selected or set(candidates) != {item['id'] for item in selected}:
        raise ValueError('Provide exactly one candidate for every item in the selected split')
    # Validate every reference BEFORE computing any score.
    for item in selected:
        if item.get('reference_verified') is not True:
            raise ValueError(f'Reference {item["id"]} has not been human verified')
        if not isinstance(item.get('reference_sha256'), str) or digest(path.parent / item['reference']) != item['reference_sha256']:
            raise ValueError(f'Reference {item["id"]} changed after verification; human re-verification required')
    results = []
    for item in selected:
        reference_path = path.parent / item['reference']
        candidate_path = Path(candidates[item['id']])
        # Read/hash the same bytes to avoid an edited file producing mismatched
        # score provenance; repeat hash below catches concurrent modifications.
        reference = reference_path.read_text(encoding='utf-8-sig')
        candidate_hash = digest(candidate_path)
        candidate = candidate_text(candidate_path)
        measures = metrics(reference, candidate)
        if digest(reference_path) != item['reference_sha256'] or digest(candidate_path) != candidate_hash:
            raise ValueError('Evaluation input changed while scoring')
        results.append({'id': item['id'], 'split': split, 'reference_sha256': item['reference_sha256'],
                        'candidate_sha256': candidate_hash, 'candidate_path': str(candidate_path.resolve()),
                        'reference_text': reference, 'candidate_text': candidate, **measures,
                        'manual_critical_annotations': item.get('manual_critical_annotations', []),
                        'semantic_metric': None})
    aggregate = {}
    for metric in ('wer', 'cer'):
        counts = {key: sum(item[metric][key] for item in results) for key in ('edits', 'substitutions', 'deletions', 'insertions', 'reference_length', 'candidate_length')}
        aggregate[metric] = {**counts, 'rate': counts['edits'] / counts['reference_length'] if counts['reference_length'] else None}
    return {'schema_version': 1, 'split': split, 'normalization': NORMALIZATION, 'items': results,
            'aggregate': aggregate, 'quality_promotion': False, 'semantic_metric': None,
            'note': 'Text edit rates only. Numbers, negations and terms require human audio review. Holdout is separate from tuning.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    verify = commands.add_parser('mark-verified', help='Only after a human has checked the reference against audio')
    verify.add_argument('manifest', type=Path)
    verify.add_argument('--item', required=True)
    verify.add_argument('--acknowledge-human-review', action='store_true', required=True)
    score = commands.add_parser('score')
    score.add_argument('manifest', type=Path)
    score.add_argument('--split', choices=['tuning', 'holdout'], required=True)
    score.add_argument('--candidate', action='append', required=True, metavar='ID=PATH')
    score.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        if args.command == 'mark-verified':
            mark_verified(args.manifest, args.item, args.acknowledge_human_review)
            print('Reference hash saved after explicit human review acknowledgement.')
        else:
            pairs = [value.split('=', 1) for value in args.candidate]
            if any(len(pair) != 2 for pair in pairs) or len({pair[0] for pair in pairs}) != len(pairs):
                raise ValueError('Candidates must be unique ID=PATH pairs')
            result = score_manifest(args.manifest, dict(pairs), args.split)
            atomic_json(args.output, result)
            print(json.dumps({'output': str(args.output), 'split': args.split, 'aggregate': result['aggregate']}, allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError) as error:
        p.exit(1, f'Evaluation refused: {error}\n')


if __name__ == '__main__':
    main()
