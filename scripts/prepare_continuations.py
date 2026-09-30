"""Build validation continuation targets from verified cached game passages."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from prepare_conversations import DEFAULT_SOURCES, flow_graph, tree, walk

from endless_voices.assessment import load_dataset
from endless_voices.continuations import continuation


def prepare(manifest, provenance, source, output, split="validation", exclude_ambiguous=False):
    """Save complete validation targets and source coordinates without model calls."""
    records = load_dataset(manifest, split)
    ledger = {r['id']: r for r in json.loads(provenance.read_text()) if r['id'] in records}
    boundaries = {}
    for sid, record in records.items():
        key = record['metadata']['conversation_id']
        boundaries.setdefault(key, set()).update(
            s['line'] for m in ledger[sid]['messages'] if m['role'] == 'user'
            for s in m.get('spans', []))
    rows, parsed, excluded = [], {}, []
    for sid, record in records.items():
        origin = ledger[sid]
        relative = Path(origin['source_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid source path')
        path = source / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != origin['source_sha256']:
            raise ValueError('Cached source hash differs')
        if relative not in parsed:
            parsed[relative] = tree(path.read_text())
        start = int(record['metadata']['conversation_id'].rsplit('-l', 1)[1])
        conversation = next(n for n, _ in walk(parsed[relative])
                            if n['tokens'][:1] == ['conversation'] and n['line'] == start)
        anchors = [s['line'] for s in origin['messages'][-1]['spans']]
        try:
            result = continuation(conversation, flow_graph(conversation), anchors,
                                  boundaries[record['metadata']['conversation_id']])
        except ValueError as error:
            if not exclude_ambiguous:
                raise ValueError(f'{sid}: {error}') from error
            excluded.append({'sample_id': sid, 'reason': str(error)})
            continue
        rows.append({'sample_id': sid, 'source_path': str(relative),
                     'source_revision': origin['source_revision'],
                     'source_sha256': origin['source_sha256'], **result})
    output.mkdir(parents=True, exist_ok=False)
    raw = (json.dumps(rows, ensure_ascii=False, indent=2) + '\n').encode()
    (output / 'targets.json').write_bytes(raw)
    (output / 'manifest.json').write_text(json.dumps({
        'format': 'source-continuation-v1', 'split': split,
        'sample_ids': [r['sample_id'] for r in rows], 'excluded': excluded,
        'targets_sha256': hashlib.sha256(raw).hexdigest(),
        'source_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'annotation_provenance_sha256': hashlib.sha256(provenance.read_bytes()).hexdigest(),
        'boundary': 'First target paragraph through selected route to next choice or route end',
        'normalization': 'Remove game source delimiters only; preserve paragraph text. '
                         'Game variables are substituted by the consumer.',
    }, indent=2) + '\n')
    shutil.copytree(manifest.parent.parent / 'licensing', output / 'licensing')
    (output / 'licensing/ATTRIBUTION.md').write_text(
        '# Attribution\n\nThese source continuations derive from Endless Sky at revision '
        + rows[0]['source_revision']
        + '. Preserve the accompanying license, copyright, credits and source notices. '
        'Unlike the historical speech-only dataset, these targets preserve complete authored '
        'paragraphs, including narration, actions and quotation marks. Source delimiters are '
        'removed; runtime variables remain for consumer substitution. No prose is generated.\n')
    print(f'Saved {len(rows)} {split} continuations; excluded {len(excluded)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=Path('data/pilot-v1/samples/manifest.json'))
    parser.add_argument('--provenance', type=Path,
                        default=Path('data/pilot-v1/evidence/provenance.json'))
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCES)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'validation'], default='validation')
    parser.add_argument('--exclude-ambiguous', action='store_true')
    args = parser.parse_args()
    prepare(args.manifest, args.provenance, args.source, args.output,
            args.split, args.exclude_ambiguous)


if __name__ == '__main__':
    main()
