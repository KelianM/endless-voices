"""Prepare reviewed replacement annotations under mission-level split validation."""

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path

from prepare_conversations import DEFAULT_SOURCES, tree, walk

from endless_voices.contracts import validate_manifest
from endless_voices.prepare_context import file_hash
from endless_voices.splits import check_mission_splits


def prepare(output):
    """Build preparation records without changing existing evaluation examples."""
    base = Path('data/pilot-v1')
    manifest = json.loads((base / 'samples/manifest.json').read_text())
    records = {}
    for split, files in manifest['files'].items():
        records[split] = []
        for entry in files:
            path = base / 'samples' / entry['path']
            if file_hash(path) != entry['sha256']:
                raise ValueError('Source split hash differs')
            records[split].extend(json.loads(line) for line in path.read_text().splitlines())
    original_train = {r['metadata']['id']: r for r in records['train']}
    ledger = {r['id']: r for r in json.loads((base / 'evidence/provenance.json').read_text())}
    excluded = {r['sample_id'] for r in json.loads(
        Path('data/scene-training-v1/exclusions.json').read_text())}
    records['train'] = [r for r in records['train'] if r['metadata']['id'] not in excluded]
    annotations = json.loads(Path('data/training-replacements-v1/annotations.json').read_text())
    additions = []
    for annotation in annotations:
        template = annotation['template_sample_id']
        row, origin = deepcopy(original_train[template]), deepcopy(ledger[template])
        path = DEFAULT_SOURCES / origin['source_path']
        if file_hash(path) != origin['source_sha256']:
            raise ValueError('Source file hash differs')
        start = int(row['metadata']['conversation_id'].rsplit('-l', 1)[1])
        conversation = next(n for n, _ in walk(tree(path.read_text()))
                            if n['line'] == start and n['tokens'][0] == 'conversation')
        node = next(n for n in conversation['children'] if n['line'] == annotation['target_line'])
        if not node['raw'].lstrip().startswith('`') or node['children']:
            raise ValueError('Replacement requires an unconditional source paragraph')
        sid = 'replacement-' + row['metadata']['conversation_id'] + '-l' + str(node['line'])
        row['metadata'].update(id=sid, review_status='reviewed')
        row['messages'] = [row['messages'][0], {'role': 'user', 'content': 'Continue the scene.'},
                           {'role': 'assistant', 'content': node['tokens'][0]}]
        origin.update(id=sid, annotation_id=sid, review_notes=annotation['review'],
                      route='Source route ending before the replacement target')
        origin['messages'] = [origin['messages'][0],
                              {'role': 'user', 'origin': 'agent', 'spans': []},
                              {'role': 'assistant', 'origin': 'upstream', 'spans': [
                                  {'line': node['line'], 'start': 0, 'end': len(node['raw']),
                                   'normalization': 'Complete source paragraph'}]}]
        records['train'].append(row)
        ledger[sid] = origin
        additions.append(sid)
    check_mission_splits([row for rows in records.values() for row in rows])
    output.mkdir(parents=True, exist_ok=False)
    samples = output
    new_manifest = {'schema_version': 1, 'dataset_version': 'mission-training-preparation-v1',
                    'split_unit': 'mission', 'files': {}}
    for split, rows in records.items():
        path = samples / (split + '.jsonl')
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        new_manifest['files'][split] = [{'path': path.name, 'sha256': file_hash(path)}]
    (samples / 'split-manifest.json').write_text(json.dumps(new_manifest, indent=2) + '\n')
    ids = {r['metadata']['id'] for rows in records.values() for r in rows}
    (output / 'provenance.json').write_text(json.dumps(
        [r for sid, r in ledger.items() if sid in ids], indent=2) + '\n')
    (output / 'changes.json').write_text(json.dumps(
        {'excluded': sorted(excluded), 'added': additions}, indent=2) + '\n')
    shutil.copytree(base / 'licensing', output / 'licensing')
    coverage = validate_manifest(samples / 'split-manifest.json')
    print(json.dumps({split: stats['records'] for split, stats in coverage.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args().output)
