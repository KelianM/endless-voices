"""Export selected contexts and authored targets for preliminary continuation training."""

import argparse
import json
import re
import shutil
from copy import deepcopy
from pathlib import Path

from endless_voices.assessment import load_dataset
from endless_voices.continuations import load_targets, target_text
from endless_voices.contracts import validate_manifest
from endless_voices.prepare_context import file_hash, load_selections


def prepare(context, targets, tokenizer, output, manifest):
    """Save training conversations with verified inputs, targets and token measurements."""
    validate_manifest(manifest)
    if json.loads(manifest.read_text()).get("split_unit") != "mission":
        raise ValueError("Training export requires a mission-split manifest")
    selections = load_selections(context)
    records = load_dataset(manifest, "train")
    references = load_targets(targets)
    rows, measurements = [], []
    for selection in selections:
        if selection.provenance.get('variable_policy') != 'mission-scoped':
            raise ValueError('Training release requires mission-scoped variables')
        target = target_text(references[selection.sample_id], selection.messages,
                             selection.provenance['game_variables'])
        messages = selection.training_messages(target)
        encoded = tokenizer.apply_chat_template(messages, tokenize=True,
                                                add_generation_prompt=False)
        prompt = tokenizer.apply_chat_template(selection.messages, tokenize=True,
                                               add_generation_prompt=False)
        ids = encoded['input_ids'] if hasattr(encoded, 'keys') else encoded
        prefix = prompt['input_ids'] if hasattr(prompt, 'keys') else prompt
        if ids[:len(prefix)] != prefix or len(prefix) >= len(ids):
            raise ValueError('Chat template does not preserve the supplied conversation prefix')
        row = deepcopy(records[selection.sample_id])
        row['messages'] = messages
        rows.append(row)
        measurements.append({'sample_id': selection.sample_id,
                             'context_sha256': selection.provenance['messages_sha256'],
                             'input_tokens': len(prefix),
                             'target_tokens': len(ids) - len(prefix), 'total_tokens': len(ids),
                             'remaining_markers': sorted(set(re.findall(
                                 r'<[^>]+>', json.dumps(messages))))})
    output.mkdir(parents=True, exist_ok=False)
    (output / 'train.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n'
                                              for row in rows))
    (output / 'measurements.json').write_text(json.dumps(measurements, indent=2) + '\n')
    source_manifest = json.loads(manifest.read_text())
    split_manifest = {'schema_version': 1, 'dataset_version': output.name,
                      'split_unit': 'mission', 'files': {
                          'train': [{'path': 'train.jsonl',
                                     'sha256': file_hash(output / 'train.jsonl')}]}}
    for split in ['validation', 'test']:
        split_manifest['files'][split] = []
        for index, entry in enumerate(source_manifest['files'][split]):
            name = f'{split}-{index}.jsonl'
            shutil.copyfile(manifest.parent / entry['path'], output / name)
            split_manifest['files'][split].append(
                {'path': name, 'sha256': file_hash(output / name)})
    (output / 'split-manifest.json').write_text(json.dumps(split_manifest, indent=2) + '\n')
    validate_manifest(output / 'split-manifest.json')
    (output / 'manifest.json').write_text(json.dumps({
        'purpose': 'Preliminary game-writing fine-tuning; validation is development data',
        'samples': len(rows), 'loss': 'final assistant continuation only',
        'max_total_tokens': max(row['total_tokens'] for row in measurements),
        'artifacts': {name: file_hash(output / name)
                      for name in ['train.jsonl', 'measurements.json']},
        'context_provenance_sha256': file_hash(context / 'provenance.json'),
        'target_manifest_sha256': file_hash(targets / 'manifest.json'),
        'exporter_sha256': file_hash(__file__),
        'source_manifest_sha256': file_hash(manifest),
    }, indent=2) + '\n')
    shutil.copytree(targets / 'licensing', output / 'licensing')
    print(json.dumps({'samples': len(rows),
                      'max_total_tokens': max(r['total_tokens'] for r in measurements),
                      'max_target_tokens': max(r['target_tokens'] for r in measurements)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['context', 'targets', 'tokenizer', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/scene-training-v2/split-manifest.json"))
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prepare(args.context, args.targets, tokenizer, args.output, args.manifest)


if __name__ == '__main__':
    main()
