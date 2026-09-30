"""Export selected contexts and authored targets for preliminary continuation training."""

import argparse
import json
import re
import shutil
from pathlib import Path

from endless_voices.assessment import load_dataset
from endless_voices.continuations import load_targets, target_text
from endless_voices.prepare_context import file_hash, load_selections


def prepare(context, targets, tokenizer, output, manifest):
    """Save training conversations with verified inputs, targets and token measurements."""
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
        rows.append({'messages': messages, 'metadata': {
            'id': selection.sample_id,
            'conversation_id': records[selection.sample_id]['metadata']['conversation_id'],
            'split': 'train', 'context_sha256': selection.provenance['messages_sha256']}})
        measurements.append({'sample_id': selection.sample_id, 'input_tokens': len(prefix),
                             'target_tokens': len(ids) - len(prefix), 'total_tokens': len(ids),
                             'remaining_markers': sorted(set(re.findall(
                                 r'<[^>]+>', json.dumps(messages))))})
    output.mkdir(parents=True, exist_ok=False)
    (output / 'train.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n'
                                              for row in rows))
    (output / 'measurements.json').write_text(json.dumps(measurements, indent=2) + '\n')
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
                        default=Path("data/pilot-v1/samples/manifest.json"))
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prepare(args.context, args.targets, tokenizer, args.output, args.manifest)


if __name__ == '__main__':
    main()
