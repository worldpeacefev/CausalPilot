from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from AsgardBench.causal.evaluator import canonical_action, parse_action
from AsgardBench.causal.metrics import grouped_metrics


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def score(dataset: Path, predictions: Path, allow_partial=False, iterations=1000):
    rows = json.loads(dataset.read_text())['triplets']
    truth = {(t['triplet_id'], v): (t, t['variants'][v]) for t in rows for v in ('anchor', 'causal', 'nuisance')}
    groups = defaultdict(set)
    scored = []
    metadata_cache = {}
    for line in predictions.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row['triplet_id'], row['variant'])
        if key not in truth:
            raise ValueError(f'Unknown observation: {key}')
        group = (row.get('model', 'unspecified'), row.get('method', 'unspecified'))
        if key in groups[group]:
            raise ValueError(f'Duplicate observation for {group}: {key}')
        groups[group].add(key)
        t, variant = truth[key]
        metadata_path = dataset.parent / variant['metadata']
        if metadata_path not in metadata_cache:
            metadata_cache[metadata_path] = sorted({o['objectType'] for o in json.loads(metadata_path.read_text())['objects']})
        types = metadata_cache[metadata_path]
        output = row.get('response', row.get('prediction'))
        if output is None:
            action, obj = None, None
        else:
            action, obj = parse_action(json.dumps(output) if isinstance(output, dict) else str(output), types)
        legal = {a.split()[0].lower() for a in t['action_interface']}
        if action not in legal or (action != 'done' and str(obj).lower() not in {o.lower() for o in types}):
            action, obj = None, None
        prediction = canonical_action(action, obj)
        acceptable = {canonical_action(a['action'], a.get('object')) for a in variant['acceptable_actions']}
        scored.append({'model': group[0], 'method': group[1], 'triplet_id': key[0], 'variant': key[1],
                       'causal_state_unit': t['causal_state_unit'], 'prediction': prediction,
                       'correct': prediction is not None and prediction in acceptable})
    if not groups:
        raise ValueError('No predictions')
    if not allow_partial:
        for group, observations in groups.items():
            if observations != set(truth):
                raise ValueError(f'{group}: {len(observations)} observations; expected {len(truth)}')
    return json_safe(grouped_metrics(scored, bootstrap_iterations=iterations))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score action predictions against the released benchmark labels')
    parser.add_argument('--dataset', type=Path, default=Path(__file__).resolve().parent / 'data/benchmark.json')
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--bootstrap-iterations', type=int, default=1000)
    args = parser.parse_args()
    result = score(args.dataset, args.predictions, args.allow_partial, args.bootstrap_iterations)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n')
    else:
        print(text)
