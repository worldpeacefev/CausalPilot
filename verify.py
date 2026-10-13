from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

ROLES = ('anchor', 'causal', 'nuisance')


def verify(root: Path, checksums: bool = True) -> dict:
    root = root.resolve()
    data = root / 'data'
    rows = json.loads((data / 'benchmark.json').read_text())['triplets']
    states = json.loads((data / 'causal_states.json').read_text())['triplets']
    state_ids = {t['causal_state_unit'] for t in states}
    assert len(states) == len(state_ids) == 71
    assert len(rows) == len({t['triplet_id'] for t in rows}) == 996
    assert {t['causal_state_unit'] for t in rows} == state_ids
    assert Counter(t['replica_profile'] for t in rows)['simulator_lighting'] == 71
    images = set()
    for t in [*states, *rows]:
        assert set(t['variants']) == set(ROLES), t['triplet_id']
        assert (root / t['source_plan']).is_file(), t['source_plan']
        actions = {}
        values = {}
        for role in ROLES:
            v = t['variants'][role]
            path = (data / v['image']).resolve()
            assert path.is_relative_to(root) and path.is_file(), path
            if path not in images:
                with Image.open(path) as image:
                    image.verify()
                images.add(path)
            metadata_path = (data / v['metadata']).resolve()
            assert metadata_path.is_relative_to(root) and metadata_path.is_file(), metadata_path
            metadata = json.loads(metadata_path.read_text())
            types = {o['objectType'].lower() for o in metadata['objects']}
            legal = {a.split()[0].lower() for a in t['action_interface']}
            actions[role] = {(a['action'].lower(), (a.get('object') or 'none').lower()) for a in v['acceptable_actions']}
            assert actions[role]
            assert all(a in legal and (a == 'done' or o in types) for a, o in actions[role])
            values[role] = v['symbolic_state']['objects'][t['target']['predicate_object_name']][t['predicate']]
            assert isinstance(values[role], bool)
            assert values[role] == v['validation']['predicate_value']
        assert actions['anchor'].isdisjoint(actions['causal']), t['triplet_id']
        assert actions['anchor'] == actions['nuisance'], t['triplet_id']
        assert values['anchor'] is not values['causal'], t['triplet_id']
        assert values['anchor'] is values['nuisance'], t['triplet_id']
    observation_paths = {t['variants'][v]['image'] for t in rows for v in ROLES}
    assert len(observation_paths) == 2988
    if checksums:
        manifest = json.loads((root / 'checksums.json').read_text())
        for record in manifest['files']:
            file = root / record['path']
            assert file.is_file() and file.stat().st_size == record['bytes'], record['path']
            assert hashlib.sha256(file.read_bytes()).hexdigest() == record['sha256'], record['path']
    return {'causal_state_units': 71, 'triplets': 996, 'observations': 2988,
            'states_by_predicate': dict(sorted(Counter(t['predicate'] for t in states).items())),
            'passed': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--skip-checksums', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.root, not args.skip_checksums), ensure_ascii=False))
