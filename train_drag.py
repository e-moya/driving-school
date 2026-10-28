"""Train steering AND throttle from obstacle-practice CSVs; keep train.py separate."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import re
from pathlib import Path
import sklearn
from sklearn.tree import DecisionTreeClassifier

FEATURES = ['left', 'front_left', 'front', 'front_right', 'right', 'speed']
LABELS = ['steering', 'throttle']
ACTIONS = [['left', 'straight', 'right'], ['accelerate', 'brake']]
MAX_DEPTH = 16
MIN_SAMPLES_LEAF = 8


def load_examples(paths):
    rows = []
    skipped_idle = 0
    for path in paths:
        with Path(path).open(newline='') as file:
            reader = csv.DictReader(file)
            required = FEATURES + LABELS + ['schema_version', 'track_id', 'split']
            if not reader.fieldnames or not set(required) <= set(reader.fieldnames):
                raise ValueError(f'{path}: use CSV downloaded from Record drive on drag.html.')
            for number, row in enumerate(reader, 2):
                try:
                    if None in row or any(v is None for v in row.values()):
                        raise ValueError('incomplete row')
                    if row['schema_version'] != 'drag-2':
                        raise ValueError('old recording physics: record fresh drag-2 examples with automatic braking; keep old CSVs outside the training folder')
                    if not (row['track_id'] in ('drag-practice-1', 'drag-practice-2', 'drag-practice-tight-1', 'drag-practice-mixed-1') or re.fullmatch(r'drag-practice-random-v[12]-[1-9][0-9]{0,5}', row['track_id'])) or row['split'] != 'train':
                        raise ValueError('only obstacle-practice training recordings are accepted')
                    for key in FEATURES:
                        row[key] = float(row[key])
                        if not math.isfinite(row[key]) or not 0 <= row[key] <= 1:
                            raise ValueError('features must be finite numbers in [0, 1]')
                    if any(row[key] not in allowed for key, allowed in zip(LABELS, ACTIONS)):
                        raise ValueError('invalid steering or throttle')
                except (ValueError, TypeError) as error:
                    raise ValueError(f'{path}, row {number}: {error}') from error
                # Waiting for the first key is not a demonstration of driving.
                # Keep moving braking examples; exclude stationary waits.
                if row['speed'] == 0 and row['throttle'] == 'brake':
                    skipped_idle += 1
                    continue
                rows.append(row)
    if skipped_idle:
        print(f'Skipped {skipped_idle} stationary brake rows (waiting for input). Original CSVs unchanged.')
    if not rows:
        raise ValueError('No examples. Save obstacle-practice CSVs in runtime/drag-recordings/.')
    for key, allowed in zip(LABELS, ACTIONS):
        counts = Counter(row[key] for row in rows)
        print(f'{key}: {dict(counts)}')
        missing = set(allowed) - counts.keys()
        if missing:
            print(f'Warning: no examples for {sorted(missing)}.')
        if max(counts.values()) / len(rows) > .85:
            print(f'Warning: more than 85% of examples have one {key} label.')
    return rows


def train(rows):
    X = [[row[column] for column in FEATURES] for row in rows]
    y = [[row[column] for column in LABELS] for row in rows]
    model = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF, random_state=42)
    model.fit(X, y)
    return model


def export_model(model, rows, paths):
    tree = model.tree_
    return dict(schema='drag-model-2', features=FEATURES, outputs=LABELS,
        classes=[classes.tolist() for classes in model.classes_],
        nodes=[dict(left=int(tree.children_left[i]), right=int(tree.children_right[i]),
            feature=int(tree.feature[i]), threshold=float(tree.threshold[i]),
            predictions=[int(tree.value[i][j][:len(model.classes_[j])].argmax()) for j in range(2)])
            for i in range(tree.node_count)],
        metadata=dict(samples=len(rows), parameters=model.get_params(), sklearn=sklearn.__version__,
            recording_schema='drag-2', physics='drag-2',
            sources=[dict(file=Path(p).name, sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in paths]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recordings', nargs='*', type=Path)
    parser.add_argument('--output', type=Path, default=Path('runtime/drag-model.json'))
    args = parser.parse_args()
    paths = args.recordings or sorted(Path('runtime/drag-recordings').glob('*.csv'))
    rows = load_examples(paths)
    artifact = export_model(train(rows), rows, paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False))
    print(f'Saved {args.output}. Load this JSON into a driver slot on web/race.html.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from None
