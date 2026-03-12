import argparse
import os
import pickle
from collections import Counter


def parse_class_and_id(path):
    norm = path.replace('\\', '/')
    parts = [p for p in norm.split('/') if p]
    if len(parts) < 3:
        return 'unknown', 'unknown'
    return parts[-3], parts[-2]


def main():
    parser = argparse.ArgumentParser(description='Inspect embeddings database pickle file.')
    parser.add_argument(
        '--db',
        default='bot/dataset_crop/database_embeddings.pkl',
        help='Path to pickle database file',
    )
    parser.add_argument(
        '--head',
        type=int,
        default=10,
        help='How many sample paths to print',
    )
    args = parser.parse_args()

    db_path = args.db
    if not os.path.exists(db_path):
        print(f'File not found: {db_path}')
        return

    with open(db_path, 'rb') as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        print(f'Unexpected format: {type(data)}')
        return

    print('DB path:', db_path)
    print('Top-level keys:', sorted(list(data.keys())))

    embeddings = data.get('embeddings')
    paths = data.get('paths')

    if embeddings is None or paths is None:
        print("Missing 'embeddings' or 'paths' keys")
        return

    count_embeddings = len(embeddings)
    count_paths = len(paths)
    emb_dim = len(embeddings[0]) if count_embeddings > 0 else 0

    print('Total paths:', count_paths)
    print('Total embeddings:', count_embeddings)
    print('Embedding dim:', emb_dim)

    if count_embeddings != count_paths:
        print('WARNING: embeddings count != paths count')

    class_counter = Counter()
    id_counter = Counter()

    for p in paths:
        cls, ind = parse_class_and_id(p)
        class_counter[cls] += 1
        id_counter[(cls, ind)] += 1

    print('\nClass distribution:')
    for cls, n in class_counter.most_common():
        print(f'  {cls}: {n}')

    print('\nUnique individuals:', len(id_counter))

    print(f'\nFirst {min(args.head, count_paths)} paths:')
    for i, p in enumerate(paths[: args.head], 1):
        cls, ind = parse_class_and_id(p)
        print(f'  {i}. [{cls}/{ind}] {p}')

    metadata = data.get('metadata')
    if metadata is not None:
        print('\nMetadata:')
        if isinstance(metadata, dict):
            for k in sorted(metadata.keys()):
                print(f'  {k}: {metadata[k]}')
        else:
            print('  metadata is present but not dict')


if __name__ == '__main__':
    main()
