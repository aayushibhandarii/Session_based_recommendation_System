import argparse
import pickle
from collections import defaultdict

import torch


def build_global_graph(train_sessions, num_items, window_size=3, max_neighbors=12):
    co_occurrence = defaultdict(lambda: defaultdict(int))

    for session in train_sessions:
        seq = [it for it in session if it != 0]  # drop the 0 padding id
        n = len(seq)
        for i in range(n):
            for j in range(i + 1, min(i + 1 + window_size, n)):
                a, b = seq[i], seq[j]
                if a == b:
                    continue
                co_occurrence[a][b] += 1
                co_occurrence[b][a] += 1

    neighbor_idx = torch.zeros((num_items, max_neighbors), dtype=torch.long)
    neighbor_weight = torch.zeros((num_items, max_neighbors), dtype=torch.float)

    for item, neighbors in co_occurrence.items():
        if item >= num_items:
            continue
        top = sorted(neighbors.items(), key=lambda kv: kv[1], reverse=True)[:max_neighbors]
        if not top:
            continue
        weights = torch.tensor([w for _, w in top], dtype=torch.float)
        weights = weights / weights.sum()  # normalize per item so weights are comparable across items
        for k, (nbr, _) in enumerate(top):
            if nbr < num_items:
                neighbor_idx[item, k] = nbr
                neighbor_weight[item, k] = weights[k]

    return neighbor_idx, neighbor_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_pkl', default='dataset/amazon_beauty/all_train_seq.txt',
                         help='pickle file with the raw per-session sequences. Use '
                              'all_train_seq.txt (full sessions), NOT train.txt (which '
                              'holds overlapping next-item-prediction fragments of the '
                              'same sessions and would badly overcount co-occurrences).')
    parser.add_argument('--n_node', type=int, default=None,
                         help='vocabulary size, same n_node you pass to SessionGraph. '
                              'If omitted, loaded from dataset/amazon_beauty/vocab_size.pkl.')
    parser.add_argument('--window_size', type=int, default=3)
    parser.add_argument('--max_neighbors', type=int, default=12)
    parser.add_argument('--out_prefix', default='dataset/amazon_beauty/global_graph')
    args = parser.parse_args()

    with open(args.train_pkl, 'rb') as f:
        loaded = pickle.load(f)
    # all_train_seq.txt is a plain list of sequences; train.txt (if used instead)
    # is a (sequences, labels) tuple — support either.
    sessions = loaded[0] if isinstance(loaded, tuple) else loaded

    n_node = args.n_node
    if n_node is None:
        with open('dataset/amazon_beauty/vocab_size.pkl', 'rb') as f:
            n_node = pickle.load(f)

    neighbor_idx, neighbor_weight = build_global_graph(
        sessions, n_node, args.window_size, args.max_neighbors
    )

    torch.save(neighbor_idx, f'{args.out_prefix}_idx.pt')
    torch.save(neighbor_weight, f'{args.out_prefix}_weight.pt')
    print(f"Saved global graph: {neighbor_idx.shape} neighbors/item "
          f"-> {args.out_prefix}_idx.pt / _weight.pt")


if __name__ == '__main__':
    main()