"""Check our compression and layout logic against published numbers. Runs no downloaded code."""
import collections
import json
import sys

sys.path.insert(0, '/work/u4320956/engram-research/scripts')
from expA_common import *  # noqa

out = {}
for name, path in [('v3', V3), ('v41', V41)]:
    tk = load_tok(path)
    lut, n = compressed_map(tk)
    size = tk.get_vocab_size(with_added_tokens=True)
    c = collections.Counter(lut.tolist())
    inv = collections.defaultdict(list)
    for i, x in enumerate(lut.tolist()):
        inv[x].append(i)
    cjk = [i for i in range(size) if any('一' <= ch <= '鿿' for ch in tk.decode([i]))]
    out[name] = {
        'vocab': size, 'compressed': n, 'reduction_pct': round(100 * (1 - n / size), 2),
        'top_merges': [{'count': k, 'examples': [tk.decode([i]) for i in inv[v][:6]]} for v, k in c.most_common(6)],
        'cjk_tokens': len(cjk), 'cjk_in_merged_group': sum(c[lut[i]] > 1 for i in cjk),
    }
    if name == 'v41':
        import opencc
        conv = opencc.OpenCC('t2s')
        pairs = same = 0
        ex = []
        for i in cjk:
            s = tk.decode([i])
            if '�' in s:
                continue
            z = conv.convert(s)
            if z == s:
                continue
            ids = tk.encode(z, add_special_tokens=False).ids
            if len(ids) == 1:
                pairs += 1
                eq = bool(lut[ids[0]] == lut[i])
                same += eq
                if len(ex) < 10:
                    ex.append([s, z, eq])
        out['trad_simp_single_token_pairs'] = {'pairs': pairs, 'same_compressed_id': same, 'examples': ex}
        out['v41_matches_config_99092'] = n == 99092

# Layout: the rows implied by 16M-start primes should match config engram_num_embeddings.
seen = set()
rows = []
for _layer in (1, 14):
    total = 0
    for _order in range(3):
        total += sum(next_primes(16000000, 8, seen))
    rows.append(total)
out['v41_layout_rows_ours'] = rows
out['v41_layout_rows_config'] = [384006168, 384016682]
out['v41_layout_matches'] = rows == out['v41_layout_rows_config']
out['v41_params_B'] = round(sum(rows) * 256 / 1e9, 2)
json.dump(out, open(R + 'results/expA/verify_config.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
