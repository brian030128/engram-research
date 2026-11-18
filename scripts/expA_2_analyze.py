"""Experiment A: n-gram statistics, multi-head collisions, hot-row hit rate, zh variant overlap."""
import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, '/work/u4320956/engram-research/scripts')
from expA_common import R, V41, compressed_map, load_tok, lookback, multipliers, next_primes, rolling

IN = R + 'results/expA/corpus/'
OUT = R + 'results/expA/'
K = 8
SIZES = [65536, 262144, 646400, 2262400, 16000000]
HOT_SIZES = [2262400, 16000000]
HOT_FRAC = [0.001, 0.01, 0.1]
BANDS = [('1', 1, 1), ('2-9', 2, 9), ('10-99', 10, 99), ('100+', 100, 10**18)]
SCALES = {'paper_262B': 262e9, 'v41_45T': 45e12}

tk = load_tok(V41)
lut, V = compressed_map(tk)
assert V == 99092
PAD = int(lut[2])
MULT = multipliers(1, 4, V)
RAW_V = tk.get_vocab_size(with_added_tokens=True)
PRIMES = {m: next_primes(m, K) for m in sorted(set(SIZES + HOT_SIZES))}


def keys_by_order(mat, base):
    """Exact integer key for each order: order 4 is built from an order-3 inverse to avoid overflow."""
    k2 = mat[:, 0] * base + mat[:, 1]
    k3 = k2 * base + mat[:, 2]
    _, inv = np.unique(k3, return_inverse=True)
    k4 = inv.astype(np.int64) * base + mat[:, 3]
    return {2: k2, 3: k3, 4: k4}


def coverage(freq):
    f = np.sort(freq)[::-1]
    cs = np.cumsum(f)
    U, T = len(f), cs[-1]
    out = {}
    for x in (0.001, 0.01, 0.1, 0.5):
        out[f'top_{x}'] = float(cs[max(0, math.ceil(U * x) - 1)] / T)
    out['hapax_type_share'] = float((f == 1).mean())
    out['hapax_token_share'] = float((f == 1).sum() / T)
    return out


def collisions(mix, freq, m):
    U, T = len(mix), freq.sum()
    allshared = np.ones(U, bool)
    owned_any = np.zeros(U, bool)
    owned_cnt = np.zeros(U, dtype=np.int16)
    single_t = single_w = 0.0
    sig = np.zeros(U, dtype=np.uint64)
    rng = np.random.default_rng(7)
    for p in PRIMES[m]:
        b = mix % p
        cnt = np.bincount(b, minlength=p)
        shared = cnt[b] > 1
        top = np.zeros(p, dtype=np.int64)
        np.maximum.at(top, b, freq)
        owner = freq == top[b]
        single_t += shared.mean()
        single_w += (freq * shared).sum() / T
        allshared &= shared
        owned_any |= owner
        owned_cnt += owner
        c = np.uint64(int(rng.integers(1, 2**62)) * 2 + 1)
        sig = sig * np.uint64(1099511628211) + b.astype(np.uint64) * c
    p1 = 1 - math.exp(-(U - 1) / m)
    bands = {}
    for name, lo, hi in BANDS:
        s = (freq >= lo) & (freq <= hi)
        if s.any():
            bands[name] = {'types': int(s.sum()), 'mean_heads_owned': float(owned_cnt[s].mean()),
                           'owned_none': float((~owned_any[s]).mean())}
    return {
        'M': m, 'load': U / m,
        'single_head_type': single_t / K, 'single_head_token': single_w / K, 'single_head_theory': p1,
        'all_heads_type': float(allshared.mean()), 'all_heads_token': float((freq * allshared).sum() / T),
        'all_heads_theory_independent': p1 ** K,
        'owned_none_type': float((~owned_any).mean()), 'owned_none_token': float((freq * (~owned_any)).sum() / T),
        'full_signature_duplicates': int(U - len(np.unique(sig))), 'identical_mix_duplicates': int(U - len(np.unique(mix))),
        'bands': bands,
    }


def heaps(key):
    T = len(key)
    pts = []
    for d in (32, 16, 8, 4, 2, 1):
        n = T // d
        pts.append((n, int(len(np.unique(key[:n])))))
    x, y = np.log([p[0] for p in pts[-4:]]), np.log([p[1] for p in pts[-4:]])
    b, a = np.polyfit(x, y, 1)
    pred = {}
    for name, tokens in SCALES.items():
        u = math.exp(a) * tokens ** b
        pred[name] = {'types': u}
        for m in HOT_SIZES:
            load = u / m
            p1 = 1 - math.exp(-load)
            pred[name][f'M{m}'] = {'load': load, 'single_head': p1, 'all_heads': p1 ** K}
    return {'points': pts, 'b': float(b), 'extrapolated': pred}


results = {}
store = {}
hot_sets = {}
for name in ('wiki_en', 'support_en', 'wiki_zh_tw', 'wiki_zh_cn'):
    t0 = time.time()
    ids = np.load(IN + name + '.ids.npy').astype(np.int64)
    st = np.load(IN + name + '.starts.npy')
    cid = lut[ids]
    mat_c = lookback(cid, st, 4, PAD)
    mat_r = lookback(ids, st, 4, 2)
    keys_c = keys_by_order(mat_c, V + 1)
    keys_r = keys_by_order(mat_r, RAW_V + 1)
    rolls = rolling(mat_c, MULT)
    del mat_r
    r = {'tokens': int(len(ids)), 'docs': int(len(st)), 'orders': {}}
    store[name] = {}
    split = int(len(ids) * 0.8)
    for n in (2, 3, 4):
        kc, kr = keys_c[n], keys_r[n]
        u_c, first, freq = np.unique(kc, return_index=True, return_counts=True)
        u_r = len(np.unique(kr))
        mix_all = rolls[n - 2]
        mix = mix_all[first]
        o = {'types_raw': int(u_r), 'types_compressed': int(len(u_c)),
             'type_reduction_pct': 100 * (1 - len(u_c) / u_r),
             'coverage': coverage(freq), 'heaps': heaps(kc),
             'collisions': [collisions(mix, freq, m) for m in SIZES]}
        hot = {}
        for m in HOT_SIZES:
            p = PRIMES[m][0]
            tr = np.bincount(mix_all[:split] % p, minlength=p)
            te = mix_all[split:] % p
            order = np.argsort(tr)[::-1]
            hot[m] = {}
            for fr in HOT_FRAC:
                k = max(1, int(p * fr))
                inset = np.zeros(p, bool)
                inset[order[:k]] = True
                hot[m][str(fr)] = {'rows': k, 'test_hit': float(inset[te].mean())}
                if name == 'wiki_en':
                    hot_sets[(n, m, fr)] = inset
                elif name == 'support_en' and (n, m, fr) in hot_sets:
                    hot[m][str(fr)]['hit_with_wiki_en_hotset'] = float(hot_sets[(n, m, fr)][mix_all % p].mean())
            hot[m]['rows_touched_in_train'] = int((tr > 0).sum())
        o['hot_rows'] = hot
        r['orders'][str(n)] = o
        store[name][n] = (mix, freq)
        print(name, n, 'types', len(u_c), 'secs', round(time.time() - t0), flush=True)
    results[name] = r
    json.dump(results, open(OUT + 'analysis_partial.json', 'w'), indent=1)
    del mat_c, keys_c, keys_r, rolls, ids, cid

over = {}
for n in (2, 3, 4):
    a_mix, a_f = store['wiki_zh_tw'][n]
    b_mix, b_f = store['wiki_zh_cn'][n]
    a_in = np.isin(a_mix, b_mix)
    b_in = np.isin(b_mix, a_mix)
    over[str(n)] = {'tw_types_in_cn': float(a_in.mean()), 'tw_tokens_in_cn': float((a_f * a_in).sum() / a_f.sum()),
                    'cn_types_in_tw': float(b_in.mean()), 'cn_tokens_in_tw': float((b_f * b_in).sum() / b_f.sum())}
results['zh_overlap'] = over
results['config'] = {'K': K, 'sizes': SIZES, 'V': V, 'layer_multiplier_layer': 1}
json.dump(results, open(OUT + 'analysis.json', 'w'), indent=1)
print('ANALYSIS_DONE', flush=True)
