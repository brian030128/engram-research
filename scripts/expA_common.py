"""Shared helpers for Experiment A. All logic here is our own; no downloaded code is imported.

The hash and compression logic is an independent re-implementation of the scheme described in
Engram paper section 2.2 and read from the demo and V4.1 inference/engram.py.
"""
import numpy as np
from sympy import isprime
from tokenizers import Regex, normalizers

R = '/work/u4320956/engram-research/'
V3 = R + '.cache/hf/hub/models--deepseek-ai--DeepSeek-V3/snapshots/e815299b0bcbac849fa540c768ef21845365c9eb'
V41 = R + '.cache/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/314113f8a7dadb388215396ba5fbee2c610a04d6'


def load_tok(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path + '/tokenizer.json')


def compressed_map(tk):
    sentinel = ''
    norm = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r'[ \t\r\n]+'), ' '), normalizers.Replace(Regex(r'^ $'), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, ' '),
    ])
    n = tk.get_vocab_size(with_added_tokens=True)
    key2new, lookup = {}, np.zeros(n, dtype=np.int64)
    for i in range(n):
        text = tk.decode([i], skip_special_tokens=False)
        if '�' in text:
            key = tk.id_to_token(i)
        else:
            nm = norm.normalize_str(text)
            key = nm if nm else text
        lookup[i] = key2new.setdefault(key, len(key2new))
    return lookup, len(key2new)


def multipliers(layer_id, max_n, vocab, seed=0):
    bound = max(1, (np.iinfo(np.int64).max // vocab) // 2)
    g = np.random.default_rng(seed + 10007 * layer_id)
    return g.integers(0, bound, size=(max_n,), dtype=np.int64) * 2 + 1


def next_primes(start, count, seen=None):
    seen = set() if seen is None else seen
    out, cur = [], start - 1
    for _ in range(count):
        cur += 1
        while not isprime(cur) or cur in seen:
            cur += 1
        seen.add(cur)
        out.append(cur)
    return out


def lookback(cids, starts, max_n, pad):
    """[T, max_n] matrix of x_t, x_{t-1}, ... with look-back blocked at document start."""
    T = len(cids)
    mark = np.zeros(T, dtype=np.int64)
    mark[starts] = 1
    doc = np.cumsum(mark) - 1
    offs = np.arange(T) - np.asarray(starts)[doc]
    out = np.empty((T, max_n), dtype=np.int64)
    for k in range(max_n):
        s = np.empty(T, dtype=np.int64)
        s[k:] = cids[:T - k] if k else cids
        s[:k] = pad
        s[offs < k] = pad
        out[:, k] = s
    return out


def rolling(mat, mult):
    """res[i] is the mixed value of the (i+2)-gram; each head then takes it modulo its prime."""
    r = mat[:, 0] * mult[0]
    res = []
    for i in range(1, mat.shape[1]):
        r = np.bitwise_xor(r, mat[:, i] * mult[i])
        res.append(r.copy())
    return res
