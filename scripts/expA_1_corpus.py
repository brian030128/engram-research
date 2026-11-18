"""Build tokenized corpora for Experiment A. Writes token arrays and doc starts as .npy files."""
import json
import sys
import time

import numpy as np
from datasets import load_dataset

sys.path.insert(0, '/work/u4320956/engram-research/scripts')
from expA_common import R, V41, load_tok

OUT = R + 'results/expA/corpus/'
tk = load_tok(V41)


def build(name, texts, target):
    ids, starts, n, docs = [], [], 0, 0
    t0 = time.time()
    batch = []

    def flush():
        nonlocal n, docs
        for e in tk.encode_batch(batch, add_special_tokens=False):
            if len(e.ids) < 8:
                continue
            starts.append(n)
            a = np.array(e.ids, dtype=np.int32)
            ids.append(a)
            n += len(a)
            docs += 1
        batch.clear()

    for tx in texts:
        batch.append(tx)
        if len(batch) >= 256:
            flush()
            if n >= target:
                break
    if batch and n < target:
        flush()
    arr = np.concatenate(ids)[:target]
    st = np.array([s for s in starts if s < len(arr)], dtype=np.int64)
    np.save(OUT + name + '.ids.npy', arr)
    np.save(OUT + name + '.starts.npy', st)
    meta = {'name': name, 'tokens': int(len(arr)), 'docs': int(len(st)), 'seconds': round(time.time() - t0, 1)}
    print(json.dumps(meta), flush=True)
    return meta


def wiki(lang):
    ds = load_dataset('wikimedia/wikipedia', '20231101.' + lang, split='train', streaming=True)
    for row in ds:
        yield row['text']


meta = []
meta.append(build('wiki_en', wiki('en'), 30_000_000))

support = load_dataset('bitext/Bitext-customer-support-llm-chatbot-training-dataset', split='train')
meta.append(build('support_en', (r['instruction'] + '\n' + r['response'] for r in support), 30_000_000))

import opencc
to_tw, to_cn = opencc.OpenCC('s2twp'), opencc.OpenCC('t2s')
zh_docs = []
chars = 0
for tx in wiki('zh'):
    zh_docs.append(tx)
    chars += len(tx)
    if chars >= 16_000_000:
        break
print('zh chars', chars, 'docs', len(zh_docs), flush=True)
meta.append(build('wiki_zh_tw', (to_tw.convert(t) for t in zh_docs), 10_000_000))
meta.append(build('wiki_zh_cn', (to_cn.convert(t) for t in zh_docs), 10_000_000))
json.dump(meta, open(OUT + 'meta.json', 'w'), indent=1)
print('CORPUS_DONE', flush=True)
