"""Instrument the official demo (issue #277 resource): trace shapes and compare our hash to its hash."""
import collections
import importlib.util
import json
import sys

import numpy as np
import torch
import transformers

sys.path.insert(0, '/work/u4320956/engram-research/scripts')
from expA_common import R, V3, load_tok, lookback, rolling

orig = transformers.AutoTokenizer.from_pretrained
transformers.AutoTokenizer.from_pretrained = lambda name, *a, **k: orig(V3, *a, **{**k, 'trust_remote_code': False})
spec = importlib.util.spec_from_file_location('engram_demo', R + 'third_party/repo/engram_demo_v1.py')
demo = importlib.util.module_from_spec(spec)
sys.modules['engram_demo'] = demo
spec.loader.exec_module(demo)

out = {}
torch.manual_seed(0)
eng = demo.Engram(1)
groups = collections.Counter()
for n, p in eng.named_parameters():
    groups[n.split('.')[0]] += int(p.numel())
tk = load_tok(V3)
ids = torch.tensor(tk.encode('Only Alexander the Great could tame the horse Bucephalus.').ids)[None]
H = torch.randn(1, ids.shape[1], 4, 1024)
hash_ids = torch.from_numpy(eng.hash_mapping.hash(ids)[1])
rows = eng.multi_head_embedding(hash_ids)
flat = rows.flatten(start_dim=-2)
gates = []
for i in range(4):
    key = eng.norm1[i](eng.key_projs[i](flat))
    q = eng.norm2[i](H[:, :, i, :])
    g = (key * q).sum(-1) / (1024 ** 0.5)
    gates.append(g.abs().clamp_min(1e-6).sqrt() * g.sign())
g = torch.sigmoid(torch.stack(gates, 2))
out['trace'] = {
    'param_groups': dict(groups), 'total': sum(groups.values()),
    'table_rows': int(eng.multi_head_embedding.embedding.num_embeddings),
    'head_dim': int(eng.multi_head_embedding.embedding_dim),
    'conv_weight_abs_mean_at_init': float(eng.short_conv.conv.weight.abs().mean()),
    'gate_mean_random_init': float(g.mean()),
    'shapes': {'input_ids': list(ids.shape), 'hash_ids': list(hash_ids.shape), 'rows': list(rows.shape),
               'e_t': list(flat.shape), 'hidden': list(H.shape), 'gate': list(g.shape),
               'output': list(eng(H, ids).shape)},
}
hm = eng.hash_mapping
rng = np.random.default_rng(0)
raw = rng.integers(0, tk.get_vocab_size(with_added_tokens=True), size=(4, 256))
official = hm.hash(raw)[1]
cids = hm.compressed_tokenizer(raw).reshape(-1)
mine = rolling(lookback(cids, np.arange(0, 4 * 256, 256), 3, hm.pad_id), hm.layer_multipliers[1])
ok, col = True, 0
for i in range(2):
    for j in range(8):
        ok &= bool(((mine[i] % hm.vocab_size_across_layers[1][i][j]) == official[:, :, col].reshape(-1)).all())
        col += 1
out['our_hash_equals_demo_hash'] = ok
out['demo_compressed_vocab'] = len(hm.compressed_tokenizer)
json.dump(out, open(R + 'results/demo/trace.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
