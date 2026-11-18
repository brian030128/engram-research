"""Same articles in zh-TW and zh-CN: token inflation and n-gram overlap on an aligned subset."""
import json,sys
import numpy as np
from datasets import load_dataset
import opencc
sys.path.insert(0,'/work/u4320956/engram-research/scripts')
from expA_common import *
tk=load_tok(V41); lut,V=compressed_map(tk); PAD=int(lut[2]); MULT=multipliers(1,4,V)
tw,cn=opencc.OpenCC('s2twp'),opencc.OpenCC('t2s')
docs=[];c=0
for r in load_dataset('wikimedia/wikipedia','20231101.zh',split='train',streaming=True):
    docs.append(r['text']);c+=len(r['text'])
    if c>=4_000_000:break
def run(conv):
    ids=[];st=[];n=0
    for e in tk.encode_batch([conv.convert(d) for d in docs],add_special_tokens=False):
        st.append(n);ids.append(np.array(e.ids,dtype=np.int64));n+=len(e.ids)
    keep=[i for i in range(len(st)) if len(ids[i])>0]
    a=np.concatenate(ids);s=np.array([st[i] for i in keep]);return a,s
A,sa=run(tw);B,sb=run(cn)
out={'docs':len(docs),'chars':c,'tokens_tw':int(len(A)),'tokens_cn':int(len(B)),'tw_over_cn':len(A)/len(B),'orders':{}}
ra=rolling(lookback(lut[A],sa,4,PAD),MULT);rb=rolling(lookback(lut[B],sb,4,PAD),MULT)
for i,n in enumerate((2,3,4)):
    ua,fa=np.unique(ra[i],return_counts=True);ub,fb=np.unique(rb[i],return_counts=True)
    ina=np.isin(ua,ub);inb=np.isin(ub,ua)
    out['orders'][n]={'tw_types_in_cn':float(ina.mean()),'tw_tokens_in_cn':float((fa*ina).sum()/fa.sum()),'cn_tokens_in_tw':float((fb*inb).sum()/fb.sum()),'types_tw':int(len(ua)),'types_cn':int(len(ub))}
json.dump(out,open(R+'results/expA/zh_aligned.json','w'),indent=1);print(json.dumps(out,indent=1))
