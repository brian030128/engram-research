import fitz,re,difflib,glob,os,sys
R='/work/u4320956/engram-research/'
txt={}
for n,p in [('v1','third_party/pdf/arxiv_v1.pdf'),('v2','third_party/pdf/arxiv_v2.pdf')]:
    d=fitz.open(R+p); txt[n]=[pg.get_text() for pg in d]
    open(R+'third_party/pdf/%s.fitz.txt'%n,'w').write('\n'.join(txt[n])); print(n,len(d))
def sents(ps):
    t=re.sub(r'\s+',' ',' '.join(ps)); return re.split(r'(?<=[.;:])\s',t)
a,b=sents(txt['v1']),sents(txt['v2'])
out=[]
for l in difflib.unified_diff(a,b,lineterm='',n=0):
    if l[:2] in('--','++','@@'):continue
    out.append(l)
open(R+'notes/v1_v2_sentence_diff.txt','w').write('\n'.join(out)); print('diff lines',len(out))
for f in glob.glob(R+'third_party/arxiv_src/figs/*.pdf'):
    d=fitz.open(f); pix=d[0].get_pixmap(dpi=130); pix.save(R+'notes/fig_render/'+os.path.basename(f)[:-4]+'.png')
