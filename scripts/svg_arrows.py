import re,sys
p=sys.argv[1]; s=open(p,encoding='utf8').read()
def head(m):
    d=m.group(1); x=y=px=py=0
    for c,a in re.findall(r'([MHV])([-\d. ]+)',d):
        v=a.split(); px,py=x,y
        if c=='M': x,y=float(v[0]),float(v[1])
        if c=='H': x=float(v[0])
        if c=='V': y=float(v[0])
    dx,dy=(x>px)-(x<px),(y>py)-(y<py); L,W=9,4.5
    pts=f'{x},{y} {x-dx*L-dy*W},{y-dy*L-dx*W} {x-dx*L+dy*W},{y-dy*L+dx*W}'
    return m.group(0).replace(' marker-end="url(#a)"','')+f'<polygon points="{pts}" fill="#3d4654"/>'
s=re.sub(r'<path [^>]*?d="([^"]+)"[^>]*marker-end="url\(#a\)"[^>]*/>',head,s)
s=re.sub(r'<defs>.*?</defs>','',s,flags=re.S)
open(p,'w',encoding='utf8').write(s)
